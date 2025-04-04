#[allow(unused_imports)]
use crate::openai::distributed::{Comm, Id};
use crate::openai::sampling_params::{Logprobs, SamplingParams};
use bincode;
use core::ffi::c_char;
use interprocess::local_socket::traits::{Listener, Stream};
use interprocess::local_socket::{GenericNamespaced, Name, ToNsName};
use interprocess::local_socket::{ListenerOptions, Stream as LocalStream};
use serde::{Deserialize, Serialize};
use serde_big_array::BigArray;
use std::env;
use std::io::Read;
use std::io::{BufRead, BufReader, Write};
use std::process::Command;
use std::time::SystemTime;
use tokenizers::Encoding;
use tracing::{info, warn};
pub(crate) const DAEMON_PAYLOAD: &str = "__CANDLE_VLLM_DAEMON_INTERNAL";
use lazy_static::lazy_static;
use std::sync::Mutex;

lazy_static! {
    static ref IS_DAEMON: Mutex<bool> = Mutex::new(false);
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(transparent)]
pub struct CommID(#[serde(with = "BigArray")] pub [c_char; 128]);

#[derive(Serialize, Deserialize, Debug)]
pub enum RankData {
    Init {
        id: CommID,
        rank: usize,
        device_id: usize,
    },
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct TaskData {
    pub prompt: Encoding,
    pub request_id: String,
    pub created: SystemTime,
    pub sampling_params: SamplingParams,
    pub use_logprobs: bool,
}

#[derive(Serialize, Deserialize, Debug)]
pub enum TaskSampleData {
    Token(Logprobs),
    StopReason(String),
}

#[derive(Serialize, Deserialize, Debug)]
pub enum MessageType {
    Start,
    Data(Vec<TaskData>),
    Sample(Vec<TaskSampleData>),
    Continue,
    Finish,
    HeartBeat,
    Close,
}

#[derive(Debug)]
pub struct DaemonManager {
    daemon_streams: Option<Vec<LocalStream>>,
    main_stream: Option<LocalStream>,
    daemon_streams_command: Option<Vec<LocalStream>>,
    main_stream_command: Option<LocalStream>,
}

impl DaemonManager {
    pub fn ipc_default_name() -> anyhow::Result<Name<'static>> {
        let printname = "candle_vllm_daemon.sock";
        Ok(printname.to_ns_name::<GenericNamespaced>()?)
    }

    pub fn ipc_command_name(command_name: &str) -> anyhow::Result<Name<'static>> {
        let printname = format!("command_{}_candle_vllm.sock", command_name);
        Ok(printname.to_ns_name::<GenericNamespaced>()?)
    }

    pub fn is_daemon() -> bool {
        *IS_DAEMON.lock().unwrap()
    }

    //called by main process
    pub fn new(num_subprocess: usize) -> anyhow::Result<Self> {
        let name = DaemonManager::ipc_default_name()?;
        *IS_DAEMON.lock().unwrap() = false;
        let listener = ListenerOptions::new().name(name).create_sync()?;
        let mut streams = Vec::with_capacity(num_subprocess);
        for _ in 0..num_subprocess {
            let stream = listener.accept()?;
            streams.push(stream);
        }

        for stream in streams.iter_mut() {
            let mut reader = BufReader::new(stream);
            let mut message = String::new();
            reader.read_line(&mut message)?;
            if message.trim() == "ready" {
                info!("one daemon process connected!");
            }
        }
        Ok(Self {
            daemon_streams: Some(streams),
            main_stream: None,
            daemon_streams_command: None,
            main_stream_command: None,
        })
    }

    //called by daemon processes
    pub fn connect() -> anyhow::Result<Self> {
        *IS_DAEMON.lock().unwrap() = true;
        let name = DaemonManager::ipc_default_name()?;
        let mut stream = LocalStream::connect(name)?;
        stream.write_all(b"ready\n")?;
        Ok(Self {
            daemon_streams: None,
            main_stream: Some(stream),
            daemon_streams_command: None,
            main_stream_command: None,
        })
    }

    pub fn send(streams: &mut Vec<LocalStream>, message: &MessageType) -> std::io::Result<()> {
        let serialized = bincode::serialize(message).expect("Serialization failed");
        for stream in streams.iter_mut() {
            stream.write_all(&(serialized.len() as u32).to_le_bytes())?;
            stream.write_all(&serialized)?;
            stream.flush()?; // Ensure data is sent immediately
                             // Wait for acknowledgment
            let mut ack_buf = [0u8; 1];
            if let Err(e) = stream.read_exact(&mut ack_buf) {
                info!(
                    "Timeout waiting for acknowledgment from subprocess: {:?}",
                    e
                );
            } else if ack_buf[0] != 1 {
                info!("Unexpected acknowledgment value from subprocess");
            }
        }
        Ok(())
    }

    pub fn send_message(&mut self, message: &MessageType) -> std::io::Result<()> {
        assert!(
            !DaemonManager::is_daemon(),
            "must be called in the main process!"
        );
        assert!(self.daemon_streams.is_some(), "No daomon process found!");
        let streams = self.daemon_streams.as_mut().unwrap();
        DaemonManager::send(streams, message)
    }

    pub fn receive(stream: &mut LocalStream) -> std::io::Result<MessageType> {
        let mut length_buf = [0u8; 4];
        stream.read_exact(&mut length_buf)?;
        let length = u32::from_le_bytes(length_buf) as usize;

        let mut serialized = vec![0u8; length];
        stream.read_exact(&mut serialized)?;
        let message: MessageType =
            bincode::deserialize(&serialized).expect("Deserialization failed");
        // Send acknowledgment
        stream.write_all(&[1])?;
        stream.flush()?;
        Ok(message)
    }

    pub fn receive_message(&mut self) -> std::io::Result<MessageType> {
        assert!(
            DaemonManager::is_daemon(),
            "must be called in the daemon processes!"
        );
        assert!(
            self.main_stream.is_some(),
            "not connected to the main process!"
        );
        DaemonManager::receive(self.main_stream.as_mut().unwrap())
    }

    pub fn new_command(command_name: &str, num_subprocess: Option<usize>) -> std::io::Result<Self> {
        let name = DaemonManager::ipc_command_name(command_name).unwrap();
        if DaemonManager::is_daemon() {
            warn!("connect to main process' command channel!");
            let mut stream = LocalStream::connect(name)?;
            stream.write_all(b"ready\n")?;
            warn!("connected to the main process' command channel!");
            Ok(Self {
                daemon_streams: None,
                main_stream: None,
                daemon_streams_command: None,
                main_stream_command: Some(stream),
            })
        } else {
            warn!("build command channel for the main process!");
            let num_subprocess = num_subprocess.unwrap();
            let listener = ListenerOptions::new().name(name).create_sync()?;
            let mut streams = Vec::with_capacity(num_subprocess);
            for _ in 0..num_subprocess {
                let stream = listener.accept()?;
                warn!("accept one daemon process!");
                streams.push(stream);
            }

            for stream in streams.iter_mut() {
                let mut reader = BufReader::new(stream);
                let mut message = String::new();
                reader.read_line(&mut message)?;
                if message.trim() == "ready" {
                    warn!("one daemon process connected!");
                }
            }
            warn!("command channel is built!");
            Ok(Self {
                daemon_streams: None,
                main_stream: None,
                daemon_streams_command: Some(streams),
                main_stream_command: None,
            })
        }
    }

    pub fn send_command(&mut self, message: &MessageType) -> std::io::Result<()> {
        assert!(
            !DaemonManager::is_daemon(),
            "must be called in the main process!"
        );
        assert!(
            self.daemon_streams_command.is_some(),
            "No daomon process found!"
        );
        let streams = self.daemon_streams_command.as_mut().unwrap();
        DaemonManager::send(streams, message)
    }

    pub fn receive_command(&mut self) -> std::io::Result<MessageType> {
        assert!(
            DaemonManager::is_daemon(),
            "must be called in the daemon processes!"
        );
        assert!(
            self.main_stream_command.is_some(),
            "not connected to the main process!"
        );
        DaemonManager::receive(self.main_stream_command.as_mut().unwrap())
    }

    //This will block the callers
    pub fn heartbeat(&mut self) -> std::io::Result<()> {
        if DaemonManager::is_daemon() {
            assert!(
                self.main_stream_command.is_some(),
                "Current daemon process is not connected to the main process!"
            );
            match self.receive_command() {
                Ok(MessageType::HeartBeat) => Ok(()),
                Err(e) => Err(e),
                _ => Ok(()),
            }
        } else {
            assert!(
                self.daemon_streams_command.is_some(),
                "No daemon processed connected to this main process!"
            );
            self.send_command(&MessageType::HeartBeat)
        }
    }
}

pub fn init_subprocess(device_ids: Vec<usize>) -> anyhow::Result<(Id, usize, DaemonManager)> {
    let (id, local_rank, daemon_manager) = if let Ok(payload) = env::var(DAEMON_PAYLOAD) {
        let payload: RankData = serde_json::from_str(&payload)?;
        let RankData::Init {
            id: new_id,
            rank,
            device_id,
        } = payload;
        let id = Id::uninit(new_id.0);

        let daemon_manager = DaemonManager::connect()?;
        warn!("Connected to the main process!");
        (id, rank, daemon_manager)
    } else {
        let id = Id::new().unwrap();
        use rayon::iter::IndexedParallelIterator;
        use rayon::iter::IntoParallelRefIterator;
        use rayon::iter::ParallelIterator;

        let _: Vec<std::process::Child> = device_ids[1..]
            .par_iter()
            .enumerate()
            .map(|(rank, dev_id)| {
                let exe_path = env::current_exe().expect("Failed to get current exe");
                let args: Vec<String> = env::args().collect();
                let mut cmd = Command::new(exe_path);
                cmd.args(&args[1..]);

                let data = RankData::Init {
                    id: CommID(*id.internal()),
                    rank: rank + 1,
                    device_id: *dev_id,
                };

                cmd.env(DAEMON_PAYLOAD, serde_json::to_string(&data).unwrap());
                cmd.env("RUST_LOG", "info,warn");

                cmd.stdout(std::process::Stdio::null());
                cmd.stderr(std::process::Stdio::null());
                cmd.stdin(std::process::Stdio::null());

                cmd.spawn().expect("Failed to spawn process")
            })
            .collect();
        let daemon_manager = DaemonManager::new(device_ids.len() - 1)?;
        warn!("All workers have received the ids!");
        (id, 0, daemon_manager)
    };
    Ok((id, local_rank, daemon_manager))
}
