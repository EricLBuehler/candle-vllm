#[allow(unused_imports)]
use crate::openai::distributed::{Comm, Id};
use crate::openai::sampling_params::Logprobs;
use crate::openai::TaskData;
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

use tracing::{info, warn};
pub(crate) const DAEMON_PAYLOAD: &str = "__CANDLE_VLLM_DAEMON_INTERNAL";
use lazy_static::lazy_static;
#[cfg(not(feature = "mpi"))]
pub struct SimpleCommunicator {}
#[cfg(feature = "mpi")]
use mpi::topology::SimpleCommunicator;
#[cfg(feature = "mpi")]
use mpi::traits::*;
use std::fmt;
use std::sync::{Arc, Mutex};

pub struct ThreadSafeCommunicator(SimpleCommunicator);

unsafe impl Send for ThreadSafeCommunicator {}
unsafe impl Sync for ThreadSafeCommunicator {}

impl ThreadSafeCommunicator {
    pub fn new(comm: SimpleCommunicator) -> Self {
        Self(comm)
    }

    pub fn inner(&self) -> &SimpleCommunicator {
        &self.0
    }

    pub fn inner_mut(&mut self) -> &mut SimpleCommunicator {
        &mut self.0
    }
}

#[cfg(feature = "mpi")]
impl ThreadSafeCommunicator {
    pub fn broadcast_into<T: Equivalence>(&self, root_rank: usize, value: &mut [T]) {
        let root_process = self.0.process_at_rank(root_rank as i32);
        root_process.broadcast_into(value);
    }

    pub fn rank(&self) -> usize {
        self.0.rank() as usize
    }

    pub fn size(&self) -> usize {
        self.0.size() as usize
    }
}

#[cfg(feature = "mpi")]
impl fmt::Debug for ThreadSafeCommunicator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ThreadSafeCommunicator {{ rank: {}, size: {} }}",
            self.0.rank(),
            self.0.size()
        )
    }
}

lazy_static! {
    static ref IS_DAEMON: Mutex<bool> = Mutex::new(false);
}

lazy_static! {
    static ref IS_MASTER_RANK: Mutex<bool> = Mutex::new(false);
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
    Abort(Vec<usize>),
    Finish,
    HeartBeat,                //command channel
    Progress((usize, usize)), //command channel
    Close,
}

pub struct DaemonManager {
    daemon_streams: Option<Vec<LocalStream>>,
    main_stream: Option<LocalStream>,
    #[cfg(feature = "mpi")]
    _universe: Option<mpi::environment::Universe>,
    pub mpi_world: Option<Arc<ThreadSafeCommunicator>>, // 👈 this for multi-node
    mpi_rank: Option<i32>,                              // 👈 store rank for multi-node
    mpi_size: Option<i32>,                              // 👈 store size for multi-node
}

impl fmt::Debug for DaemonManager {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DaemonManager")
            .field("daemon_streams", &self.daemon_streams)
            .field("main_stream", &self.main_stream)
            .field("mpi_rank", &self.mpi_rank)
            .field("mpi_size", &self.mpi_size)
            .finish()
    }
}

impl DaemonManager {
    pub fn ipc_default_name() -> anyhow::Result<&'static str> {
        Ok("candle_vllm_daemon")
    }

    pub fn ipc_command_name(command_name: &str) -> anyhow::Result<String> {
        let printname = format!("command_{}_candle_vllm", command_name);
        Ok(printname)
    }

    pub fn to_channel_name(name: &str) -> anyhow::Result<Name<'static>> {
        let printname = format!("{}.sock", name);
        Ok(printname.to_ns_name::<GenericNamespaced>()?)
    }

    pub fn is_daemon() -> bool {
        *IS_DAEMON.lock().unwrap()
    }

    pub fn is_master_rank() -> bool {
        *IS_MASTER_RANK.lock().unwrap()
    }

    pub fn set_master_rank(master: bool) {
        *IS_MASTER_RANK.lock().unwrap() = master;
    }

    #[cfg(feature = "mpi")]
    pub fn new_mpi() -> Self {
        let (universe, threading) =
            mpi::initialize_with_threading(mpi::Threading::Multiple).unwrap();
        if threading != mpi::Threading::Multiple {
            panic!("MPI implementation may not support `threading::Multiple`");
        }
        let world = universe.world();
        Self::set_master_rank(world.rank() == 0);
        Self {
            daemon_streams: None,
            main_stream: None,
            mpi_rank: Some(world.rank()),
            mpi_size: Some(world.size()),
            _universe: Some(universe),
            mpi_world: Some(Arc::new(ThreadSafeCommunicator { 0: world })),
        }
    }

    pub fn is_distributed(&self) -> bool {
        self.mpi_world.is_some()
    }

    pub fn is_running_under_mpirun() -> bool {
        // Check for OpenMPI or MPICH env vars
        #[cfg(not(feature = "mpi"))]
        return false;
        #[cfg(feature = "mpi")]
        {
            std::env::var("OMPI_COMM_WORLD_SIZE").is_ok() || std::env::var("PMI_RANK").is_ok()
        }
    }

    pub fn mpi_sync(&mut self) -> bool {
        if DaemonManager::is_running_under_mpirun() && self.is_distributed() {
            //sync mpi processes across nodes
            if DaemonManager::is_master_rank() {
                info!("Sync MPI processes across nodes (from master rank)!");
                self.send_message(&MessageType::Continue).is_ok()
            } else {
                info!("Sync MPI processes across nodes (from daemon rank)!");
                match self.receive_message() {
                    Ok(MessageType::Continue) => true,
                    _ => false,
                }
            }
        } else {
            false
        }
    }

    //inter-node communication
    pub fn send_local(
        streams: &mut Vec<LocalStream>,
        message: &MessageType,
    ) -> std::io::Result<()> {
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

    //intra-node communication
    #[cfg(feature = "mpi")]
    pub fn send_mpi(&self, message: &MessageType) -> std::io::Result<()> {
        let serialized = bincode::serialize(message).expect("Serialization failed");
        let msg_len = serialized.len() as u64;

        let world = self.mpi_world.as_ref().unwrap();

        // Step 1: Broadcast length (u64)
        let mut len_buf = [0u64; 1];
        len_buf[0] = msg_len;
        world.broadcast_into(0, &mut len_buf);

        // Step 2: Broadcast message body
        let mut body_buf = serialized.clone(); // serialized Vec<u8>
        world.broadcast_into(0, &mut body_buf[..]);

        Ok(())
    }

    //inter- and intra-node communication
    pub fn send_message(&mut self, message: &MessageType) -> std::io::Result<()> {
        if self.is_distributed() {
            #[cfg(not(feature = "mpi"))]
            panic!("mpi feature is not enabled!");
            #[cfg(feature = "mpi")]
            self.send_mpi(message)
        } else {
            assert!(
                !DaemonManager::is_daemon(),
                "must be called in the main process!"
            );
            assert!(self.daemon_streams.is_some(), "No daomon process found!");
            let streams = self.daemon_streams.as_mut().unwrap();
            DaemonManager::send_local(streams, message)
        }
    }

    //inter-node communication
    pub fn receive_local(stream: &mut LocalStream) -> std::io::Result<MessageType> {
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

    //intra-node communication
    #[cfg(feature = "mpi")]
    pub fn receive_mpi(&self) -> std::io::Result<MessageType> {
        let world = self.mpi_world.as_ref().unwrap();
        // Step 1: Receive length
        let mut len_buf = [0u64; 1];
        world.broadcast_into(0, &mut len_buf);
        let msg_len = len_buf[0] as usize;

        // Step 2: Allocate buffer and receive message body
        let mut serialized = vec![0u8; msg_len];
        world.broadcast_into(0, &mut serialized[..]);

        let message: MessageType =
            bincode::deserialize(&serialized).expect("Deserialization failed");
        Ok(message)
    }

    //inter- and intra-node communication
    pub fn receive_message(&mut self) -> std::io::Result<MessageType> {
        if self.is_distributed() {
            #[cfg(not(feature = "mpi"))]
            panic!("mpi feature is not enabled!");
            #[cfg(feature = "mpi")]
            self.receive_mpi()
        } else {
            assert!(
                DaemonManager::is_daemon(),
                "must be called in the daemon processes!"
            );
            assert!(
                self.main_stream.is_some(),
                "not connected to the main process!"
            );
            DaemonManager::receive_local(self.main_stream.as_mut().unwrap())
        }
    }

    //inter-node communication
    pub fn new_command(command_name: &str, num_subprocess: Option<usize>) -> std::io::Result<Self> {
        let name = DaemonManager::ipc_command_name(command_name).unwrap();
        DaemonManager::new_channel(&name.as_str(), true, num_subprocess)
    }

    //inter-node communication
    pub fn new_channel(
        channel_name: &str,
        is_command: bool,
        num_subprocess: Option<usize>,
    ) -> std::io::Result<Self> {
        let sock_name = Self::to_channel_name(channel_name).unwrap();
        if DaemonManager::is_daemon() {
            info!(
                "connect to main process' {} channel!",
                if is_command { "command" } else { "data" }
            );
            let mut stream = LocalStream::connect(sock_name)?;
            stream.write_all(b"ready\n")?;
            warn!(
                "connected to the main process' {} channel!",
                if is_command { "command" } else { "data" }
            );
            Ok(Self {
                daemon_streams: None,
                main_stream: Some(stream),
                mpi_rank: None,
                mpi_size: None,
                mpi_world: None,
                #[cfg(feature = "mpi")]
                _universe: None,
            })
        } else {
            info!(
                "build {} channel for the main process!",
                if is_command { "command" } else { "data" }
            );
            let num_subprocess = num_subprocess.unwrap();
            let listener = ListenerOptions::new().name(sock_name).create_sync()?;
            let mut streams = Vec::with_capacity(num_subprocess);
            for _ in 0..num_subprocess {
                let stream = listener.accept()?;
                info!(
                    "accept one daemon process in {} channel!",
                    if is_command { "command" } else { "data" }
                );
                streams.push(stream);
            }

            for stream in streams.iter_mut() {
                let mut reader = BufReader::new(stream);
                let mut message = String::new();
                reader.read_line(&mut message)?;
                if message.trim() == "ready" {
                    info!(
                        "one daemon process connected to the {} channel!",
                        if is_command { "command" } else { "data" }
                    );
                }
            }
            warn!(
                "{} channel is built!",
                if is_command { "command" } else { "data" }
            );
            Ok(Self {
                daemon_streams: Some(streams),
                main_stream: None,
                mpi_rank: None,
                mpi_size: None,
                mpi_world: None,
                #[cfg(feature = "mpi")]
                _universe: None,
            })
        }
    }

    //inter-node communication
    pub fn send_command(&mut self, message: &MessageType) -> std::io::Result<()> {
        assert!(
            !DaemonManager::is_daemon(),
            "must be called in the main process!"
        );
        assert!(self.daemon_streams.is_some(), "No daomon process found!");
        let streams = self.daemon_streams.as_mut().unwrap();
        DaemonManager::send_local(streams, message)
    }

    //inter-node communication
    pub fn receive_command(&mut self) -> std::io::Result<MessageType> {
        assert!(
            DaemonManager::is_daemon(),
            "must be called in the daemon processes!"
        );
        assert!(
            self.main_stream.is_some(),
            "not connected to the main process!"
        );
        DaemonManager::receive_local(self.main_stream.as_mut().unwrap())
    }

    //This will block the callers, inter-node
    pub fn heartbeat(&mut self) -> std::io::Result<()> {
        if DaemonManager::is_daemon() {
            assert!(
                self.main_stream.is_some(),
                "Current daemon process is not connected to the main process!"
            );
            match self.receive_command() {
                Ok(MessageType::HeartBeat) => Ok(()),
                Err(e) => Err(e),
                _ => Ok(()),
            }
        } else {
            assert!(
                self.daemon_streams.is_some(),
                "No daemon processed connected to this main process!"
            );
            self.send_command(&MessageType::HeartBeat)
        }
    }

    //This will block the callers, inter-node
    pub fn progress(
        &mut self,
        progress: Option<(usize, usize)>,
    ) -> std::io::Result<Option<Vec<(usize, usize)>>> {
        if DaemonManager::is_daemon() {
            assert!(
                self.main_stream.is_some() && progress.is_some(),
                "Current daemon process is not connected to the main process!"
            );
            let stream = self.main_stream.as_mut().unwrap();
            let message = MessageType::Progress(progress.unwrap());
            let serialized = bincode::serialize(&message).expect("Serialization failed");
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
            Ok(None)
        } else {
            assert!(
                self.daemon_streams.is_some(),
                "No daemon processed connected to this main process!"
            );
            let mut ret = Vec::<(usize, usize)>::new();
            for i in 0..self.daemon_streams.as_ref().unwrap().len() {
                let streams = self.daemon_streams.as_mut().unwrap();
                if let Ok(MessageType::Progress((r, p))) =
                    DaemonManager::receive_local(&mut streams[i])
                {
                    ret.push((r, p));
                }
            }
            Ok(Some(ret))
        }
    }
}

#[allow(unused_variables)]
pub fn init_subprocess(
    device_ids: Vec<usize>,
) -> anyhow::Result<(Id, usize, usize, usize, DaemonManager)> {
    let local_world_size = device_ids.len();
    let (local_rank, global_rank, global_world_size, id, daemon_manager) =
        if DaemonManager::is_running_under_mpirun() {
            #[cfg(not(feature = "mpi"))]
            panic!("mpi feature is not enabled!");
            //multi-node
            #[cfg(feature = "mpi")]
            {
                info!("Running with MPI!");
                let daemon_manager = DaemonManager::new_mpi();
                let world = daemon_manager.mpi_world.as_ref().unwrap();
                let world_size = world.size(); // total processes
                let rank = world.rank(); // global rank
                                         // let root_process = world.process_at_rank(0);
                info!("[Rank {}] Hello from MPI!", rank);
                let mut raw_id = if rank == 0 {
                    // Rank 0 generates unique ID
                    *Id::new().unwrap().internal()
                } else {
                    [0 as ::core::ffi::c_char; 128usize]
                };

                // Broadcast the raw_id to all ranks
                world.broadcast_into(0, &mut raw_id);
                let id = Id::uninit(raw_id);
                info!("unique id: {:?}", id);
                let local_rank = rank as usize % local_world_size;
                *IS_DAEMON.lock().unwrap() = local_rank != 0;
                (
                    local_rank,
                    rank as usize,
                    world_size as usize,
                    id,
                    daemon_manager,
                )
            }
        } else {
            info!("Running on single node!");
            //single node
            if let Ok(payload) = env::var(DAEMON_PAYLOAD) {
                let payload: RankData = serde_json::from_str(&payload)?;
                let RankData::Init {
                    id: new_id,
                    rank,
                    device_id,
                } = payload;
                let id = Id::uninit(new_id.0);
                *IS_DAEMON.lock().unwrap() = true;
                let mut daemon_manager =
                    DaemonManager::new_channel(&DaemonManager::ipc_default_name()?, false, None);
                let mut connect_retry_count = 0;
                loop {
                    if daemon_manager.is_ok() {
                        info!("Connected to the main process!");
                        break;
                    } else if connect_retry_count < 60 {
                        connect_retry_count += 1;
                        warn!(
                            "Retry connect to main process' data channel ({:?})!",
                            daemon_manager
                        );
                        let _ = std::thread::sleep(std::time::Duration::from_millis(1000 as u64));
                        daemon_manager = DaemonManager::new_channel(
                            DaemonManager::ipc_default_name()?,
                            false,
                            None,
                        );
                        continue;
                    } else {
                        info!("{:?}", daemon_manager);
                        break;
                    }
                }
                (rank, rank, local_world_size, id, daemon_manager.unwrap())
            } else {
                let id = Id::new().unwrap();
                use rayon::iter::IndexedParallelIterator;
                use rayon::iter::IntoParallelRefIterator;
                use rayon::iter::ParallelIterator;

                let nodes_map: Vec<u32> = if let Ok(p) = std::env::var("MAP_NUMA_NODE") {
                    p.split(',')
                        .filter_map(|s| s.trim().parse::<u32>().ok())
                        .collect()
                } else {
                    vec![]
                };

                *IS_DAEMON.lock().unwrap() = false;
                DaemonManager::set_master_rank(true);
                let _: Vec<std::process::Child> = device_ids[1..]
                    .par_iter()
                    .enumerate()
                    .map(|(rank, dev_id)| {
                        let exe_path = env::current_exe().expect("Failed to get current exe");
                        let args: Vec<String> = env::args().collect();
                        let mut cmd = if nodes_map.len() > 0 && rank + 1 < nodes_map.len() {
                            #[cfg(any(target_os = "linux", target_os = "macos"))]
                            {
                                if which::which("numactl").is_err() {
                                    panic!("numactl not installed!\nInstall with: sudo apt-get install numactl");
                                } else {
                                    tracing::info!("numactl found!");
                                }
                            }
                            let mut cmd = Command::new("numactl");
                            cmd.args([
                                format!("--cpunodebind={}", nodes_map[rank + 1]),
                                format!("--membind={}", nodes_map[rank + 1]),
                            ]);
                            tracing::info!(
                                "run subprocess (rank {}) bind to numa node {}",
                                rank + 1,
                                nodes_map[rank + 1]
                            );
                            cmd.arg(exe_path);
                            cmd
                        } else {
                            Command::new(exe_path)
                        };
                        cmd.args(&args[1..]);

                        let data = RankData::Init {
                            id: CommID(*id.internal()),
                            rank: rank + 1,
                            device_id: *dev_id,
                        };

                        cmd.env(DAEMON_PAYLOAD, serde_json::to_string(&data).unwrap());
                        cmd.env(
                            "RUST_LOG",
                            std::env::var("RUST_LOG").unwrap_or("warn".to_string()),
                        );

                        cmd.stdout(std::process::Stdio::null());
                        cmd.stderr(std::process::Stdio::null());
                        cmd.stdin(std::process::Stdio::null());

                        cmd.spawn().expect(format!("Failed to spawn process {:?}", cmd).as_str())
                    })
                    .collect();
                let daemon_manager = DaemonManager::new_channel(
                    &DaemonManager::ipc_default_name()?,
                    false,
                    Some(local_world_size - 1),
                )?;
                warn!("All subprocess workers have connected to the main processes!");
                (0, 0, local_world_size, id, daemon_manager)
            }
        };

    info!(
        "local_rank {}, global_rank {}, local_world_size {}, global_world_size {}",
        local_rank, global_rank, local_world_size, global_world_size,
    );

    Ok((
        id,
        local_rank,
        global_rank,
        global_world_size,
        daemon_manager,
    ))
}
