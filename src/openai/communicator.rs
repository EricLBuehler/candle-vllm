#[allow(unused_imports)]
use crate::openai::distributed::{Comm, Id};
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
use std::net::TcpStream;
use std::process::Command;

use tracing::{info, warn};
pub(crate) const DAEMON_PAYLOAD: &str = "__CANDLE_VLLM_DAEMON_INTERNAL";
use lazy_static::lazy_static;
use std::fmt;
use std::sync::Mutex;

const MAX_WIRE_MESSAGE_BYTES: usize = 1024 * 1024 * 1024;

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
        global_rank: usize,
        global_world_size: usize,
        device_id: usize,
    },
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct FlashInferHostData {
    pub indptr: Vec<u32>,
    pub indices: Vec<u32>,
    pub last_len: Vec<u32>,
    pub kv_len_arr: Vec<u32>,
    pub use_cuda_graph: bool,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ForwardPayload {
    pub input_ids: Vec<u32>,
    pub positions: Vec<i64>,
    pub slot_mapping: Vec<i64>,
    pub block_tables_flat: Vec<u32>,
    pub block_tables_rows: usize,
    pub block_tables_cols: usize,
    pub context_lens: Option<Vec<u32>>,
    pub cu_seqlens_q: Option<Vec<u32>>,
    pub cu_seqlens_k: Option<Vec<u32>>,
    pub max_seqlen_q: usize,
    pub max_seqlen_k: usize,
    pub max_context_len: usize,
    pub sequence_ids: Vec<usize>,
    pub seqlens: Option<Vec<u32>>,
    pub is_prefill: bool,
    pub is_mla: bool,
    pub flashinfer_host: Option<FlashInferHostData>,
}

#[derive(Serialize, Deserialize, Debug)]
pub enum MessageType {
    RunForward(ForwardPayload),
    FinishSequences(Vec<usize>),
    Shutdown,
    HeartBeat,
    Progress((usize, usize)),
    Close,
}

pub struct DaemonManager {
    daemon_streams: Option<Vec<LocalStream>>,
    main_stream: Option<LocalStream>,
    /// TCP streams to remote worker nodes (master only, multi-node mode)
    tcp_worker_streams: Option<Vec<TcpStream>>,
    /// TCP stream to master node (worker only, multi-node mode)
    tcp_master_stream: Option<TcpStream>,
    tcp_rank: Option<usize>,
    tcp_size: Option<usize>,
}

impl fmt::Debug for DaemonManager {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DaemonManager")
            .field("daemon_streams", &self.daemon_streams)
            .field("main_stream", &self.main_stream)
            .field("tcp_rank", &self.tcp_rank)
            .field("tcp_size", &self.tcp_size)
            .finish()
    }
}

/// Configuration for TCP-based multi-node inference (replaces MPI).
#[derive(Clone, Debug)]
pub struct MultiNodeConfig {
    pub num_nodes: usize,
    pub node_rank: usize,
    pub master_addr: String,
    pub master_port: u16,
    pub local_num_gpus: usize,
}

impl MultiNodeConfig {
    pub fn global_world_size(&self) -> usize {
        self.num_nodes * self.local_num_gpus
    }

    pub fn global_rank_offset(&self) -> usize {
        self.node_rank * self.local_num_gpus
    }

    pub fn is_master(&self) -> bool {
        self.node_rank == 0
    }

    /// Port used for forward-pass coordination TCP connections.
    /// Offset from master_port by 1 to avoid collision with NCCL ID exchange.
    pub fn forward_coord_port(&self) -> std::io::Result<u16> {
        self.master_port.checked_add(1).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "--master-port must be less than 65535 for multi-node coordination",
            )
        })
    }
}

/// Send a length-prefixed bincode message over a TcpStream.
pub fn send_tcp(stream: &mut TcpStream, data: &[u8]) -> std::io::Result<()> {
    let len = u32::try_from(data.len()).map_err(|_| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("TCP message too large: {} bytes", data.len()),
        )
    })?;
    stream.write_all(&len.to_le_bytes())?;
    stream.write_all(data)?;
    stream.flush()?;
    Ok(())
}

/// Receive a length-prefixed bincode message from a TcpStream.
pub fn recv_tcp(stream: &mut TcpStream) -> std::io::Result<Vec<u8>> {
    let mut len_buf = [0u8; 4];
    stream.read_exact(&mut len_buf)?;
    let len = u32::from_le_bytes(len_buf) as usize;
    if len > MAX_WIRE_MESSAGE_BYTES {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("TCP message length {} exceeds limit", len),
        ));
    }
    let mut buf = vec![0u8; len];
    stream.read_exact(&mut buf)?;
    Ok(buf)
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

    /// Create a TCP-based multi-node coordinator (replaces MPI).
    /// Master (node_rank=0) generates NCCL ID, listens for workers;
    /// Workers connect and receive the NCCL ID.
    #[cfg(feature = "nccl")]
    pub fn new_multi_node(config: &MultiNodeConfig) -> (Self, Id) {
        let (id, tcp_worker_streams, tcp_master_stream) = if config.is_master() {
            let id = Id::new().unwrap();
            #[cfg(target_arch = "aarch64")]
            let raw_id: &[u8; 128] = id.internal();
            #[cfg(not(target_arch = "aarch64"))]
            let raw_id: &[i8; 128] = id.internal();
            let raw_bytes: &[u8] =
                unsafe { std::slice::from_raw_parts(raw_id.as_ptr() as *const u8, 128) };

            let bind_addr = format!("{}:{}", config.master_addr, config.master_port);
            info!(
                "[Multi-Node] Master node listening on {} for {} worker node(s)...",
                bind_addr,
                config.num_nodes - 1
            );

            let listener = std::net::TcpListener::bind(&bind_addr).unwrap_or_else(|e| {
                panic!(
                    "Failed to bind multi-node coordinator at {}: {}",
                    bind_addr, e
                )
            });

            let mut worker_streams = Vec::new();
            while worker_streams.len() < config.num_nodes - 1 {
                let (mut stream, peer_addr) = listener.accept().unwrap();
                info!(
                    "[Multi-Node] Worker connected from {} ({}/{})",
                    peer_addr,
                    worker_streams.len() + 1,
                    config.num_nodes - 1
                );
                stream.write_all(raw_bytes).unwrap();
                stream.flush().unwrap();

                let mut ack = [0u8; 1];
                stream.read_exact(&mut ack).unwrap();
                assert_eq!(ack[0], 1, "Worker did not acknowledge NCCL ID");
                stream.set_nodelay(true).unwrap();
                worker_streams.push(stream);
            }

            info!(
                "[Multi-Node] All {} worker node(s) received NCCL ID",
                config.num_nodes - 1
            );
            (id, Some(worker_streams), None)
        } else {
            let addr = format!("{}:{}", config.master_addr, config.master_port);
            info!("[Multi-Node] Worker connecting to master at {}...", addr);

            let mut stream = {
                let mut attempts = 0;
                loop {
                    match TcpStream::connect(&addr) {
                        Ok(s) => break s,
                        Err(e) => {
                            attempts += 1;
                            if attempts > 120 {
                                panic!(
                                    "Failed to connect to master at {} after {} attempts: {}",
                                    addr, attempts, e
                                );
                            }
                            info!(
                                "[Multi-Node] Waiting for master at {} (attempt {})...",
                                addr, attempts
                            );
                            std::thread::sleep(std::time::Duration::from_secs(1));
                        }
                    }
                }
            };

            let mut raw_bytes = [0u8; 128];
            stream.read_exact(&mut raw_bytes).unwrap();
            stream.write_all(&[1u8]).unwrap();
            stream.flush().unwrap();
            stream.set_nodelay(true).unwrap();

            let mut arr = [0i8; 128];
            unsafe {
                std::ptr::copy_nonoverlapping(raw_bytes.as_ptr(), arr.as_mut_ptr() as *mut u8, 128);
            }

            #[cfg(not(target_arch = "aarch64"))]
            let id = Id::uninit(arr);
            #[cfg(target_arch = "aarch64")]
            let id = Id::uninit(arr.map(|b| b as u8));

            info!("[Multi-Node] Received NCCL ID from master");
            (id, None, Some(stream))
        };

        Self::set_master_rank(config.is_master());

        let dm = Self {
            daemon_streams: None,
            main_stream: None,
            tcp_worker_streams,
            tcp_master_stream,
            tcp_rank: Some(config.node_rank),
            tcp_size: Some(config.num_nodes),
        };
        (dm, id)
    }

    pub fn is_distributed(&self) -> bool {
        self.tcp_worker_streams.is_some() || self.tcp_master_stream.is_some()
    }

    pub fn is_multi_node_mode(&self) -> bool {
        self.is_distributed()
    }

    pub fn mpi_sync(&mut self) -> bool {
        if self.is_distributed() {
            if DaemonManager::is_master_rank() {
                info!("Sync multi-node processes (from master rank)!");
                self.send_message(&MessageType::HeartBeat).is_ok()
            } else {
                info!("Sync multi-node processes (from daemon rank)!");
                matches!(self.receive_message(), Ok(MessageType::HeartBeat))
            }
        } else {
            false
        }
    }

    pub fn send_local(
        streams: &mut Vec<LocalStream>,
        message: &MessageType,
    ) -> std::io::Result<()> {
        let serialized = bincode::serialize(message).map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("local IPC serialize failed: {e}"),
            )
        })?;
        for stream in streams.iter_mut() {
            let len = u32::try_from(serialized.len()).map_err(|_| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!("local IPC message too large: {} bytes", serialized.len()),
                )
            })?;
            stream.write_all(&len.to_le_bytes())?;
            stream.write_all(&serialized)?;
            stream.flush()?;
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

    /// Send a message via TCP to all remote worker nodes (master only).
    fn send_tcp_broadcast(&mut self, message: &MessageType) -> std::io::Result<()> {
        let serialized = bincode::serialize(message).map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("TCP serialize failed: {e}"),
            )
        })?;
        let streams = self
            .tcp_worker_streams
            .as_mut()
            .expect("send_tcp_broadcast called on non-master node");
        for stream in streams.iter_mut() {
            send_tcp(stream, &serialized)?;
        }
        Ok(())
    }

    /// Receive a message via TCP from master (worker only).
    fn receive_tcp_from_master(&mut self) -> std::io::Result<MessageType> {
        let stream = self
            .tcp_master_stream
            .as_mut()
            .expect("receive_tcp_from_master called on non-worker node");
        let data = recv_tcp(stream)?;
        let message: MessageType = bincode::deserialize(&data).map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("TCP deserialize failed ({} bytes): {e}", data.len()),
            )
        })?;
        Ok(message)
    }

    pub fn send_message(&mut self, message: &MessageType) -> std::io::Result<()> {
        if self.is_distributed() {
            if let Some(streams) = self.daemon_streams.as_mut() {
                DaemonManager::send_local(streams, message)?;
            }
            if self.tcp_worker_streams.is_some() {
                self.send_tcp_broadcast(message)?;
            }
            Ok(())
        } else {
            assert!(
                !DaemonManager::is_daemon(),
                "must be called in the main process!"
            );
            assert!(self.daemon_streams.is_some(), "No daemon process found!");
            let streams = self.daemon_streams.as_mut().unwrap();
            DaemonManager::send_local(streams, message)
        }
    }

    pub fn receive_local(stream: &mut LocalStream) -> std::io::Result<MessageType> {
        let mut length_buf = [0u8; 4];
        stream.read_exact(&mut length_buf)?;
        let length = u32::from_le_bytes(length_buf) as usize;
        if length > MAX_WIRE_MESSAGE_BYTES {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("local IPC message length {} exceeds limit", length),
            ));
        }

        let mut serialized = vec![0u8; length];
        stream.read_exact(&mut serialized)?;
        let message: MessageType = bincode::deserialize(&serialized).map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "local IPC deserialize failed ({} bytes): {e}",
                    serialized.len()
                ),
            )
        })?;
        stream.write_all(&[1])?;
        stream.flush()?;
        Ok(message)
    }

    pub fn receive_message(&mut self) -> std::io::Result<MessageType> {
        if self.is_distributed() {
            let message = self.receive_tcp_from_master()?;
            if let Some(streams) = self.daemon_streams.as_mut() {
                DaemonManager::send_local(streams, &message)?;
            }
            Ok(message)
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

    pub fn send_to_main(&mut self, message: &MessageType) -> std::io::Result<()> {
        assert!(
            DaemonManager::is_daemon(),
            "must be called in the daemon processes!"
        );
        assert!(
            self.main_stream.is_some(),
            "not connected to the main process!"
        );
        let stream = self.main_stream.as_mut().unwrap();
        let serialized = bincode::serialize(message).map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("local IPC serialize failed: {e}"),
            )
        })?;
        let len = u32::try_from(serialized.len()).map_err(|_| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("local IPC message too large: {} bytes", serialized.len()),
            )
        })?;
        stream.write_all(&len.to_le_bytes())?;
        stream.write_all(&serialized)?;
        stream.flush()?;
        let mut ack_buf = [0u8; 1];
        stream.read_exact(&mut ack_buf)?;
        if ack_buf[0] != 1 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Unexpected acknowledgment value from main process",
            ));
        }
        Ok(())
    }

    pub fn receive_from_daemons(&mut self) -> std::io::Result<Vec<MessageType>> {
        assert!(
            !DaemonManager::is_daemon(),
            "must be called in the main process!"
        );
        assert!(self.daemon_streams.is_some(), "No daemon process found!");
        let streams = self.daemon_streams.as_mut().unwrap();
        let mut messages = Vec::with_capacity(streams.len());
        for stream in streams.iter_mut() {
            messages.push(DaemonManager::receive_local(stream)?);
        }
        Ok(messages)
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
                tcp_worker_streams: None,
                tcp_master_stream: None,
                tcp_rank: None,
                tcp_size: None,
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
                tcp_worker_streams: None,
                tcp_master_stream: None,
                tcp_rank: None,
                tcp_size: None,
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
            let len = u32::try_from(serialized.len()).map_err(|_| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!("local IPC message too large: {} bytes", serialized.len()),
                )
            })?;
            stream.write_all(&len.to_le_bytes())?;
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

fn spawn_local_gpu_daemons(
    device_ids: &[usize],
    id: &Id,
    global_rank_offset: usize,
    global_world_size: usize,
) -> anyhow::Result<()> {
    if device_ids.len() <= 1 {
        return Ok(());
    }

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

    let _: Vec<std::process::Child> = device_ids[1..]
        .par_iter()
        .enumerate()
        .map(|(rank, dev_id)| {
            let local_rank = rank + 1;
            let exe_path = env::current_exe().expect("Failed to get current exe");
            let args: Vec<String> = env::args().collect();
            let mut cmd = if nodes_map.len() > 0 && local_rank < nodes_map.len() {
                #[cfg(any(target_os = "linux", target_os = "macos"))]
                {
                    if which::which("numactl").is_err() {
                        panic!(
                            "numactl not installed!\nInstall with: sudo apt-get install numactl"
                        );
                    } else {
                        tracing::info!("numactl found!");
                    }
                }
                let mut cmd = Command::new("numactl");
                cmd.args([
                    format!("--cpunodebind={}", nodes_map[local_rank]),
                    format!("--membind={}", nodes_map[local_rank]),
                ]);
                tracing::info!(
                    "run subprocess (local rank {}) bind to numa node {}",
                    local_rank,
                    nodes_map[local_rank]
                );
                cmd.arg(exe_path);
                cmd
            } else {
                Command::new(exe_path)
            };
            cmd.args(&args[1..]);

            let data = RankData::Init {
                id: CommID(*id.internal()),
                rank: local_rank,
                global_rank: global_rank_offset + local_rank,
                global_world_size,
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

            cmd.spawn()
                .unwrap_or_else(|_| panic!("Failed to spawn process {:?}", cmd))
        })
        .collect();

    Ok(())
}

/// Initialize multi-node subprocess coordination via TCP (replaces MPI).
/// When `multi_node_config` is Some, uses TCP-based NCCL ID exchange.
/// Otherwise uses existing single-node subprocess spawning.
#[allow(unused_variables)]
pub fn init_subprocess(
    device_ids: Vec<usize>,
    multi_node_config: Option<&MultiNodeConfig>,
) -> anyhow::Result<(Id, usize, usize, usize, DaemonManager)> {
    let local_world_size = device_ids.len();
    let (local_rank, global_rank, global_world_size, id, daemon_manager) =
        if let Ok(payload) = env::var(DAEMON_PAYLOAD) {
            let payload: RankData = serde_json::from_str(&payload)?;
            let RankData::Init {
                id: new_id,
                rank,
                global_rank,
                global_world_size,
                device_id: _,
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
                    daemon_manager =
                        DaemonManager::new_channel(DaemonManager::ipc_default_name()?, false, None);
                    continue;
                } else {
                    info!("{:?}", daemon_manager);
                    break;
                }
            }
            (
                rank,
                global_rank,
                global_world_size,
                id,
                daemon_manager.unwrap(),
            )
        } else if let Some(mn_config) = multi_node_config {
            #[cfg(not(feature = "nccl"))]
            panic!("Multi-node inference requires the `nccl` feature to be enabled!");
            #[cfg(feature = "nccl")]
            {
                info!(
                    "Running multi-node via TCP! node_rank={}, num_nodes={}, local_gpus={}",
                    mn_config.node_rank, mn_config.num_nodes, mn_config.local_num_gpus
                );
                let (daemon_manager, id) = DaemonManager::new_multi_node(mn_config);

                let local_rank = 0; // Each node's process is local_rank 0; local subprocess spawning handles ranks 1..N
                let global_rank = mn_config.global_rank_offset();
                let global_world_size = mn_config.global_world_size();

                *IS_DAEMON.lock().unwrap() = !mn_config.is_master();

                spawn_local_gpu_daemons(&device_ids, &id, global_rank, global_world_size)?;
                let mut daemon_manager = daemon_manager;
                if local_world_size > 1 {
                    let local_daemon_manager = DaemonManager::new_channel(
                        &DaemonManager::ipc_default_name()?,
                        false,
                        Some(local_world_size - 1),
                    )?;
                    daemon_manager.daemon_streams = local_daemon_manager.daemon_streams;
                    warn!("All local subprocess workers have connected to this node!");
                }

                (
                    local_rank,
                    global_rank,
                    global_world_size,
                    id,
                    daemon_manager,
                )
            }
        } else {
            info!("Running on single node!");
            {
                let id = Id::new().unwrap();
                *IS_DAEMON.lock().unwrap() = false;
                DaemonManager::set_master_rank(true);
                spawn_local_gpu_daemons(&device_ids, &id, 0, local_world_size)?;
                let daemon_manager = if local_world_size > 1 {
                    DaemonManager::new_channel(
                        &DaemonManager::ipc_default_name()?,
                        false,
                        Some(local_world_size - 1),
                    )?
                } else {
                    DaemonManager {
                        daemon_streams: Some(Vec::new()),
                        main_stream: None,
                        tcp_worker_streams: None,
                        tcp_master_stream: None,
                        tcp_rank: None,
                        tcp_size: None,
                    }
                };
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
