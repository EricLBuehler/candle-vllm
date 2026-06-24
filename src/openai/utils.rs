use std::net::{SocketAddr, ToSocketAddrs};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use candle_core::quantized::gguf_file;
use candle_core::{DType, Device, Result, Tensor};

use super::models::qwen3_vl::config::VisionConfig;
use super::models::Config;

pub(crate) fn get_created_time_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("Time travel has occurred...")
        .as_secs()
}

/// Fail fast if `host:port` is already bound, before spending minutes loading
/// model weights.  Prints a user-friendly error and exits the process when the
/// port is occupied.
pub fn ensure_port_free(host: &str, port: u16) {
    let addr = (host, port)
        .to_socket_addrs()
        .ok()
        .and_then(|mut addrs| addrs.next())
        .map(|addr| addr.to_string())
        .unwrap_or_else(|| format!("{host}:{port}"));
    match std::net::TcpListener::bind(&addr) {
        Ok(_listener) => { /* port is free; drop the listener immediately */ }
        Err(e) => {
            eprintln!(
                "\n❌ Port {port} is already in use ({e}).\n   \
                 Free the port or choose a different one with --p <port>.\n"
            );
            std::process::exit(1);
        }
    }
}

#[derive(Clone, Debug)]
pub enum ServerBindAddr {
    Tcp(SocketAddr),
    Unix(PathBuf),
}

pub enum ApiListener {
    Tcp(tokio::net::TcpListener),
    Unix(tokio::net::UnixListener),
}

fn tcp_authority_has_port(authority: &str) -> bool {
    if authority.starts_with('[') {
        return authority
            .rsplit_once("]:")
            .is_some_and(|(_, port)| port.parse::<u16>().is_ok());
    }

    if authority.matches(':').count() != 1 {
        return false;
    }

    let Some((host, port)) = authority.rsplit_once(':') else {
        return false;
    };
    !host.is_empty() && port.parse::<u16>().is_ok()
}

fn tcp_host_without_port(authority: &str) -> &str {
    authority
        .strip_prefix('[')
        .and_then(|host| host.strip_suffix(']'))
        .unwrap_or(authority)
}

pub fn resolve_server_bind_addr(host: &str, port: u16) -> Result<ServerBindAddr> {
    for prefix in ["file://", "socket://", "unix://"] {
        if let Some(path) = host.strip_prefix(prefix) {
            if path.is_empty() {
                candle_core::bail!("Unix socket path cannot be empty.");
            }
            return Ok(ServerBindAddr::Unix(PathBuf::from(path)));
        }
    }

    let authority = host
        .strip_prefix("tcp://")
        .or_else(|| host.strip_prefix("http://"))
        .unwrap_or(host);

    let sock_addr = if tcp_authority_has_port(authority) {
        authority
            .to_socket_addrs()
            .map_err(|e| candle_core::Error::msg(format!("Failed to resolve {authority}: {e}")))?
            .next()
    } else {
        let host = tcp_host_without_port(authority);
        (host, port)
            .to_socket_addrs()
            .map_err(|e| candle_core::Error::msg(format!("Failed to resolve {authority}: {e}")))?
            .next()
    }
    .ok_or_else(|| candle_core::Error::msg(format!("No addresses resolved for {authority}")))?;

    Ok(ServerBindAddr::Tcp(sock_addr))
}

fn ensure_unix_socket_available(path: &Path) -> Result<()> {
    let listener = std::os::unix::net::UnixListener::bind(path).map_err(|e| {
        candle_core::Error::msg(format!(
            "Unix socket {} is not available ({e}).",
            path.display()
        ))
    })?;
    drop(listener);
    std::fs::remove_file(path).map_err(|e| {
        candle_core::Error::msg(format!(
            "Failed to clean up temporary Unix socket check at {} ({e}).",
            path.display()
        ))
    })
}

pub fn ensure_server_bindings_or_exit(addr: &ServerBindAddr, with_ui_server: bool) -> Result<()> {
    match addr {
        ServerBindAddr::Tcp(sock_addr) => {
            ensure_port_free(&sock_addr.ip().to_string(), sock_addr.port());
        }
        ServerBindAddr::Unix(path) => {
            if with_ui_server {
                candle_core::bail!("--ui-server is not supported with Unix sockets.");
            }
            ensure_unix_socket_available(path)?;
        }
    }

    if with_ui_server {
        let ServerBindAddr::Tcp(sock_addr) = addr else {
            candle_core::bail!("--ui-server is not supported with Unix sockets.");
        };
        let ui_port = sock_addr.port().checked_sub(1).ok_or_else(|| {
            candle_core::Error::msg(
                "Cannot start UI server because API port 0 has no preceding UI port.",
            )
        })?;
        ensure_port_free("0.0.0.0", ui_port);
    }

    Ok(())
}

pub fn bind_addr_for_rank(addr: &ServerBindAddr, global_rank: usize) -> ServerBindAddr {
    match addr {
        // Non-zero ranks still bind a distinct local endpoint, matching the
        // previous `port += global_rank` behavior while also supporting Unix sockets.
        ServerBindAddr::Tcp(sock_addr) if global_rank > 0 => ServerBindAddr::Tcp(SocketAddr::new(
            sock_addr.ip(),
            sock_addr.port().saturating_add(global_rank as u16),
        )),
        ServerBindAddr::Unix(path) if global_rank > 0 => ServerBindAddr::Unix(PathBuf::from(
            format!("{}.rank{}", path.display(), global_rank),
        )),
        _ => addr.clone(),
    }
}

pub fn tcp_api_url(addr: SocketAddr) -> String {
    format!("http://{addr}/v1/")
}

pub async fn bind_api_listener(addr: &ServerBindAddr) -> Result<ApiListener> {
    match addr {
        ServerBindAddr::Tcp(sock_addr) => tokio::net::TcpListener::bind(sock_addr)
            .await
            .map(ApiListener::Tcp)
            .map_err(|e| {
                candle_core::Error::msg(format!("Failed to bind API server to {sock_addr}: {e}"))
            }),
        ServerBindAddr::Unix(path) => tokio::net::UnixListener::bind(path)
            .map(ApiListener::Unix)
            .map_err(|e| {
                candle_core::Error::msg(format!(
                    "Failed to bind API server to {}: {e}",
                    path.display()
                ))
            }),
    }
}

// ---------------------------------------------------------------------------
// Tokenizer helpers
// ---------------------------------------------------------------------------

pub fn tokenizer_token_id(tokenizer: &tokenizers::Tokenizer, token: &str) -> Result<u32> {
    tokenizer
        .get_vocab(true)
        .get(token)
        .copied()
        .ok_or_else(|| candle_core::Error::Msg(format!("missing multimodal token `{token}`")))
}

// ---------------------------------------------------------------------------
// GGUF vision-tower helpers
// ---------------------------------------------------------------------------

fn qwen3_vl_deepstack_indexes(ct: &gguf_file::Content) -> Result<Vec<usize>> {
    let values = ct
        .metadata
        .get("clip.vision.is_deepstack_layers")
        .and_then(|v| v.to_vec().ok())
        .cloned();
    Ok(values
        .unwrap_or_default()
        .into_iter()
        .enumerate()
        .filter_map(|(idx, v)| match v.to_bool() {
            Ok(true) => Some(idx),
            _ => None,
        })
        .collect())
}

pub fn build_vision_config_from_gguf(mmproj_ct: &gguf_file::Content) -> Result<VisionConfig> {
    let md_get = |s: &str| match mmproj_ct.metadata.get(s) {
        None => candle_core::bail!("cannot find {s} in mmproj GGUF metadata"),
        Some(v) => Ok(v),
    };
    let patch_size = md_get("clip.vision.patch_size")?.to_u32()? as usize;
    let image_size = md_get("clip.vision.image_size")?.to_u32()? as usize;

    Ok(VisionConfig {
        depth: md_get("clip.vision.block_count")?.to_u32()? as usize,
        hidden_size: md_get("clip.vision.embedding_length")?.to_u32()? as usize,
        out_hidden_size: md_get("clip.vision.projection_dim")?.to_u32()? as usize,
        hidden_act: if md_get("clip.use_gelu")
            .and_then(|v| v.to_bool())
            .unwrap_or(true)
        {
            candle_nn::Activation::Gelu
        } else {
            candle_nn::Activation::Silu
        },
        intermediate_size: md_get("clip.vision.feed_forward_length")?.to_u32()? as usize,
        num_heads: md_get("clip.vision.attention.head_count")?.to_u32()? as usize,
        in_chans: 3,
        patch_size,
        spatial_merge_size: md_get("clip.vision.spatial_merge_size")?.to_u32()? as usize,
        temporal_patch_size: 2,
        num_position_embeddings: (image_size / patch_size).pow(2),
        deepstack_visual_indexes: qwen3_vl_deepstack_indexes(mmproj_ct)?,
    })
}

fn map_gguf_vision_tensor_name(gguf_name: &str) -> Option<String> {
    if let Some(rest) = gguf_name.strip_prefix("v.blk.") {
        let dot = rest.find('.')?;
        let idx = &rest[..dot];
        let suffix = &rest[dot + 1..];
        let mapped_suffix = match suffix {
            "attn_qkv.weight" => "attn.qkv.weight",
            "attn_qkv.bias" => "attn.qkv.bias",
            "attn_out.weight" => "attn.proj.weight",
            "attn_out.bias" => "attn.proj.bias",
            "ffn_up.weight" => "mlp.linear_fc1.weight",
            "ffn_up.bias" => "mlp.linear_fc1.bias",
            "ffn_down.weight" => "mlp.linear_fc2.weight",
            "ffn_down.bias" => "mlp.linear_fc2.bias",
            "ln1.weight" => "norm1.weight",
            "ln1.bias" => "norm1.bias",
            "ln2.weight" => "norm2.weight",
            "ln2.bias" => "norm2.bias",
            _ => return None,
        };
        Some(format!("v.blocks.{idx}.{mapped_suffix}"))
    } else {
        match gguf_name {
            "v.post_ln.weight" => Some("v.merger.norm.weight".to_string()),
            "v.post_ln.bias" => Some("v.merger.norm.bias".to_string()),
            "mm.0.weight" => Some("v.merger.linear_fc1.weight".to_string()),
            "mm.0.bias" => Some("v.merger.linear_fc1.bias".to_string()),
            "mm.2.weight" => Some("v.merger.linear_fc2.weight".to_string()),
            "mm.2.bias" => Some("v.merger.linear_fc2.bias".to_string()),
            "v.position_embd.weight" => Some("v.pos_embed.weight".to_string()),
            "v.patch_embd.bias" => Some("v.patch_embed.proj.bias".to_string()),
            "v.patch_embd.weight" | "v.patch_embd.weight.1" => None,
            _ => {
                if gguf_name.starts_with("v.ds_merger.") {
                    let rest = gguf_name.strip_prefix("v.ds_merger.")?;
                    let dot = rest.find('.')?;
                    let idx = &rest[..dot];
                    let suffix = &rest[dot + 1..];
                    let mapped_suffix = match suffix {
                        "norm.weight" => "norm.weight",
                        "norm.bias" => "norm.bias",
                        "fc1.weight" => "linear_fc1.weight",
                        "fc1.bias" => "linear_fc1.bias",
                        "fc2.weight" => "linear_fc2.weight",
                        "fc2.bias" => "linear_fc2.bias",
                        _ => return None,
                    };
                    Some(format!("v.deepstack_merger_list.{idx}.{mapped_suffix}"))
                } else {
                    None
                }
            }
        }
    }
}

pub fn load_gguf_vision_tensors<R: std::io::Seek + std::io::Read>(
    ct: &gguf_file::Content,
    reader: &mut R,
    _device: &Device,
) -> Result<std::collections::HashMap<String, Tensor>> {
    let cpu = Device::Cpu;
    let mut tensors = std::collections::HashMap::new();
    let mut patch_w1: Option<Tensor> = None;
    let mut patch_w2: Option<Tensor> = None;

    for (name, _) in ct.tensor_infos.iter() {
        let qtensor = ct.tensor(reader, name, &cpu)?;
        let tensor = qtensor.dequantize(&cpu)?;

        if name == "v.patch_embd.weight" {
            patch_w1 = Some(tensor);
            continue;
        }
        if name == "v.patch_embd.weight.1" {
            patch_w2 = Some(tensor);
            continue;
        }

        if let Some(mapped_name) = map_gguf_vision_tensor_name(name) {
            tensors.insert(mapped_name, tensor);
        } else {
            tracing::debug!("Skipping unmapped GGUF vision tensor: {name}");
        }
    }

    if let (Some(w1), Some(w2)) = (patch_w1, patch_w2) {
        let conv_weight = Tensor::stack(&[w1, w2], 2)?;
        tensors.insert("v.patch_embed.proj.weight".to_string(), conv_weight);
    }

    Ok(tensors)
}

/// Build a `ShardedVarBuilder` from in-memory tensors without touching disk.
///
/// On Linux, uses `memfd_create` to create an anonymous in-memory file, serializes
/// the tensors in safetensors wire format into it, then mmaps it via `/proc/self/fd/N`.
/// The memfd lives in RAM only -- no disk I/O occurs.
/// Falls back to `/dev/shm` (tmpfs) or the OS temp dir on other platforms.
pub fn gguf_vision_tensors_to_vb(
    tensors: std::collections::HashMap<String, Tensor>,
    dtype: DType,
    device: &Device,
) -> Result<candle_nn::var_builder::ShardedVarBuilder<'static>> {
    let data: Vec<(&str, Tensor)> = tensors
        .iter()
        .map(|(k, v)| (k.as_str(), v.clone()))
        .collect();
    let bytes = safetensors::tensor::serialize(data, None)
        .map_err(|e| candle_core::Error::Msg(format!("safetensors serialize: {e}")))?;

    let path = write_to_memfd_or_tmpfs(&bytes)?;

    let vb = unsafe {
        candle_nn::var_builder::ShardedSafeTensors::var_builder(&[&path], dtype, device)?
    };
    Ok(vb)
}

/// Write bytes to an anonymous in-memory file (memfd on Linux) and return a path
/// suitable for `File::open` / mmap.  No disk I/O on Linux.
fn write_to_memfd_or_tmpfs(bytes: &[u8]) -> Result<std::path::PathBuf> {
    use std::io::Write;

    #[cfg(target_os = "linux")]
    {
        use std::os::unix::io::FromRawFd;
        let name = std::ffi::CString::new("candle_vllm_mmproj").unwrap();
        let fd = unsafe { libc::memfd_create(name.as_ptr(), 0) };
        if fd >= 0 {
            let mut file = unsafe { std::fs::File::from_raw_fd(fd) };
            file.write_all(bytes).map_err(candle_core::Error::wrap)?;
            let path = std::path::PathBuf::from(format!("/proc/self/fd/{fd}"));
            // Intentionally leak the fd so the memfd stays alive for mmap.
            // This is a one-time cost during model loading.
            std::mem::forget(file);
            return Ok(path);
        }
        tracing::debug!("memfd_create failed, falling back to tmpfs");
    }

    let dir = if std::path::Path::new("/dev/shm").exists() {
        std::path::PathBuf::from("/dev/shm")
    } else {
        std::env::temp_dir()
    };
    let path = dir.join(format!(
        "candle_vllm_mmproj_{}.safetensors",
        std::process::id()
    ));
    let mut f = std::fs::File::create(&path).map_err(candle_core::Error::wrap)?;
    f.write_all(bytes).map_err(candle_core::Error::wrap)?;
    Ok(path)
}

// ---------------------------------------------------------------------------
// Multimodal config helpers
// ---------------------------------------------------------------------------

pub fn build_multimodal_extra_config(
    base_cfg: &Config,
    vision_cfg: &VisionConfig,
    image_token_id: u32,
) -> Result<String> {
    let mut root = if let Some(existing) = base_cfg.extra_config_json.as_ref() {
        serde_json::from_str::<serde_json::Value>(existing).unwrap_or(serde_json::json!({}))
    } else {
        serde_json::json!({})
    };

    root["vision_config"] = serde_json::json!({
        "spatial_merge_size": vision_cfg.spatial_merge_size,
        "temporal_patch_size": vision_cfg.temporal_patch_size,
        "patch_size": vision_cfg.patch_size,
    });
    root["image_token_id"] = serde_json::json!(image_token_id);

    serde_json::to_string(&root).map_err(|e| candle_core::Error::Msg(e.to_string()))
}
