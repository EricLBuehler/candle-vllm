use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{Device, Result, Shape};
use std::fs::File;
use std::sync::Arc;
use std::sync::Mutex;

struct GgufShard {
    content: candle_core::quantized::gguf_file::Content,
    file: File,
}

#[derive(Clone)]
pub struct VarBuilder {
    shards: Arc<Mutex<Vec<GgufShard>>>,
    tensor_to_shard: Arc<std::collections::HashMap<String, usize>>,
    cache: Arc<Mutex<Option<(String, Arc<QTensor>)>>>,
    path: Vec<String>,
    device: Device,
    file_path: Arc<String>,
}

impl VarBuilder {
    pub fn from_gguf<P: AsRef<std::path::Path>>(p: P, device: &Device) -> Result<Self> {
        Self::from_gguf_files(&[p.as_ref().to_path_buf()], device)
    }

    pub fn from_gguf_files(paths: &[std::path::PathBuf], device: &Device) -> Result<Self> {
        assert!(!paths.is_empty(), "No GGUF files provided!");
        let file_path = paths[0].to_string_lossy().to_string();
        let mut shards = Vec::with_capacity(paths.len());
        let mut tensor_to_shard = std::collections::HashMap::new();

        for (shard_idx, path) in paths.iter().enumerate() {
            let mut file = File::open(path)?;
            let content = candle_core::quantized::gguf_file::Content::read(&mut file)?;
            for name in content.tensor_infos.keys() {
                tensor_to_shard.insert(name.clone(), shard_idx);
            }
            shards.push(GgufShard { content, file });
        }

        if paths.len() > 1 {
            tracing::info!(
                "Loaded {} GGUF shards with {} total tensors",
                paths.len(),
                tensor_to_shard.len()
            );
        }

        Ok(Self {
            shards: Arc::new(Mutex::new(shards)),
            tensor_to_shard: Arc::new(tensor_to_shard),
            cache: Arc::new(Mutex::new(None)),
            path: Vec::new(),
            device: device.clone(),
            file_path: Arc::new(file_path),
        })
    }

    pub fn gguf_path(&self) -> &str {
        &self.file_path
    }

    pub fn pp<S: ToString>(&self, s: S) -> Self {
        let mut path = self.path.clone();
        path.push(s.to_string());
        Self {
            shards: self.shards.clone(),
            tensor_to_shard: self.tensor_to_shard.clone(),
            cache: self.cache.clone(),
            path,
            device: self.device.clone(),
            file_path: self.file_path.clone(),
        }
    }

    pub fn path(&self, tensor_name: &str) -> String {
        if self.path.is_empty() {
            tensor_name.to_string()
        } else {
            [&self.path.join("."), tensor_name].join(".")
        }
    }

    fn resolve_shard(&self, tensor_path: &str) -> Result<usize> {
        self.tensor_to_shard
            .get(tensor_path)
            .copied()
            .ok_or_else(|| {
                candle_core::Error::Msg(format!(
                    "cannot find tensor {tensor_path} in any GGUF shard"
                ))
            })
    }

    pub fn get<S: Into<Shape>>(&self, s: S, name: &str) -> Result<Arc<QTensor>> {
        let path = self.path(name);

        {
            let cache_guard = self.cache.lock().unwrap();
            if let Some((ref cached_name, ref cached_tensor)) = *cache_guard {
                if cached_name == &path {
                    let shape = s.into();
                    if cached_tensor.shape() != &shape {
                        candle_core::bail!(
                            "shape mismatch for {name}, got {:?}, expected {shape:?}",
                            cached_tensor.shape()
                        );
                    }
                    return Ok(cached_tensor.clone());
                }
            }
        }

        let shard_idx = self.resolve_shard(&path)?;
        let mut shards = self.shards.lock().unwrap();
        let shard = &mut shards[shard_idx];
        let tensor = shard.content.tensor(&mut shard.file, &path, &self.device)?;
        let tensor = Arc::new(tensor);
        *self.cache.lock().unwrap() = Some((path.clone(), tensor.clone()));

        let shape = s.into();
        if tensor.shape() != &shape {
            candle_core::bail!(
                "shape mismatch for {name}, got {:?}, expected {shape:?}",
                tensor.shape()
            );
        }
        Ok(tensor)
    }

    /// Shard a tensor along `dim` for rank `rank` out of `world_size`.
    /// Uses candle's native raw byte sharding when possible, falling back to
    /// dequantize-narrow-requantize when the format doesn't support direct sharding.
    pub fn get_sharded<S: Into<Shape>>(
        &self,
        s: S,
        name: &str,
        dim: usize,
        rank: usize,
        world_size: usize,
    ) -> Result<Option<Arc<QTensor>>> {
        if world_size <= 1 {
            return self.get(s, name).map(Some);
        }

        let path = self.path(name);
        let shape = s.into();
        if dim >= shape.dims().len() {
            candle_core::bail!(
                "cannot shard tensor {path} with shape {:?} on dim {dim}",
                shape
            );
        }
        if shape.dims()[dim] % world_size != 0 {
            candle_core::bail!(
                "cannot shard tensor {path} dim {dim} size {} into {world_size} parts",
                shape.dims()[dim]
            );
        }
        let mut shard_shape = shape.dims().to_vec();
        shard_shape[dim] /= world_size;

        let shard_idx = self.resolve_shard(&path)?;
        let mut shards = self.shards.lock().unwrap();
        let shard = &mut shards[shard_idx];
        let Some(tensor) = shard.content.tensor_shard(
            &mut shard.file,
            &path,
            dim,
            rank,
            world_size,
            &self.device,
        )?
        else {
            return Ok(None);
        };
        if tensor.shape().dims() != shard_shape.as_slice() {
            candle_core::bail!(
                "shape mismatch for sharded {name}, got {:?}, expected {:?}",
                tensor.shape(),
                shard_shape
            );
        }
        Ok(Some(Arc::new(tensor)))
    }

    pub fn get_no_shape(&self, name: &str) -> Result<Arc<QTensor>> {
        let path = self.path(name);
        let shard_idx = self.resolve_shard(&path)?;
        let mut shards = self.shards.lock().unwrap();
        let shard = &mut shards[shard_idx];
        let tensor = shard.content.tensor(&mut shard.file, &path, &self.device)?;
        Ok(Arc::new(tensor))
    }

    pub fn clear_cache(&self) {
        *self.cache.lock().unwrap() = None;
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn contains_key(&self, key: &str) -> bool {
        let path = self.path(key);
        self.tensor_to_shard.contains_key(&path)
    }

    pub fn tensor_shape(&self, key: &str) -> Option<Vec<usize>> {
        let path = self.path(key);
        let shard_idx = self.tensor_to_shard.get(&path)?;
        let shards = self.shards.lock().unwrap();
        shards[*shard_idx]
            .content
            .tensor_infos
            .get(&path)
            .map(|info| info.shape.dims().to_vec())
    }

    pub fn tensor_dtype(&self, key: &str) -> Option<GgmlDType> {
        let path = self.path(key);
        let shard_idx = self.tensor_to_shard.get(&path)?;
        let shards = self.shards.lock().unwrap();
        shards[*shard_idx]
            .content
            .tensor_infos
            .get(&path)
            .map(|info| info.ggml_dtype)
    }

    /// Shard a tensor along `dim` without requiring the caller to know the shape.
    /// Uses raw byte-range sharding when possible; falls back to dequant→narrow→requant.
    pub fn get_sharded_no_shape(
        &self,
        name: &str,
        dim: usize,
        rank: usize,
        world_size: usize,
    ) -> Result<Arc<QTensor>> {
        if world_size <= 1 {
            return self.get_no_shape(name);
        }
        let path = self.path(name);
        let shape_dims = self
            .tensor_shape_from_path(&path)
            .ok_or_else(|| candle_core::Error::Msg(format!("cannot find tensor {path}")))?;
        let shape = Shape::from_dims(&shape_dims);
        match self.get_sharded(shape, name, dim, rank, world_size)? {
            Some(t) => Ok(t),
            None => {
                let ws = self.get_no_shape(name)?;
                let orig_dtype = ws.dtype();
                let t = ws.dequantize_f16(&self.device)?;
                self.clear_cache();
                let dim_size = t.dim(dim)?;
                let chunk_size = dim_size / world_size;
                let local = t.narrow(dim, rank * chunk_size, chunk_size)?.contiguous()?;
                drop(t);
                let local_last_dim = local.dim(candle_core::D::Minus1)?;
                let wdtype = if local_last_dim % orig_dtype.block_size() != 0 {
                    GgmlDType::Q8_0
                } else {
                    orig_dtype
                };
                Ok(Arc::new(QTensor::quantize_owned(local, wdtype)?))
            }
        }
    }

    fn tensor_shape_from_path(&self, path: &str) -> Option<Vec<usize>> {
        let shard_idx = self.tensor_to_shard.get(path)?;
        let shards = self.shards.lock().unwrap();
        shards[*shard_idx]
            .content
            .tensor_infos
            .get(path)
            .map(|info| info.shape.dims().to_vec())
    }

    pub fn first_content_metadata(
        &self,
    ) -> std::collections::HashMap<String, candle_core::quantized::gguf_file::Value> {
        let shards = self.shards.lock().unwrap();
        shards[0].content.metadata.clone()
    }
}
