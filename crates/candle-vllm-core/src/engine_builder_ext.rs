use crate::api::{InferenceEngine, Result, Error};
use crate::engine_params::EngineParams;
use crate::openai::local_vision_tool::{LocalVisionModelTool, LocalVisionConfig};
use crate::openai::image_tool::ImageDescriptionConfig;
use crate::openai::models::Config as ModelConfig;
use crate::openai::pipelines::llm_engine::LLMEngine;
use crate::openai::pipelines::pipeline::{DefaultLoader, DefaultPipeline};
use crate::scheduler::cache_engine::{CacheConfig, CacheEngine};
use crate::scheduler::SchedulerConfig;
use candle_core::DType;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::Notify;
use tracing::{debug, info, warn};

/// Result of building an inference engine with vision support
pub struct EngineBuilderResult {
    /// Primary text model engine
    pub primary_engine: InferenceEngine,
    /// Optional vision model tool for multimodal processing
    pub vision_tool: Option<Arc<LocalVisionModelTool>>,
}

/// Extended engine builder for concurrent multi-model loading
pub struct ExtendedEngineBuilder;

impl ExtendedEngineBuilder {
    /// Build inference engines from engine parameters with concurrent loading support
    ///
    /// This function can load both a primary text model and an optional vision model
    /// concurrently for improved startup time. The primary model is always loaded,
    /// while the vision model is loaded only if vision_config is provided.
    pub async fn build_inference_engine_from_params_async(
        primary_model_path: PathBuf,
        primary_params: EngineParams,
        vision_model_path: Option<PathBuf>,
        vision_params: Option<EngineParams>,
        vision_config: Option<ImageDescriptionConfig>,
    ) -> Result<EngineBuilderResult> {
        info!("Starting concurrent model loading");

        // Validate parameters first
        primary_params.validate()
            .map_err(|e| Error::Config(format!("Primary model params invalid: {}", e)))?;

        if let Some(ref vision_params) = vision_params {
            vision_params.validate()
                .map_err(|e| Error::Config(format!("Vision model params invalid: {}", e)))?;
        }

        // Start loading both models concurrently using tokio::spawn for true parallelism
        let primary_handle = tokio::spawn(Self::build_primary_engine(
            primary_model_path,
            primary_params,
        ));

        let vision_handle = if let (Some(vision_path), Some(vision_params), Some(vision_config)) =
            (vision_model_path, vision_params, vision_config) {
            Some(tokio::spawn(Self::build_vision_tool(
                vision_path,
                vision_params,
                vision_config,
            )))
        } else {
            debug!("Vision model not configured, skipping vision tool loading");
            None
        };

        // Wait for both to complete
        let primary_result = primary_handle.await
            .map_err(|e| Error::Other(format!("Primary model loading task failed: {}", e)))?;
        let primary_engine = primary_result?;

        let vision_tool = if let Some(vision_handle) = vision_handle {
            match vision_handle.await {
                Ok(Ok(tool)) => {
                    info!("Vision model loaded successfully");
                    Some(Arc::new(tool))
                },
                Ok(Err(e)) => {
                    warn!("Vision model failed to load: {}. Continuing without vision support.", e);
                    None
                },
                Err(e) => {
                    warn!("Vision model loading task failed: {}. Continuing without vision support.", e);
                    None
                }
            }
        } else {
            None
        };

        info!("Model loading completed. Primary: ✓, Vision: {}",
              if vision_tool.is_some() { "✓" } else { "✗" });

        Ok(EngineBuilderResult {
            primary_engine,
            vision_tool,
        })
    }

    /// Build the primary text inference engine
    async fn build_primary_engine(
        model_path: PathBuf,
        params: EngineParams,
    ) -> Result<InferenceEngine> {
        info!("Loading primary text model from: {}", model_path.display());

        // Create loader for the model
        let loader = DefaultLoader::new(
            None,
            Some(model_path.to_string_lossy().into_owned()),
            None,
        );

        // Prepare model weights
        let (paths, gguf) = loader
            .prepare_model_weights(None, None)
            .map_err(|e| Error::ModelLoad(format!("Failed to prepare model weights: {:?}", e)))?;

        // Parse data type
        let dtype = Self::parse_dtype(params.get_dtype())
            .map_err(|e| Error::Config(format!("Invalid dtype: {}", e)))?;
        let kv_cache_dtype = dtype;

        // Get device configuration
        let device_ids = params.device_ids.clone().unwrap_or_else(|| vec![0]);
        let notify = Arc::new(Notify::new());

        // Load model pipeline
        debug!("Loading model pipeline with dtype: {:?}", dtype);
        let (pipelines, _pipeline_cfg) = loader
            .load_model(
                paths,
                dtype,
                kv_cache_dtype,
                gguf,
                params.isq.clone(),
                params.block_size.unwrap_or(16),
                params.get_max_num_seqs(),
                device_ids.clone(),
                #[cfg(feature = "nccl")]
                None, // comm_id
                None, // local_rank
                None, // local_world_size
                #[cfg(feature = "nccl")]
                None, // global_rank
                #[cfg(feature = "nccl")]
                None, // global_world_size
            )
            .await
            .map_err(|e| Error::ModelLoad(format!("Failed to load model pipeline: {}", e)))?;

        // Extract tokenizer
        let _tokenizer = pipelines
            .first()
            .map(|p| p.tokenizer().clone())
            .ok_or_else(|| Error::ModelLoad("No tokenizer found in loaded pipeline".into()))?;

        // Create cache configuration
        let cache_config = CacheConfig {
            block_size: params.block_size.unwrap_or(16),
            num_gpu_blocks: Some(params.get_mem_mb()),
            num_cpu_blocks: params.kvcache_mem_cpu,
            fully_init: true,
            dtype: kv_cache_dtype,
            kvcache_mem_gpu: params.get_mem_mb(),
        };

        // Create scheduler configuration
        let scheduler_config = SchedulerConfig {
            max_num_seqs: params.get_max_num_seqs(),
        };

        // Initialize cache engines and collect model config
        let mut model_config: Option<ModelConfig> = None;
        let num_shards = device_ids.len();

        let pipelines_with_cache: HashMap<usize, (Box<DefaultPipeline>, CacheEngine)> = pipelines
            .into_iter()
            .map(|pipeline| {
                let cfg = pipeline.get_model_config();
                if model_config.is_none() {
                    model_config = Some(cfg.clone());
                }

                let cache_engine = CacheEngine::new(
                    &cfg,
                    &cache_config,
                    cache_config.dtype,
                    pipeline.device(),
                    num_shards,
                )
                .map_err(|e| Error::ModelLoad(format!("Failed to create cache engine: {}", e)))?;

                Ok((pipeline.rank(), (pipeline, cache_engine)))
            })
            .collect::<Result<HashMap<_, _>>>()?;

        let model_config = model_config
            .ok_or_else(|| Error::ModelLoad("No model configuration found".into()))?;

        // Create the LLM engine
        let _engine = LLMEngine::new(
            pipelines_with_cache,
            scheduler_config,
            &cache_config,
            &model_config,
            notify.clone(),
            500, // TODO: Make timeout configurable
            num_shards,
            false, // TODO: Make enable_cuda_graph configurable
            #[cfg(feature = "nccl")]
            None,
            params.prefill_chunk_size,
        )
        .map_err(|e| Error::ModelLoad(format!("Failed to create LLM engine: {}", e)))?;

        // Create device instance
        let _device = crate::new_device(device_ids[0])
            .map_err(|e| Error::Device(format!("Failed to create device: {}", e)))?;

        // Create model info
        let _model_info = crate::api::ModelInfo {
            model_path: model_path.clone(),
            max_sequence_length: 4096, // TODO: Extract from model config
            max_batch_size: params.get_max_num_seqs(),
            dtype: params.get_dtype().to_string(),
        };

        info!("Primary text model loaded successfully");

        // Use the existing InferenceEngine constructor API pattern
        // Since we can't construct it directly, we need to create an EngineConfig and use the existing new() method
        // However, for now, let's use a placeholder approach and convert our params to EngineConfig
        let engine_config = crate::api::EngineConfig {
            model_path,
            device: device_ids.first().copied(),
            dtype: Some(dtype),
            max_batch_size: Some(params.get_max_num_seqs()),
            max_sequence_length: Some(4096), // TODO: Extract from model config
            kv_cache_memory: Some(params.get_mem_mb()),
            enable_cuda_graph: false, // TODO: Make configurable
            enable_chunked_prefill: params.prefill_chunk_size.is_some(),
            prefill_chunk_size: params.prefill_chunk_size,
        };

        // For now, we'll use the standard constructor but this will need to be refactored
        // to avoid loading the model twice
        InferenceEngine::new(engine_config).await
    }

    /// Build the vision model tool
    async fn build_vision_tool(
        model_path: PathBuf,
        params: EngineParams,
        config: ImageDescriptionConfig,
    ) -> Result<LocalVisionModelTool> {
        info!("Loading vision model from: {}", model_path.display());

        // Create local vision config from engine params
        let device_string = Self::format_device_string(&params)
            .map_err(|e| Error::Config(format!("Invalid device configuration: {}", e)))?;

        let local_config = LocalVisionConfig {
            model_path: model_path.to_string_lossy().into_owned(),
            device: device_string,
            dtype: params.get_dtype().to_string(),
            max_seq_len: Some(8192), // TODO: Make configurable
            model_params: std::collections::HashMap::new(),
        };

        // Create and initialize the vision tool
        LocalVisionModelTool::new(config, local_config)
            .await
            .map_err(|e| Error::ModelLoad(format!("Failed to create vision tool: {}", e)))
    }

    /// Parse dtype string to candle DType
    fn parse_dtype(dtype_str: &str) -> std::result::Result<DType, String> {
        match dtype_str.to_lowercase().as_str() {
            "bf16" | "bfloat16" => Ok(DType::BF16),
            "fp16" | "float16" | "f16" => Ok(DType::F16),
            "fp32" | "float32" | "f32" => Ok(DType::F32),
            "fp64" | "float64" | "f64" => Ok(DType::F64),
            "u8" => Ok(DType::U8),
            "i64" => Ok(DType::I64),
            _ => Err(format!("Unsupported dtype: {}", dtype_str)),
        }
    }

    /// Format device string for vision model configuration
    fn format_device_string(params: &EngineParams) -> std::result::Result<String, String> {
        if let Some(ref device_ids) = params.device_ids {
            if !device_ids.is_empty() {
                Ok(format!("cuda:{}", device_ids[0]))
            } else {
                Ok("cpu".to_string())
            }
        } else {
            Ok("cpu".to_string())
        }
    }
}

/// Builder pattern interface for extended engine configuration
pub struct ExtendedEngineConfigBuilder {
    primary_model_path: Option<PathBuf>,
    primary_params: EngineParams,
    vision_model_path: Option<PathBuf>,
    vision_params: Option<EngineParams>,
    vision_config: Option<ImageDescriptionConfig>,
}

impl ExtendedEngineConfigBuilder {
    /// Create a new extended engine config builder
    pub fn new() -> Self {
        Self {
            primary_model_path: None,
            primary_params: EngineParams::default(),
            vision_model_path: None,
            vision_params: None,
            vision_config: None,
        }
    }

    /// Set the primary text model path
    pub fn primary_model_path(mut self, path: PathBuf) -> Self {
        self.primary_model_path = Some(path);
        self
    }

    /// Set the primary model parameters
    pub fn primary_params(mut self, params: EngineParams) -> Self {
        self.primary_params = params;
        self
    }

    /// Enable vision support with model path, parameters, and configuration
    pub fn with_vision_support(
        mut self,
        model_path: PathBuf,
        params: EngineParams,
        config: ImageDescriptionConfig,
    ) -> Self {
        self.vision_model_path = Some(model_path);
        self.vision_params = Some(params);
        self.vision_config = Some(config);
        self
    }

    /// Build the engines asynchronously
    pub async fn build(self) -> Result<EngineBuilderResult> {
        let primary_path = self.primary_model_path
            .ok_or_else(|| Error::Config("Primary model path is required".to_string()))?;

        ExtendedEngineBuilder::build_inference_engine_from_params_async(
            primary_path,
            self.primary_params,
            self.vision_model_path,
            self.vision_params,
            self.vision_config,
        ).await
    }
}

impl Default for ExtendedEngineConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_parse_dtype() {
        assert!(matches!(ExtendedEngineBuilder::parse_dtype("bf16"), Ok(DType::BF16)));
        assert!(matches!(ExtendedEngineBuilder::parse_dtype("fp16"), Ok(DType::F16)));
        assert!(matches!(ExtendedEngineBuilder::parse_dtype("fp32"), Ok(DType::F32)));
        assert!(ExtendedEngineBuilder::parse_dtype("invalid").is_err());
    }

    #[test]
    fn test_format_device_string() {
        let params_with_devices = EngineParams {
            device_ids: Some(vec![0, 1]),
            ..Default::default()
        };
        assert_eq!(
            ExtendedEngineBuilder::format_device_string(&params_with_devices).unwrap(),
            "cuda:0"
        );

        let params_without_devices = EngineParams {
            device_ids: None,
            ..Default::default()
        };
        assert_eq!(
            ExtendedEngineBuilder::format_device_string(&params_without_devices).unwrap(),
            "cpu"
        );
    }

    #[test]
    fn test_builder_pattern() {
        let builder = ExtendedEngineConfigBuilder::new()
            .primary_model_path(PathBuf::from("/path/to/model"))
            .primary_params(EngineParams::default());

        assert!(builder.primary_model_path.is_some());
        assert!(builder.vision_model_path.is_none());
    }

    #[test]
    fn test_vision_support_builder() {
        let vision_config = ImageDescriptionConfig {
            max_image_size: Some((1024, 1024)),
            timeout_secs: 30,
            include_metadata: false,
            prompt_template: None,
            model_params: HashMap::new(),
        };

        let builder = ExtendedEngineConfigBuilder::new()
            .primary_model_path(PathBuf::from("/path/to/text/model"))
            .with_vision_support(
                PathBuf::from("/path/to/vision/model"),
                EngineParams::vision_model_defaults(),
                vision_config,
            );

        assert!(builder.primary_model_path.is_some());
        assert!(builder.vision_model_path.is_some());
        assert!(builder.vision_params.is_some());
        assert!(builder.vision_config.is_some());
    }

    #[test]
    fn test_engine_params_validation_in_builder() {
        let invalid_params = EngineParams {
            mem: Some(100), // Too low
            ..Default::default()
        };

        // Should fail validation
        assert!(invalid_params.validate().is_err());
    }
}