use crate::openai::conversation::Message;
use crate::openai::models::Config;
use crate::openai::requests::{ChatMessage, ImageUrlContent, MessageContent, MessageContentType};
use candle_core::{DType, Device, Result, Storage, Tensor};
use image::imageops::FilterType;
use image::{DynamicImage, GenericImageView};
use serde::{Deserialize, Serialize};

pub const IMAGE_PLACEHOLDER: &str = "<|CANDLE-VLLM-IMAGE|>";
pub const PLACEHOLDER: &str = "<|CANDLE-VLLM-PLACEHOLDER|>";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MultiModalModelType {
    Mistral3VL,
    Gemma3,
    Qwen3VL,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ImageData {
    pub raw: Vec<u8>,
    pub shape: Vec<usize>,
    pub patches: Vec<(usize, usize)>,
    pub image_idx: i32,
    #[serde(default)]
    pub image_token_offset: usize,
    #[serde(default)]
    pub tokens_per_image: Vec<usize>,
    #[serde(default)]
    pub image_token_id: Option<u32>,
}

impl ImageData {
    pub fn to_tensor_f32(&self, device: &Device) -> Result<Tensor> {
        let floats: &[f32] = bytemuck::cast_slice(&self.raw);
        Tensor::from_slice(floats, self.shape.clone(), device)
    }
}

#[derive(Clone, Debug)]
pub struct ImageProcessConfig {
    image_start_token: Option<String>,
    image_token: String,
    image_break_token: Option<String>,
    image_end_token: String,
    pub spatial_merge_size: usize,
    pub mm_tokens_per_image: Option<usize>,
    pub do_normalize: Option<bool>,
    pub do_resize: Option<bool>,
    pub image_mean: Option<[f32; 3]>,
    pub image_std: Option<[f32; 3]>,
    pub max_height: usize,
    pub max_width: usize,
    pub patch_size: usize,
    pub temporal_patch_size: Option<usize>,
    pub absolute_resize: bool,
    pub scale_factor: Option<f32>,
    pub resampling: Option<usize>,
    pub model_type: MultiModalModelType,
    pub image_token_id: Option<u32>,
}

impl ImageProcessConfig {
    #[allow(clippy::too_many_arguments)]
    pub fn default(
        image_start_token: Option<String>,
        image_token: String,
        image_break_token: Option<String>,
        image_end_token: String,
        spatial_merge_size: usize,
        temporal_patch_size: Option<usize>,
        patch_size: usize,
        image_size: usize,
        absolute_resize: bool,
    ) -> Self {
        Self {
            image_start_token,
            image_token,
            image_break_token,
            image_end_token,
            spatial_merge_size,
            mm_tokens_per_image: None,
            do_normalize: Some(true),
            do_resize: Some(true),
            image_mean: None,
            image_std: None,
            max_height: image_size,
            max_width: image_size,
            patch_size,
            temporal_patch_size,
            absolute_resize,
            scale_factor: None,
            resampling: None,
            model_type: MultiModalModelType::Mistral3VL,
            image_token_id: None,
        }
    }

    pub fn prompt_marker_tokens(&self) -> Vec<String> {
        let mut tokens = Vec::new();
        if let Some(start) = &self.image_start_token {
            if !start.is_empty() {
                tokens.push(start.clone());
            }
        }
        if !self.image_token.is_empty() {
            tokens.push(self.image_token.clone());
        }
        if let Some(brk) = &self.image_break_token {
            if !brk.is_empty() {
                tokens.push(brk.clone());
            }
        }
        if !self.image_end_token.is_empty() {
            tokens.push(self.image_end_token.clone());
        }
        tokens.sort_by_key(|token| std::cmp::Reverse(token.len()));
        tokens.dedup();
        tokens
    }
}

pub trait ImageProcessTrait: Send {
    fn process_inputs(
        &mut self,
        prompt: &mut String,
        images: &[DynamicImage],
    ) -> Result<(Tensor, Vec<(usize, usize)>)>;
}

pub fn compute_tokens_per_image(
    cfg: &ImageProcessConfig,
    image_sizes: &[(usize, usize)],
) -> Vec<usize> {
    if image_sizes.is_empty() {
        return Vec::new();
    }
    match cfg.model_type {
        MultiModalModelType::Gemma3 => {
            if let Some(tokens) = cfg.mm_tokens_per_image {
                return vec![tokens; image_sizes.len()];
            }
            let denom = cfg.patch_size * cfg.spatial_merge_size;
            if denom == 0 {
                return vec![0; image_sizes.len()];
            }
            image_sizes
                .iter()
                .map(|&(h, w)| (h / denom) * (w / denom))
                .collect()
        }
        MultiModalModelType::Qwen3VL => {
            let merge_area = cfg.spatial_merge_size * cfg.spatial_merge_size;
            image_sizes
                .iter()
                .map(|&(h, w)| {
                    if merge_area == 0 {
                        0
                    } else {
                        (h * w) / merge_area
                    }
                })
                .collect()
        }
        MultiModalModelType::Mistral3VL => {
            let denom = cfg.patch_size * cfg.spatial_merge_size;
            if denom == 0 {
                return vec![0; image_sizes.len()];
            }
            image_sizes
                .iter()
                .map(|&(h, w)| (h / denom) * (w / denom))
                .collect()
        }
    }
}

pub fn compute_image_slice(
    token_ids: &[u32],
    num_cached_tokens: usize,
    images: &ImageData,
) -> Option<(i32, usize)> {
    let base_idx = images.image_idx;
    if base_idx < 0 {
        return None;
    }
    let num_images = if !images.tokens_per_image.is_empty() {
        images.tokens_per_image.len()
    } else {
        images.patches.len()
    };
    if num_images == 0 {
        return None;
    }

    let base_idx = base_idx as usize;
    if num_cached_tokens == 0 {
        return (base_idx < num_images).then_some((base_idx as i32, 0));
    }

    let cached_len = num_cached_tokens.min(token_ids.len());
    if cached_len == 0 {
        return (base_idx < num_images).then_some((base_idx as i32, 0));
    }

    let Some(image_token_id) = images.image_token_id else {
        return (base_idx < num_images).then_some((base_idx as i32, 0));
    };
    if images.tokens_per_image.is_empty() {
        return (base_idx < num_images).then_some((base_idx as i32, 0));
    }

    let cached_image_tokens = token_ids[..cached_len]
        .iter()
        .filter(|&&id| id == image_token_id)
        .count();

    let mut remaining = cached_image_tokens;
    let mut prefix_idx = 0usize;
    let mut token_offset = 0usize;
    for &tokens in &images.tokens_per_image {
        if tokens == 0 {
            break;
        }
        if remaining >= tokens {
            remaining -= tokens;
            prefix_idx += 1;
        } else {
            token_offset = remaining;
            break;
        }
    }

    let mut image_idx = prefix_idx;
    if base_idx > image_idx {
        image_idx = base_idx;
        token_offset = 0;
    }
    if image_idx >= num_images {
        return None;
    }

    Some((image_idx.min(i32::MAX as usize) as i32, token_offset))
}

pub fn load_image_from_url(url: &str) -> Result<DynamicImage> {
    tracing::info!("Downloading image from {}", url);
    let bytes = reqwest::blocking::get(url)
        .map_err(candle_core::Error::wrap)?
        .bytes()
        .map_err(candle_core::Error::wrap)?;
    image::load_from_memory(&bytes).map_err(candle_core::Error::wrap)
}

pub fn load_image_from_base64(data: &str) -> Result<DynamicImage> {
    use base64::prelude::{Engine as _, BASE64_STANDARD};
    let base64_part = data.split(',').next_back().unwrap_or(data);
    let bytes = BASE64_STANDARD
        .decode(base64_part)
        .map_err(candle_core::Error::wrap)?;
    image::load_from_memory(&bytes).map_err(candle_core::Error::wrap)
}

pub fn get_tensor_raw_data(t: &Tensor) -> Result<(Vec<u8>, Vec<usize>)> {
    let shape = t.dims().to_vec();
    let (storage, _) = t.storage_and_layout();
    let storage = match &*storage {
        Storage::Cpu(cpu) => cpu,
        _ => candle_core::bail!("multimodal tensor serialization requires CPU tensor"),
    };
    let bytes = match t.dtype() {
        DType::F32 => {
            let slice = storage.as_slice::<f32>()?;
            bytemuck::cast_slice(slice).to_vec()
        }
        _ => candle_core::bail!("unsupported dtype {:?} for tensor serialization", t.dtype()),
    };
    Ok((bytes, shape))
}

fn image_resize(
    image: &DynamicImage,
    mut height: usize,
    mut width: usize,
    max_height: usize,
    max_width: usize,
    patch_size: usize,
    filter: FilterType,
) -> DynamicImage {
    let ratio = (height as f64 / max_height as f64).max(width as f64 / max_width as f64);
    if ratio > 1.0 {
        height = (height as f64 / ratio).floor() as usize;
        width = (width as f64 / ratio).floor() as usize;
    }

    let num_height_tokens = (height.saturating_sub(1)) / patch_size + 1;
    let num_width_tokens = (width.saturating_sub(1)) / patch_size + 1;
    image.resize_exact(
        (num_width_tokens * patch_size) as u32,
        (num_height_tokens * patch_size) as u32,
        filter,
    )
}

pub fn to_tensor(
    images: &[DynamicImage],
    image_mean: Option<[f32; 3]>,
    image_std: Option<[f32; 3]>,
    scale_factor: Option<f32>,
) -> Result<(Tensor, Vec<(usize, usize)>)> {
    let mut image_sizes = Vec::with_capacity(images.len());
    let mut pixel_values = Vec::with_capacity(images.len());
    for image in images {
        let (width, height) = image.dimensions();
        image_sizes.push((height as usize, width as usize));

        let rgb = image.to_rgb32f();
        let (w, h) = rgb.dimensions();
        let mut data = rgb.into_raw();
        if let Some(scale) = scale_factor {
            for v in &mut data {
                *v *= scale;
            }
        }

        let t = Tensor::from_vec(data, (h as usize, w as usize, 3), &Device::Cpu)?
            .permute((2, 0, 1))?
            .contiguous()?;
        let t = if let (Some(mean), Some(std)) = (image_mean, image_std) {
            let mean = Tensor::new(&mean[..], &Device::Cpu)?.reshape((3, 1, 1))?;
            let std = Tensor::new(&std[..], &Device::Cpu)?.reshape((3, 1, 1))?;
            t.broadcast_sub(&mean)?.broadcast_div(&std)?
        } else {
            t
        };
        pixel_values.push(t.unsqueeze(0)?);
    }
    Ok((Tensor::cat(&pixel_values, 0)?, image_sizes))
}

pub trait ToFilter {
    fn to_filter(self) -> Result<FilterType>;
}

impl ToFilter for Option<usize> {
    fn to_filter(self) -> Result<FilterType> {
        match self {
            Some(0) => Ok(FilterType::Nearest),
            Some(1) => Ok(FilterType::Lanczos3),
            Some(2) | None => Ok(FilterType::Triangle),
            Some(3) => Ok(FilterType::CatmullRom),
            Some(4) => Ok(FilterType::Nearest),
            Some(x) => candle_core::bail!("unsupported image filter {}", x),
        }
    }
}

pub struct ImageProcessor {
    cfg: ImageProcessConfig,
    fixed_width: Option<usize>,
    fixed_height: Option<usize>,
}

impl ImageProcessor {
    const DEFAULT_MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];
    const DEFAULT_STD: [f32; 3] = [0.26862954, 0.26130258, 0.27577711];

    pub fn new(cfg: &ImageProcessConfig) -> Self {
        Self {
            cfg: cfg.clone(),
            fixed_width: None,
            fixed_height: None,
        }
    }

    fn preprocess_images(&mut self, images: &[DynamicImage]) -> Vec<DynamicImage> {
        let filter = self
            .cfg
            .resampling
            .to_filter()
            .unwrap_or(FilterType::Nearest);
        if self.cfg.absolute_resize {
            return images
                .iter()
                .map(|img| {
                    img.resize_exact(
                        self.cfg.max_width as u32,
                        self.cfg.max_height as u32,
                        filter,
                    )
                })
                .collect();
        }

        let mut resized = Vec::with_capacity(images.len());
        for img in images {
            let next = if let (Some(h), Some(w)) = (self.fixed_height, self.fixed_width) {
                img.resize_exact(w as u32, h as u32, filter)
            } else {
                let (h, w) = img.dimensions();
                let x = image_resize(
                    img,
                    h as usize,
                    w as usize,
                    self.cfg.max_height,
                    self.cfg.max_width,
                    self.cfg.patch_size,
                    filter,
                );
                self.fixed_height = Some(x.height() as usize);
                self.fixed_width = Some(x.width() as usize);
                x
            };
            resized.push(next);
        }
        resized
    }

    fn preprocess(&mut self, images: &[DynamicImage]) -> Result<(Tensor, Vec<(usize, usize)>)> {
        let do_normalize = self.cfg.do_normalize.unwrap_or(true);
        let image_mean = if do_normalize {
            Some(self.cfg.image_mean.unwrap_or(Self::DEFAULT_MEAN))
        } else {
            None
        };
        let image_std = if do_normalize {
            Some(self.cfg.image_std.unwrap_or(Self::DEFAULT_STD))
        } else {
            None
        };
        let images = self.preprocess_images(images);
        to_tensor(&images, image_mean, image_std, self.cfg.scale_factor)
    }
}

impl ImageProcessTrait for ImageProcessor {
    fn process_inputs(
        &mut self,
        prompt: &mut String,
        images: &[DynamicImage],
    ) -> Result<(Tensor, Vec<(usize, usize)>)> {
        let (pixel_values, image_sizes_all) = self.preprocess(images)?;
        let mut image_sizes_iter = image_sizes_all.clone().into_iter();
        let mut replace_strings = Vec::new();

        while prompt.contains(IMAGE_PLACEHOLDER) {
            let (height, width) = image_sizes_iter.next().ok_or_else(|| {
                candle_core::Error::Msg("image placeholder count exceeds supplied images".into())
            })?;
            let rows = height / (self.cfg.patch_size * self.cfg.spatial_merge_size);
            let cols = width / (self.cfg.patch_size * self.cfg.spatial_merge_size);

            let mut replace_tokens = vec![
                [
                    vec![self.cfg.image_token.clone(); cols],
                    self.cfg
                        .image_break_token
                        .as_ref()
                        .map(|t| vec![t.clone()])
                        .unwrap_or_default(),
                ]
                .concat();
                rows
            ]
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();

            if self.cfg.image_break_token.is_some() {
                if let Some(last) = replace_tokens.last_mut() {
                    *last = self.cfg.image_end_token.clone();
                }
            } else {
                replace_tokens.push(self.cfg.image_end_token.clone());
            }

            if let Some(start) = &self.cfg.image_start_token {
                let mut with_start = vec![start.clone()];
                with_start.extend(replace_tokens);
                replace_tokens = with_start;
            }

            replace_strings.push(replace_tokens.join(""));
            *prompt = prompt.replacen(IMAGE_PLACEHOLDER, PLACEHOLDER, 1);
        }

        while prompt.contains(PLACEHOLDER) {
            if let Some(replace_str) = replace_strings.pop() {
                *prompt = prompt.replacen(PLACEHOLDER, &replace_str, 1);
            } else {
                break;
            }
        }

        Ok((pixel_values, image_sizes_all))
    }
}

#[derive(Clone)]
pub struct Qwen3VLImageProcessor {
    pub cfg: ImageProcessConfig,
    pub patch_size: usize,
    pub merge_size: usize,
    pub temporal_patch_size: usize,
    pub min_pixels: usize,
    pub max_pixels: usize,
    pub fixed_width: Option<usize>,
    pub fixed_height: Option<usize>,
}

impl Qwen3VLImageProcessor {
    const DEFAULT_MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];
    const DEFAULT_STD: [f32; 3] = [0.26862954, 0.26130258, 0.27577711];
    const VISION_START: &'static str = "<|vision_start|>";
    const VISION_END: &'static str = "<|vision_end|>";
    const IMAGE_PAD: &'static str = "<|image_pad|>";

    pub fn default(cfg: &ImageProcessConfig) -> Self {
        let max_row = std::cmp::max(cfg.max_height, cfg.max_width);
        Self {
            cfg: cfg.clone(),
            patch_size: cfg.patch_size,
            merge_size: cfg.spatial_merge_size,
            temporal_patch_size: cfg.temporal_patch_size.unwrap_or(2),
            min_pixels: 256 * 256,
            max_pixels: max_row * max_row,
            fixed_width: None,
            fixed_height: None,
        }
    }

    fn smart_resize(&self, h: usize, w: usize) -> Result<(usize, usize)> {
        let factor = self.patch_size * self.merge_size;
        let mut nh = (h as f64 / factor as f64).round() as usize * factor;
        let mut nw = (w as f64 / factor as f64).round() as usize * factor;
        let pixels = nh * nw;
        if pixels > self.max_pixels {
            let beta = (pixels as f64 / self.max_pixels as f64).sqrt();
            nh = ((nh as f64 / beta) as usize / factor) * factor;
            nw = ((nw as f64 / beta) as usize / factor) * factor;
        } else if pixels < self.min_pixels && pixels > 0 {
            let beta = (self.min_pixels as f64 / pixels as f64).sqrt();
            nh = ((nh as f64 * beta) as usize / factor) * factor;
            nw = ((nw as f64 * beta) as usize / factor) * factor;
        }
        Ok((nh, nw))
    }

    fn prepreprocess(
        &mut self,
        image: &DynamicImage,
        target_hw: (u32, u32),
    ) -> Result<(Tensor, (usize, usize))> {
        let (th, tw) = target_hw;
        let (mut nh, mut nw) = self.smart_resize(th as usize, tw as usize)?;
        if let (Some(h), Some(w)) = (self.fixed_height, self.fixed_width) {
            nh = h;
            nw = w;
        } else {
            self.fixed_height = Some(nh);
            self.fixed_width = Some(nw);
        }

        let image = image
            .resize_exact(nw as u32, nh as u32, self.cfg.resampling.to_filter()?)
            .to_rgb8();
        let image_mean = Some(self.cfg.image_mean.unwrap_or(Self::DEFAULT_MEAN));
        let image_std = Some(self.cfg.image_std.unwrap_or(Self::DEFAULT_STD));
        let (mut patches, _) = to_tensor(
            &[DynamicImage::ImageRgb8(image)],
            image_mean,
            image_std,
            self.cfg.scale_factor,
        )?;

        if patches.dim(0)? == 1 {
            patches = patches.repeat((self.temporal_patch_size, 1, 1, 1))?;
        }

        let c = patches.dim(1)?;
        let grid_t = patches.dim(0)? / self.temporal_patch_size;
        let grid_h = nh / self.patch_size;
        let grid_w = nw / self.patch_size;

        patches = patches.reshape(&[
            grid_t,
            self.temporal_patch_size,
            c,
            grid_h / self.merge_size,
            self.merge_size,
            self.patch_size,
            grid_w / self.merge_size,
            self.merge_size,
            self.patch_size,
        ])?;
        patches = patches.permute([0, 3, 6, 4, 7, 2, 1, 5, 8])?;
        let patches = patches.reshape((
            grid_t * grid_h * grid_w,
            c * self.temporal_patch_size * self.patch_size * self.patch_size,
        ))?;

        Ok((patches, (grid_h, grid_w)))
    }
}

impl ImageProcessTrait for Qwen3VLImageProcessor {
    fn process_inputs(
        &mut self,
        prompt: &mut String,
        images: &[DynamicImage],
    ) -> Result<(Tensor, Vec<(usize, usize)>)> {
        let (max_w, max_h) = images
            .iter()
            .map(|i| i.dimensions())
            .fold((0, 0), |(mw, mh), (w, h)| (mw.max(w), mh.max(h)));

        let mut pixel_values = Vec::new();
        let mut grid_thw = Vec::new();
        for image in images {
            let (patches, (h, w)) = self.prepreprocess(image, (max_h, max_w))?;
            pixel_values.push(patches);
            grid_thw.push((h, w));
        }
        let pixel_values = Tensor::stack(&pixel_values, 0)?;

        let merge_len = self.merge_size * self.merge_size;
        let mut image_idx = 0usize;
        let mut replace_strings = Vec::new();
        while prompt.contains(IMAGE_PLACEHOLDER) {
            let grid = grid_thw[image_idx];
            let num_patches = (grid.0 * grid.1) / merge_len;
            let mut replace_tokens = vec![Self::VISION_START];
            replace_tokens.extend(vec![Self::IMAGE_PAD; num_patches]);
            replace_tokens.push(Self::VISION_END);

            replace_strings.push(replace_tokens.join(""));
            *prompt = prompt.replacen(IMAGE_PLACEHOLDER, PLACEHOLDER, 1);
            image_idx += 1;
        }
        while prompt.contains(PLACEHOLDER) {
            if let Some(replace_str) = replace_strings.pop() {
                *prompt = prompt.replacen(PLACEHOLDER, &replace_str, 1);
            } else {
                break;
            }
        }

        Ok((pixel_values, grid_thw))
    }
}

#[derive(Debug, Clone, Deserialize)]
struct Mistral3VisionConfig {
    patch_size: usize,
}

#[derive(Debug, Clone, Deserialize)]
struct Mistral3VLMetaConfig {
    image_token_index: usize,
    spatial_merge_size: usize,
    vision_config: Mistral3VisionConfig,
}

#[derive(Debug, Clone, Deserialize)]
struct Gemma3VisionMetaConfig {
    patch_size: usize,
}

#[derive(Debug, Clone, Deserialize)]
struct Gemma3VLMetaConfig {
    image_token_index: usize,
    mm_tokens_per_image: usize,
    vision_config: Gemma3VisionMetaConfig,
}

#[derive(Debug, Clone, Deserialize)]
struct Qwen3VLVisionMetaConfig {
    spatial_merge_size: usize,
    temporal_patch_size: usize,
    patch_size: usize,
}

#[derive(Debug, Clone, Deserialize)]
struct Qwen3VLMetaConfig {
    vision_config: Qwen3VLVisionMetaConfig,
    image_token_id: u32,
}

pub fn get_image_config(config: &Config) -> Result<Option<ImageProcessConfig>> {
    let arch = config.architectures.as_ref().and_then(|a| a.first());
    let Some(arch) = arch else {
        return Ok(None);
    };

    let Some(extra) = config.extra_config_json.as_ref() else {
        return Ok(None);
    };

    match arch.as_str() {
        "Mistral3ForConditionalGeneration" => {
            let cfg: Mistral3VLMetaConfig =
                serde_json::from_str(extra).map_err(candle_core::Error::wrap)?;
            let mut img_cfg = ImageProcessConfig::default(
                None,
                "[IMG]".to_string(),
                Some("[IMG_BREAK]".to_string()),
                "[IMG_END]".to_string(),
                cfg.spatial_merge_size,
                None,
                cfg.vision_config.patch_size,
                896,
                false,
            );
            img_cfg.model_type = MultiModalModelType::Mistral3VL;
            img_cfg.image_token_id = Some(cfg.image_token_index as u32);
            Ok(Some(img_cfg))
        }
        "Gemma3ForConditionalGeneration" => {
            let cfg: Gemma3VLMetaConfig =
                serde_json::from_str(extra).map_err(candle_core::Error::wrap)?;
            let mut img_cfg = ImageProcessConfig::default(
                Some("<start_of_image>".to_string()),
                "<image_soft_token>".to_string(),
                None,
                "<end_of_image>".to_string(),
                4,
                None,
                cfg.vision_config.patch_size,
                896,
                true,
            );
            img_cfg.model_type = MultiModalModelType::Gemma3;
            img_cfg.image_token_id = Some(cfg.image_token_index as u32);
            img_cfg.mm_tokens_per_image = Some(cfg.mm_tokens_per_image);
            img_cfg.scale_factor = Some(0.003921567);
            img_cfg.image_mean = Some([0.5, 0.5, 0.5]);
            img_cfg.image_std = Some([0.5, 0.5, 0.5]);
            Ok(Some(img_cfg))
        }
        "Qwen3VLForConditionalGeneration"
        | "Qwen3VLMoeForConditionalGeneration"
        | "Qwen3_5ForConditionalGeneration"
        | "Qwen3_5MoeForConditionalGeneration"
        | "Qwen3NextForConditionalGeneration" => {
            let cfg: Qwen3VLMetaConfig =
                serde_json::from_str(extra).map_err(candle_core::Error::wrap)?;
            let mut img_cfg = ImageProcessConfig::default(
                Some("<|vision_start|>".to_string()),
                "<|image_pad|>".to_string(),
                None,
                "<|vision_end|>".to_string(),
                cfg.vision_config.spatial_merge_size,
                Some(cfg.vision_config.temporal_patch_size),
                cfg.vision_config.patch_size,
                896,
                false,
            );
            img_cfg.model_type = MultiModalModelType::Qwen3VL;
            img_cfg.image_token_id = Some(cfg.image_token_id);
            img_cfg.image_mean = Some([0.5, 0.5, 0.5]);
            img_cfg.image_std = Some([0.5, 0.5, 0.5]);
            Ok(Some(img_cfg))
        }
        _ => Ok(None),
    }
}

fn extract_text_content(content: &MessageContentType) -> String {
    match content {
        MessageContentType::PureText(text) => text.clone(),
        MessageContentType::Single(item) => match item {
            MessageContent::Text { text } => text.clone(),
            _ => String::new(),
        },
        MessageContentType::Multi(items) => items
            .iter()
            .filter_map(|item| match item {
                MessageContent::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join(" "),
    }
}

fn parse_template_tool_arguments(arguments: Option<&str>) -> serde_json::Value {
    let Some(raw) = arguments.map(str::trim).filter(|s| !s.is_empty()) else {
        return serde_json::json!({});
    };

    match serde_json::from_str::<serde_json::Value>(raw).ok() {
        Some(serde_json::Value::Object(obj)) => serde_json::Value::Object(obj),
        Some(serde_json::Value::String(inner)) => {
            match serde_json::from_str::<serde_json::Value>(inner.trim()).ok() {
                Some(serde_json::Value::Object(obj)) => serde_json::Value::Object(obj),
                _ => serde_json::json!({}),
            }
        }
        _ => serde_json::json!({}),
    }
}

fn to_template_tool_call(call: &crate::tools::ToolCall) -> serde_json::Value {
    serde_json::json!({
        "id": call.id.clone(),
        "type": call.call_type.clone(),
        "function": {
            "name": call.function.name.clone(),
            "arguments": parse_template_tool_arguments(Some(call.function.arguments.as_str()))
        }
    })
}

fn append_message_item(
    item: &MessageContent,
    prompt: &mut String,
    images: &mut Vec<DynamicImage>,
) -> Result<()> {
    match item {
        MessageContent::Text { text } => prompt.push_str(text),
        MessageContent::ImageUrl { image_url } => {
            let url = image_url.url();
            let img = if url.starts_with("data:") {
                load_image_from_base64(url)?
            } else {
                load_image_from_url(url)?
            };
            prompt.push_str(IMAGE_PLACEHOLDER);
            images.push(img);
        }
        MessageContent::ImageBase64 { image_base64 } => {
            let img = load_image_from_base64(image_base64)?;
            prompt.push_str(IMAGE_PLACEHOLDER);
            images.push(img);
        }
    }
    Ok(())
}

pub fn convert_chat_message(
    msg: &ChatMessage,
    processor: &mut Option<Box<dyn ImageProcessTrait + Send>>,
    images_tensors: &mut Vec<(Tensor, Vec<(usize, usize)>)>,
) -> Result<Message> {
    let role = msg.role.clone();
    let mut prompt = String::new();
    let mut images = Vec::new();

    if role == "assistant" && msg.tool_calls.is_some() {
        let content = msg
            .content
            .as_ref()
            .map(extract_text_content)
            .unwrap_or_default()
            .trim()
            .to_owned();
        let template_calls = msg
            .tool_calls
            .as_ref()
            .unwrap()
            .iter()
            .map(to_template_tool_call)
            .collect::<Vec<_>>();
        return Ok(Message {
            role,
            content,
            num_images: 0,
            tool_calls: Some(template_calls),
            tool_call_id: None,
        });
    }

    if role == "tool" {
        let content = msg
            .content
            .as_ref()
            .map(extract_text_content)
            .unwrap_or_default()
            .trim()
            .to_owned();
        return Ok(Message {
            role,
            content,
            num_images: 0,
            tool_calls: None,
            tool_call_id: msg.tool_call_id.clone(),
        });
    }

    if let Some(content) = &msg.content {
        match content {
            MessageContentType::PureText(text) => prompt.push_str(text),
            MessageContentType::Single(item) => {
                append_message_item(item, &mut prompt, &mut images)?;
                prompt.push(' ');
            }
            MessageContentType::Multi(items) => {
                for item in items {
                    append_message_item(item, &mut prompt, &mut images)?;
                    prompt.push(' ');
                }
            }
        }
    }

    if !images.is_empty() {
        if let Some(processor) = processor.as_mut() {
            let (images_tensor, image_sizes) = processor.process_inputs(&mut prompt, &images)?;
            images_tensors.push((images_tensor, image_sizes));
        }
    }

    Ok(Message::new(role, prompt.trim().to_owned(), images.len()))
}

pub fn build_messages_and_images(
    messages: &[ChatMessage],
    img_cfg: Option<&ImageProcessConfig>,
) -> Result<(Vec<Message>, Option<ImageData>)> {
    let mut processor: Option<Box<dyn ImageProcessTrait + Send>> = img_cfg.map(|cfg| {
        if cfg.model_type == MultiModalModelType::Qwen3VL {
            Box::new(Qwen3VLImageProcessor::default(cfg)) as Box<dyn ImageProcessTrait + Send>
        } else {
            Box::new(ImageProcessor::new(cfg)) as Box<dyn ImageProcessTrait + Send>
        }
    });

    let mut images = Vec::<(Tensor, Vec<(usize, usize)>)>::new();
    let messages = messages
        .iter()
        .map(|m| convert_chat_message(m, &mut processor, &mut images))
        .collect::<Result<Vec<_>>>()?;

    let image_data = if !images.is_empty() && img_cfg.is_some() {
        let mut image_sizes = Vec::new();
        let mut image_tensors = Vec::new();
        for (t, s) in &images {
            image_tensors.push(t);
            image_sizes.extend(s.iter().copied());
        }
        let images_tensor = Tensor::cat(&image_tensors, 0)?;
        let (images_raw, images_shape) = get_tensor_raw_data(&images_tensor)?;
        let cfg = img_cfg.unwrap();
        let tokens_per_image = compute_tokens_per_image(cfg, &image_sizes);
        Some(ImageData {
            raw: images_raw,
            shape: images_shape,
            patches: image_sizes,
            image_idx: 0,
            image_token_offset: 0,
            tokens_per_image,
            image_token_id: cfg.image_token_id,
        })
    } else {
        None
    };

    Ok((messages, image_data))
}

impl ImageUrlContent {
    pub fn url(&self) -> &str {
        match self {
            Self::Url(url) => url,
            Self::Object { url, .. } => url,
        }
    }
}
