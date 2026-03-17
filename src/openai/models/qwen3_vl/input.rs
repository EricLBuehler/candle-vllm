use crate::openai::multimodal::{
    ImageProcessConfig, ImageProcessTrait, ToFilter, IMAGE_PLACEHOLDER, PLACEHOLDER,
};
use candle_core::{Result, Tensor};
use image::{DynamicImage, GenericImageView};
/// Qwen3-VL Image + Prompt Processor
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
    #[allow(dead_code)]
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
}

fn to_tensor(
    images: &[DynamicImage],
    image_mean: Option<[f32; 3]>,
    image_std: Option<[f32; 3]>,
) -> Result<(Tensor, Vec<(usize, usize)>)> {
    let mean = image_mean.unwrap_or([0.0, 0.0, 0.0]);
    let std = image_std.unwrap_or([1.0, 1.0, 1.0]);
    let mut tensors = Vec::new();
    let mut patches = Vec::new();
    for image in images {
        let image = image.to_rgb8();
        let (w, h) = image.dimensions();
        patches.push((h as usize, w as usize));
        let data = image.into_raw();
        let data: Vec<f32> = data
            .chunks_exact(3)
            .flat_map(|px| {
                [
                    (px[0] as f32 / 255.0 - mean[0]) / std[0],
                    (px[1] as f32 / 255.0 - mean[1]) / std[1],
                    (px[2] as f32 / 255.0 - mean[2]) / std[2],
                ]
            })
            .collect();
        let tensor =
            Tensor::from_vec(data, (h as usize, w as usize, 3), &candle_core::Device::Cpu)?
                .permute((2, 0, 1))?;
        tensors.push(tensor);
    }
    Ok((
        Tensor::stack(&tensors.iter().collect::<Vec<_>>(), 0)?,
        patches,
    ))
}

impl Qwen3VLImageProcessor {
    pub const DEFAULT_MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];
    pub const DEFAULT_STD: [f32; 3] = [0.26862954, 0.26130258, 0.27577711];

    pub const VISION_START: &str = "<|vision_start|>";
    pub const VISION_END: &str = "<|vision_end|>";
    pub const IMAGE_PAD: &str = "<|image_pad|>";

    /// Resize respecting patch constraints
    fn smart_resize(&self, h: usize, w: usize) -> Result<(usize, usize)> {
        let factor = self.patch_size * self.merge_size;

        let mut nh = (h as f64 / factor as f64).round() as usize * factor;
        let mut nw = (w as f64 / factor as f64).round() as usize * factor;

        let pixels = nh * nw;

        if pixels > self.max_pixels {
            let beta = (pixels as f64 / self.max_pixels as f64).sqrt();
            nh = ((nh as f64 / beta) as usize / factor) * factor;
            nw = ((nw as f64 / beta) as usize / factor) * factor;
        } else if pixels < self.min_pixels {
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
        };
        let image = image
            .resize_exact(nw as u32, nh as u32, self.cfg.resampling.to_filter()?)
            .to_rgb8();

        let image_mean = Some(self.cfg.image_mean.unwrap_or(Self::DEFAULT_MEAN));
        let image_std = Some(self.cfg.image_std.unwrap_or(Self::DEFAULT_STD));

        let (mut patches, _) =
            to_tensor(&vec![DynamicImage::ImageRgb8(image)], image_mean, image_std)?;

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

        Ok((patches, (grid_h as usize, grid_w as usize)))
    }
}

impl ImageProcessTrait for Qwen3VLImageProcessor {
    /// 🔹 Main entry: processes prompt + images together
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

        // ===== Prompt expansion logic (preserved & fixed) =====
        let merge_len = self.merge_size * self.merge_size;
        let mut image_idx = 0;
        let mut replace_strings = Vec::new();
        while prompt.contains(IMAGE_PLACEHOLDER) {
            let grid = grid_thw[image_idx];
            let num_patches: usize = (grid.0 * grid.1) as usize / merge_len;
            let mut replace_tokens = vec![Self::VISION_START];
            replace_tokens.extend(vec![Self::IMAGE_PAD; num_patches]);
            replace_tokens.push(Self::VISION_END);

            replace_strings.push(replace_tokens.join(""));
            *prompt = prompt.replace(IMAGE_PLACEHOLDER, PLACEHOLDER);
            image_idx += 1;
        }

        while prompt.contains(PLACEHOLDER) {
            if let Some(replace_str) = replace_strings.pop() {
                *prompt = prompt.replace(PLACEHOLDER, &replace_str);
            } else {
                break;
            }
        }

        Ok((pixel_values, grid_thw))
    }
}
