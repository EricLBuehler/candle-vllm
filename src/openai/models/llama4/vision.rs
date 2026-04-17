use super::config::VisionConfig;
use crate::openai::distributed::{ReplicatedLinear, VarBuilder};
use crate::openai::models::layers::others::{layer_norm, NormX};
use candle_core::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::{Activation, Module};

struct Llama4UnfoldConvolution {
    linear: ReplicatedLinear,
    kernel_size: usize,
    patch_size: usize,
}

impl Llama4UnfoldConvolution {
    fn new(cfg: &VisionConfig, vb: VarBuilder, _dtype: DType) -> Result<Self> {
        let kernel_size = cfg.patch_size;
        let linear = ReplicatedLinear::load_no_bias(
            cfg.num_channels * kernel_size * kernel_size,
            cfg.hidden_size,
            vb.pp("linear"),
            &None,
            &None,
        )?;
        Ok(Self {
            linear,
            kernel_size,
            patch_size: cfg.patch_size,
        })
    }

    fn unfold(&self, xs: &Tensor) -> Result<Tensor> {
        let kernel_size = (self.kernel_size, self.kernel_size);
        let stride = (self.patch_size, self.patch_size);
        let (bs, c, h, w) = xs.dims4()?;

        let h_out = (h - kernel_size.0) / stride.0 + 1;
        let w_out = (w - kernel_size.1) / stride.1 + 1;

        let mut blocks = Vec::new();
        for i in (0..=h - kernel_size.0).step_by(stride.0) {
            for j in (0..=w - kernel_size.1).step_by(stride.1) {
                let mut block = Vec::new();
                for di in 0..kernel_size.0 {
                    for dj in 0..kernel_size.1 {
                        let h_idx = i + di;
                        let w_idx = j + dj;
                        block.push(xs.i((.., .., h_idx, w_idx))?);
                    }
                }
                let mut block = Tensor::stack(&block, 1)?;
                block = block.permute((0, 2, 1))?;
                blocks.push(block);
            }
        }

        let _ = (h_out, w_out);
        let mut result = Tensor::stack(&blocks, D::Minus1)?;
        result = result.reshape((bs, c * kernel_size.0 * kernel_size.1, h_out * w_out))?;
        Ok(result)
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let mut hidden_states = self.unfold(hidden_states)?;
        hidden_states = hidden_states.transpose(1, 2)?;
        self.linear.forward(&hidden_states)
    }
}

#[derive(Clone)]
struct Llama4VisionRotaryEmbedding {
    cos: Tensor,
    sin: Tensor,
}

impl Llama4VisionRotaryEmbedding {
    fn new(cfg: &VisionConfig, device: &Device, dtype: DType) -> Result<Self> {
        let idx = cfg.image_size / cfg.patch_size;
        let mut img_idx =
            Tensor::arange(0f32, idx.pow(2) as f32, device)?.reshape((idx.pow(2), 1))?;
        img_idx = Tensor::cat(&[&img_idx, &img_idx.narrow(0, 0, 1)?], 0)?;
        img_idx = img_idx.slice_assign(
            &[
                img_idx.dim(0)? - 1..img_idx.dim(0)?,
                img_idx.dim(1)? - 1..img_idx.dim(1)?,
            ],
            &Tensor::new(-2f32, device)?.reshape((1, 1))?,
        )?;
        let img_ids_flat = img_idx.flatten_all()?.to_vec1::<f32>()?;

        let frequencies_x = {
            let freqs: Vec<f32> = img_ids_flat.iter().map(|x| x % idx as f32).collect();
            Tensor::from_vec(freqs, img_idx.shape().clone(), device)?
        };
        let frequencies_y = {
            let freqs: Vec<f32> = img_ids_flat
                .iter()
                .map(|x| (x / idx as f32).floor())
                .collect();
            Tensor::from_vec(freqs, img_idx.shape().clone(), device)?
        };

        let rope_freq = {
            let freq_dim = cfg.hidden_size / cfg.num_attention_heads / 2;
            let freqs: Vec<f32> = (0..freq_dim)
                .step_by(2)
                .take(freq_dim / 2)
                .map(|i| 1f32 / cfg.rope_theta.powf(i as f32 / freq_dim as f32))
                .collect();
            let freqs_len = freqs.len();
            Tensor::from_vec(freqs, freqs_len, device)?
        };

        let freqs_x = repeat_interleave(
            &(frequencies_x + 1.)?
                .unsqueeze(D::Minus1)?
                .broadcast_mul(&rope_freq.unsqueeze(0)?.unsqueeze(0)?)?,
            2,
            D::Minus1,
        )?;
        let freqs_y = repeat_interleave(
            &(frequencies_y + 1.)?
                .unsqueeze(D::Minus1)?
                .broadcast_mul(&rope_freq.unsqueeze(0)?.unsqueeze(0)?)?,
            2,
            D::Minus1,
        )?;

        let mut freqs = {
            let freqs = Tensor::cat(&[freqs_x, freqs_y], D::Minus1)?.contiguous()?;
            let indices_every_two = Tensor::new(
                (0..freqs.dim(D::Minus1)?)
                    .step_by(2)
                    .map(|x| x as u32)
                    .collect::<Vec<_>>(),
                device,
            )?;
            freqs.index_select(&indices_every_two, D::Minus1)?
        };
        freqs = freqs.squeeze(1)?;
        freqs = freqs.lt(0.)?.where_cond(&freqs.zeros_like()?, &freqs)?;

        Ok(Self {
            cos: freqs.cos()?.to_dtype(dtype)?,
            sin: freqs.sin()?.to_dtype(dtype)?,
        })
    }
}

fn repeat_interleave(xs: &Tensor, repeats: usize, dim: D) -> Result<Tensor> {
    let dim_idx = xs.dim(dim)?;
    let mut pieces = Vec::with_capacity(dim_idx * repeats);
    for i in 0..dim_idx {
        let slice = xs.narrow(dim, i, 1)?;
        for _ in 0..repeats {
            pieces.push(slice.clone());
        }
    }
    Tensor::cat(&pieces, dim)
}

struct Llama4VisionAttention {
    q_proj: ReplicatedLinear,
    k_proj: ReplicatedLinear,
    v_proj: ReplicatedLinear,
    o_proj: ReplicatedLinear,
    head_dim: usize,
    num_heads: usize,
    scale: f64,
    freqs: Llama4VisionRotaryEmbedding,
}

impl Llama4VisionAttention {
    fn new(
        cfg: &VisionConfig,
        vb: VarBuilder,
        freqs: Llama4VisionRotaryEmbedding,
        _dtype: DType,
    ) -> Result<Self> {
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;
        let proj_size = cfg.num_attention_heads * head_dim;
        let q_proj = ReplicatedLinear::load_b(
            cfg.hidden_size,
            proj_size,
            true,
            vb.pp("q_proj"),
            &None,
            &None,
        )?;
        let k_proj = ReplicatedLinear::load_b(
            cfg.hidden_size,
            proj_size,
            true,
            vb.pp("k_proj"),
            &None,
            &None,
        )?;
        let v_proj = ReplicatedLinear::load_b(
            cfg.hidden_size,
            proj_size,
            true,
            vb.pp("v_proj"),
            &None,
            &None,
        )?;
        let o_proj = ReplicatedLinear::load_b(
            proj_size,
            cfg.hidden_size,
            true,
            vb.pp("o_proj"),
            &None,
            &None,
        )?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            head_dim,
            num_heads: cfg.num_attention_heads,
            scale: 1.0 / (head_dim as f64).sqrt(),
            freqs,
        })
    }

    fn forward(&self, hidden_state: &Tensor) -> Result<Tensor> {
        let (bs, seq_len, _) = hidden_state.dims3()?;

        let mut q = self.q_proj.forward(hidden_state)?;
        let mut k = self.k_proj.forward(hidden_state)?;
        let v = self.v_proj.forward(hidden_state)?;

        q = q
            .reshape((bs, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        k = k
            .reshape((bs, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((bs, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        q = candle_nn::rotary_emb::rope_i(&q, &self.freqs.cos, &self.freqs.sin)?;
        k = candle_nn::rotary_emb::rope_i(&k, &self.freqs.cos, &self.freqs.sin)?;

        let attn_weights = (q.matmul(&k.t()?)? * self.scale)?;
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights
            .matmul(&v)?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((bs, seq_len, ()))?;

        self.o_proj.forward(&attn_output)
    }
}

struct Llama4VisionMlp {
    fc1: ReplicatedLinear,
    fc2: ReplicatedLinear,
    act: Activation,
}

impl Llama4VisionMlp {
    fn new(cfg: &VisionConfig, vb: VarBuilder, _dtype: DType) -> Result<Self> {
        let act = match cfg.hidden_act.as_deref() {
            Some("gelu") | None => Activation::Gelu,
            Some("silu") => Activation::Silu,
            Some(other) => candle_core::bail!("Unsupported vision activation: {other}"),
        };
        let fc1 = ReplicatedLinear::load_b(
            cfg.hidden_size,
            cfg.intermediate_size,
            true,
            vb.pp("fc1"),
            &None,
            &None,
        )?;
        let fc2 = ReplicatedLinear::load_b(
            cfg.intermediate_size,
            cfg.hidden_size,
            true,
            vb.pp("fc2"),
            &None,
            &None,
        )?;
        Ok(Self { fc1, fc2, act })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.fc2.forward(&self.act.forward(&self.fc1.forward(xs)?)?)
    }
}

struct Llama4VisionEncoderLayer {
    self_attn: Llama4VisionAttention,
    mlp: Llama4VisionMlp,
    input_layernorm: NormX,
    post_attention_layernorm: NormX,
}

impl Llama4VisionEncoderLayer {
    fn new(
        cfg: &VisionConfig,
        vb: VarBuilder,
        freqs: Llama4VisionRotaryEmbedding,
        dtype: DType,
    ) -> Result<Self> {
        let self_attn = Llama4VisionAttention::new(cfg, vb.pp("self_attn"), freqs, dtype)?;
        let mlp = Llama4VisionMlp::new(cfg, vb.pp("mlp"), dtype)?;
        let input_layernorm = layer_norm(
            cfg.hidden_size,
            cfg.norm_eps,
            true,
            vb.pp("input_layernorm"),
            dtype,
        )?;
        let post_attention_layernorm = layer_norm(
            cfg.hidden_size,
            cfg.norm_eps,
            true,
            vb.pp("post_attention_layernorm"),
            dtype,
        )?;
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    fn forward(&self, hidden_state: &Tensor) -> Result<Tensor> {
        let residual = hidden_state;
        let hidden_state = self.input_layernorm.forward(hidden_state)?;
        let hidden_state = (self.self_attn.forward(&hidden_state)? + residual)?;
        let residual = hidden_state.clone();
        let hidden_state = self.post_attention_layernorm.forward(&hidden_state)?;
        let hidden_state = self.mlp.forward(&hidden_state)?;
        residual + hidden_state
    }
}

struct Llama4VisionEncoder {
    layers: Vec<Llama4VisionEncoderLayer>,
}

impl Llama4VisionEncoder {
    fn new(
        cfg: &VisionConfig,
        vb: VarBuilder,
        freqs: Llama4VisionRotaryEmbedding,
        dtype: DType,
    ) -> Result<Self> {
        let layers_vb = vb.pp("layers");
        let mut layers = Vec::new();
        for i in 0..cfg.num_hidden_layers {
            layers.push(Llama4VisionEncoderLayer::new(
                cfg,
                layers_vb.pp(i.to_string().as_str()),
                freqs.clone(),
                dtype,
            )?);
        }
        Ok(Self { layers })
    }

    fn forward(&self, hidden_state: &Tensor) -> Result<Tensor> {
        let mut hidden_state = hidden_state.clone();
        for layer in &self.layers {
            hidden_state = layer.forward(&hidden_state)?;
        }
        Ok(hidden_state)
    }
}

struct Llama4VisionPixelShuffleMLP {
    fc1: ReplicatedLinear,
    fc2: ReplicatedLinear,
    act: Activation,
}

impl Llama4VisionPixelShuffleMLP {
    fn new(cfg: &VisionConfig, vb: VarBuilder, _dtype: DType) -> Result<Self> {
        let fc1 = ReplicatedLinear::load_no_bias(
            cfg.intermediate_size,
            cfg.projector_in_dim(),
            vb.pp("fc1"),
            &None,
            &None,
        )?;
        let fc2 = ReplicatedLinear::load_no_bias(
            cfg.projector_out_dim(),
            cfg.projector_out_dim(),
            vb.pp("fc2"),
            &None,
            &None,
        )?;
        Ok(Self {
            fc1,
            fc2,
            act: Activation::Gelu,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.act.forward(
            &self
                .fc2
                .forward(&self.act.forward(&self.fc1.forward(xs)?)?)?,
        )
    }
}

struct Llama4VisionPixelShuffle {
    mlp: Llama4VisionPixelShuffleMLP,
    pixel_shuffle_ratio: f32,
}

impl Llama4VisionPixelShuffle {
    fn new(cfg: &VisionConfig, vb: VarBuilder, dtype: DType) -> Result<Self> {
        let mlp = Llama4VisionPixelShuffleMLP::new(cfg, vb.pp("mlp"), dtype)?;
        Ok(Self {
            mlp,
            pixel_shuffle_ratio: cfg.pixel_shuffle_ratio,
        })
    }

    fn pixel_shuffle(&self, xs: &Tensor) -> Result<Tensor> {
        let (bs, num_patches, _c) = xs.dims3()?;
        let patch_size = (num_patches as f32).sqrt() as usize;

        let mut xs = xs.reshape((bs, patch_size, patch_size, ()))?;
        let (_bs, h, w, c) = xs.dims4()?;

        xs = xs.reshape((
            bs,
            h,
            (w as f32 * self.pixel_shuffle_ratio) as usize,
            (c as f32 / self.pixel_shuffle_ratio) as usize,
        ))?;
        xs = xs.permute((0, 2, 1, 3))?.contiguous()?;

        xs = xs.reshape((
            bs,
            (h as f32 * self.pixel_shuffle_ratio) as usize,
            (w as f32 * self.pixel_shuffle_ratio) as usize,
            (c as f32 / self.pixel_shuffle_ratio.powi(2)) as usize,
        ))?;
        xs = xs.permute((0, 2, 1, 3))?.contiguous()?;

        xs.reshape((bs, (), xs.dim(D::Minus1)?))
    }

    fn forward(&self, encoded_patches: &Tensor) -> Result<Tensor> {
        let encoded_patches = self.pixel_shuffle(encoded_patches)?;
        self.mlp.forward(&encoded_patches)
    }
}

pub struct Llama4VisionModel {
    patch_embedding: Llama4UnfoldConvolution,
    class_embedding: Tensor,
    positional_embedding_vlm: Tensor,
    layernorm_pre: NormX,
    layernorm_post: NormX,
    encoder: Llama4VisionEncoder,
    vision_adapter: Llama4VisionPixelShuffle,
}

impl Llama4VisionModel {
    pub fn new(cfg: &VisionConfig, vb: VarBuilder, dtype: DType, device: &Device) -> Result<Self> {
        let patch_embedding = Llama4UnfoldConvolution::new(cfg, vb.pp("patch_embedding"), dtype)?;

        let class_embedding = vb
            .get(cfg.hidden_size, "class_embedding")?
            .to_device(device)?;
        let num_patches = cfg.num_patches();
        let positional_embedding_vlm = vb
            .get((num_patches, cfg.hidden_size), "positional_embedding_vlm")?
            .to_device(device)?;

        let layernorm_pre = layer_norm(
            cfg.hidden_size,
            cfg.norm_eps,
            true,
            vb.pp("layernorm_pre"),
            dtype,
        )?;
        let layernorm_post = layer_norm(
            cfg.hidden_size,
            cfg.norm_eps,
            true,
            vb.pp("layernorm_post"),
            dtype,
        )?;

        let rotary_embedding = Llama4VisionRotaryEmbedding::new(cfg, device, dtype)?;
        let encoder = Llama4VisionEncoder::new(cfg, vb.pp("model"), rotary_embedding, dtype)?;
        let vision_adapter = Llama4VisionPixelShuffle::new(cfg, vb.pp("vision_adapter"), dtype)?;

        Ok(Self {
            patch_embedding,
            class_embedding,
            positional_embedding_vlm,
            layernorm_pre,
            layernorm_post,
            encoder,
            vision_adapter,
        })
    }

    pub fn forward(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let pixel_values = pixel_values.to_dtype(self.class_embedding.dtype())?;
        let (bs_times_num_tiles, _num_channels, _height, _width) = pixel_values.dims4()?;

        let mut hidden_state = self.patch_embedding.forward(&pixel_values)?;
        let (_, mut num_patches, hidden_dim) = hidden_state.dims3()?;

        hidden_state = hidden_state.reshape((bs_times_num_tiles, num_patches, hidden_dim))?;
        let class_embedding =
            self.class_embedding
                .expand((hidden_state.dim(0)?, 1, hidden_state.dim(D::Minus1)?))?;
        hidden_state = Tensor::cat(&[hidden_state, class_embedding], 1)?;
        num_patches += 1;

        hidden_state = hidden_state.reshape((bs_times_num_tiles, num_patches, hidden_dim))?;
        hidden_state = hidden_state.broadcast_add(&self.positional_embedding_vlm)?;

        hidden_state = self.layernorm_pre.forward(&hidden_state)?;
        hidden_state = hidden_state.reshape((bs_times_num_tiles, (), hidden_dim))?;

        hidden_state = self.encoder.forward(&hidden_state)?;
        hidden_state = self.layernorm_post.forward(&hidden_state)?;

        hidden_state = hidden_state.narrow(1, 0, hidden_state.dim(1)? - 1)?;

        self.vision_adapter.forward(&hidden_state)
    }
}
