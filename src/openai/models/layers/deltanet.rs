// src/models/layers/deltanet.rs
// Shared Qwen3.5/Qwen3Next GatedDeltaNet linear-attention layer.
// Adapted from https://github.com/guoqingbao/vllm.rs/blob/main/src/models/layers/deltanet.rs

use crate::openai::distributed::{
    Comm, MergedParallelColumnLinear, TensorParallelColumnLinear, TensorParallelRowLinear,
    VarBuilder,
};
use crate::openai::models::{resolve_qwen3_hybrid_config, Config};
use attention_rs::gdn;
use attention_rs::mamba_cache::MambaCache;
use attention_rs::InputMetadata;
use candle_core::{DType, Result, Tensor};
use std::rc::Rc;

enum GdnProjection {
    // Qwen3Next: in_proj_qkvz + in_proj_ba
    FusedQkvzBa {
        in_proj_qkvz: TensorParallelColumnLinear,
        in_proj_ba: TensorParallelColumnLinear,
    },
    // Qwen3.5: in_proj_qkv + in_proj_z + in_proj_ba + in_proj_a
    SplitQkvZaLegacy {
        in_proj_qkv: TensorParallelColumnLinear,
        in_proj_z: TensorParallelColumnLinear,
        in_proj_b: TensorParallelColumnLinear,
        in_proj_a: TensorParallelColumnLinear,
    },
    // Qwen3.5 TP-safe split for packed in_proj_qkv [q|k|v].
    SplitQkvZaMerged {
        in_proj_qkv: MergedParallelColumnLinear,
        in_proj_z: TensorParallelColumnLinear,
        in_proj_b: TensorParallelColumnLinear,
        in_proj_a: TensorParallelColumnLinear,
    },
}

pub struct GatedDeltaNet {
    projection: GdnProjection,
    out_proj: TensorParallelRowLinear,
    conv_weight: Tensor,
    conv_bias: Option<Tensor>,
    a_log: Tensor,
    dt_bias: Tensor,
    gdn_norm_weight: Tensor,
    gdn_norm_bias: Option<Tensor>,
    num_k_heads: usize,
    num_v_heads: usize,
    head_k_dim: usize,
    head_v_dim: usize,
    key_dim: usize,
    value_dim: usize,
    kv_group_size: usize,
    gdn_layer_idx: usize,
    rms_norm_eps: f64,
    scale: f64,
}

impl GatedDeltaNet {
    fn load_projection(
        vb: &VarBuilder,
        hidden_size: usize,
        key_dim_global: usize,
        value_dim_global: usize,
        num_v_heads_global: usize,
        comm: Rc<Comm>,
        config: &Config,
        dtype: DType,
        is_fp8_quant: bool,
    ) -> Result<GdnProjection> {
        let (quantization_config, _quant) = if is_fp8_quant {
            (config.quantization_config.clone(), config.isq_quant.clone())
        } else {
            (None, None)
        };
        // Qwen3Next format: fused qkvz + fused ba
        let projection_size_qkvz = key_dim_global * 2 + value_dim_global * 2;
        let projection_size_ba = num_v_heads_global * 2;
        let fused_qkvz = TensorParallelColumnLinear::load_with_hints(
            hidden_size,
            projection_size_qkvz,
            false,
            vb.pp("in_proj_qkvz"),
            comm.clone(),
            &None,
            &quantization_config,
        );

        let fused_ba = TensorParallelColumnLinear::load_with_hints(
            hidden_size,
            projection_size_ba,
            false,
            vb.pp("in_proj_ba"),
            comm.clone(),
            &None,
            &None,
        );

        if let (Ok(in_proj_qkvz), Ok(in_proj_ba)) = (fused_qkvz, fused_ba) {
            // Qwen3 Next projection
            return Ok(GdnProjection::FusedQkvzBa {
                in_proj_qkvz,
                in_proj_ba,
            });
        };

        // Qwen3.5 format: split qkv, z, b, a
        let split_z = TensorParallelColumnLinear::load_with_hints(
            hidden_size,
            value_dim_global,
            false,
            vb.pp("in_proj_z"),
            comm.clone(),
            &None,
            &quantization_config,
        );

        let split_b = TensorParallelColumnLinear::load_with_hints(
            hidden_size,
            num_v_heads_global,
            false,
            vb.pp("in_proj_b"),
            comm.clone(),
            &None,
            &None,
        );
        let split_a = TensorParallelColumnLinear::load_with_hints(
            hidden_size,
            num_v_heads_global,
            false,
            vb.pp("in_proj_a"),
            comm.clone(),
            &None,
            &None,
        );

        if let (Ok(in_proj_z), Ok(in_proj_b), Ok(in_proj_a)) = (split_z, split_b, split_a) {
            if comm.world_size() > 1 {
                // TP-safe path for packed in_proj_qkv [q|k|v]:
                // shard each semantic chunk independently (q, k, v), not as one contiguous block.
                let split_qkv_merged = MergedParallelColumnLinear::load_merged_chunks(
                    hidden_size,
                    key_dim_global * 2 + value_dim_global,
                    0,
                    vec![key_dim_global, key_dim_global, value_dim_global],
                    vb.pp("in_proj_qkv"),
                    comm.clone(),
                    &quantization_config,
                    &None,
                    dtype,
                );

                if let Ok(in_proj_qkv) = split_qkv_merged {
                    return Ok(GdnProjection::SplitQkvZaMerged {
                        in_proj_qkv,
                        in_proj_z,
                        in_proj_b,
                        in_proj_a,
                    });
                }
            }

            // Single GPU (or fallback): use legacy split loader.
            let split_qkv_legacy = TensorParallelColumnLinear::load_with_hints(
                hidden_size,
                key_dim_global * 2 + value_dim_global,
                false,
                vb.pp("in_proj_qkv"),
                comm.clone(),
                &None,
                &quantization_config,
            );

            if let Ok(in_proj_qkv) = split_qkv_legacy {
                return Ok(GdnProjection::SplitQkvZaLegacy {
                    in_proj_qkv,
                    in_proj_z,
                    in_proj_b,
                    in_proj_a,
                });
            }
        }

        candle_core::bail!("Unable to load Qwen3.5/Qwen3Next linear attention projection weights",)
    }

    fn fix_qwen3next_projection_order(
        &self,
        mixed_qkvz: &Tensor,
        mixed_ba: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)> {
        let seq_len = mixed_qkvz.dim(0)?;
        let qkvz_group_dim =
            self.head_k_dim + self.head_k_dim + self.kv_group_size * self.head_v_dim * 2;
        let ba_group_dim = 2 * self.kv_group_size;

        let mixed_qkvz = mixed_qkvz.reshape((seq_len, self.num_k_heads, qkvz_group_dim))?;
        let mixed_ba = mixed_ba.reshape((seq_len, self.num_k_heads, ba_group_dim))?;

        let mut offset = 0usize;
        let query = mixed_qkvz.narrow(2, offset, self.head_k_dim)?;
        offset += self.head_k_dim;
        let key = mixed_qkvz.narrow(2, offset, self.head_k_dim)?;
        offset += self.head_k_dim;
        let value = mixed_qkvz.narrow(2, offset, self.kv_group_size * self.head_v_dim)?;
        offset += self.kv_group_size * self.head_v_dim;
        let z = mixed_qkvz.narrow(2, offset, self.kv_group_size * self.head_v_dim)?;

        let b = mixed_ba.narrow(2, 0, self.kv_group_size)?;
        let a = mixed_ba.narrow(2, self.kv_group_size, self.kv_group_size)?;

        Ok((
            query.reshape((seq_len, self.key_dim))?,
            key.reshape((seq_len, self.key_dim))?,
            value.reshape((seq_len, self.value_dim))?,
            z.reshape((seq_len, self.value_dim))?,
            b.reshape((seq_len, self.num_v_heads))?,
            a.reshape((seq_len, self.num_v_heads))?,
        ))
    }

    fn project_inputs(
        &self,
        xs: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)> {
        match &self.projection {
            GdnProjection::FusedQkvzBa {
                in_proj_qkvz,
                in_proj_ba,
            } => {
                let mixed_qkvz = in_proj_qkvz.forward(xs)?;
                let mixed_ba = in_proj_ba.forward(xs)?;
                self.fix_qwen3next_projection_order(&mixed_qkvz, &mixed_ba)
            }
            GdnProjection::SplitQkvZaLegacy {
                in_proj_qkv,
                in_proj_z,
                in_proj_b,
                in_proj_a,
            } => {
                let proj_qkv = in_proj_qkv.forward(xs)?;
                let q = proj_qkv.narrow(1, 0, self.key_dim)?.contiguous()?;
                let k = proj_qkv
                    .narrow(1, self.key_dim, self.key_dim)?
                    .contiguous()?;
                let v = proj_qkv
                    .narrow(1, self.key_dim * 2, self.value_dim)?
                    .contiguous()?;
                let z = in_proj_z.forward(xs)?;
                let b = in_proj_b.forward(xs)?;
                let a = in_proj_a.forward(xs)?;
                Ok((q, k, v, z, b, a))
            }
            GdnProjection::SplitQkvZaMerged {
                in_proj_qkv,
                in_proj_z,
                in_proj_b,
                in_proj_a,
            } => {
                let qkv = in_proj_qkv.forward(xs)?;
                if qkv.len() != 3 {
                    candle_core::bail!(
                        "Expected 3 chunks from merged in_proj_qkv, got {}",
                        qkv.len()
                    );
                }
                let q = qkv[0].clone();
                let k = qkv[1].clone();
                let v = qkv[2].clone();
                // z/b/a are projected from the original hidden states, not q/k/v chunks.
                let z = in_proj_z.forward(xs)?;
                let b = in_proj_b.forward(xs)?;
                let a = in_proj_a.forward(xs)?;
                Ok((q, k, v, z, b, a))
            }
        }
    }

    fn repeat_kv_heads(&self, x: Tensor) -> Result<Tensor> {
        if self.num_k_heads == self.num_v_heads {
            return Ok(x);
        }
        let (seq_len, _h, _d) = x.dims3()?;
        x.unsqueeze(2)?
            .broadcast_as((
                seq_len,
                self.num_k_heads,
                self.kv_group_size,
                self.head_k_dim,
            ))?
            .reshape((seq_len, self.num_v_heads, self.head_k_dim))
    }

    pub fn new(
        vb: VarBuilder,
        comm: Rc<Comm>,
        config: &Config,
        gdn_layer_idx: usize,
    ) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let hybrid = resolve_qwen3_hybrid_config(config);
        let world_size = comm.world_size();
        let rank = comm.rank();

        let num_v_heads_global = hybrid.num_v_heads;
        let num_k_heads_global = hybrid.num_k_heads;
        if num_v_heads_global % num_k_heads_global != 0 {
            candle_core::bail!(
                "linear_num_value_heads ({}) must be divisible by linear_num_key_heads ({})",
                num_v_heads_global,
                num_k_heads_global
            );
        }
        if num_v_heads_global % world_size != 0 || num_k_heads_global % world_size != 0 {
            candle_core::bail!(
                "linear attention heads must be divisible by tensor parallel world_size (num_v_heads={}, num_k_heads={}, world_size={})",
                num_v_heads_global,
                num_k_heads_global,
                world_size
            );
        }

        let is_fp8_quant = if let Some(cfg) = config.quantization_config.as_ref() {
            cfg.quant_method == "fp8"
        } else {
            false
        };
        let dtype = vb.dtype();

        let num_v_heads = num_v_heads_global / world_size;
        let num_k_heads = num_k_heads_global / world_size;
        let head_k_dim = hybrid.key_head_dim;
        let head_v_dim = hybrid.value_head_dim;
        let key_dim_global = num_k_heads_global * head_k_dim;
        let value_dim_global = num_v_heads_global * head_v_dim;
        let key_dim = num_k_heads * head_k_dim;
        let value_dim = num_v_heads * head_v_dim;
        let kv_group_size = num_v_heads / num_k_heads;
        let conv_kernel_size = hybrid.conv_kernel_size;
        let conv_dim_global = key_dim_global * 2 + value_dim_global;

        // Learned GDN parameters
        let a_log = vb
            .get((num_v_heads_global,), "A_log")?
            .narrow(0, rank * num_v_heads, num_v_heads)?
            .contiguous()?;
        let dt_bias = vb
            .get((num_v_heads_global,), "dt_bias")?
            .narrow(0, rank * num_v_heads, num_v_heads)?
            .contiguous()?;

        let projection = Self::load_projection(
            &vb,
            hidden_size,
            key_dim_global,
            value_dim_global,
            num_v_heads_global,
            comm.clone(),
            config,
            dtype,
            is_fp8_quant,
        )?;

        // Conv1D weights are stored global; slice rank-local q/k/v channel blocks.
        let conv_weight = vb.get((conv_dim_global, 1, conv_kernel_size), "conv1d.weight")?;
        let q_start = rank * key_dim;
        let k_start = key_dim_global + rank * key_dim;
        let v_start = key_dim_global * 2 + rank * value_dim;
        let q_w = conv_weight.narrow(0, q_start, key_dim)?;
        let k_w = conv_weight.narrow(0, k_start, key_dim)?;
        let v_w = conv_weight.narrow(0, v_start, value_dim)?;
        let conv_weight = Tensor::cat(&[&q_w, &k_w, &v_w], 0)?;

        let conv_bias = vb.get((conv_dim_global,), "conv1d.bias").ok();
        let conv_bias = if let Some(cb) = conv_bias {
            let q_b = cb.narrow(0, q_start, key_dim)?;
            let k_b = cb.narrow(0, k_start, key_dim)?;
            let v_b = cb.narrow(0, v_start, value_dim)?;
            Some(Tensor::cat(&[&q_b, &k_b, &v_b], 0)?)
        } else {
            None
        };

        // Output projection
        let out_proj = TensorParallelRowLinear::load_with_hints(
            value_dim_global,
            hidden_size,
            false,
            vb.pp("out_proj"),
            comm.clone(),
            &None,
            if is_fp8_quant {
                &config.quantization_config
            } else {
                &None
            },
        )?;

        // GDN output norm (gated RMSNorm): both Qwen3.5 and Qwen3Next use per-head params.
        let gdn_norm_weight = vb.get((head_v_dim,), "norm.weight").map_err(|err| {
            candle_core::Error::Msg(format!(
                "Unable to load linear_attn.norm.weight as per-head [{head_v_dim}]: {err}"
            ))
        })?;
        let gdn_norm_bias = vb.get((head_v_dim,), "norm.bias").ok();
        let scale = 1.0f64 / (head_k_dim as f64).sqrt();
        Ok(Self {
            projection,
            out_proj,
            conv_weight,
            conv_bias,
            a_log,
            dt_bias,
            gdn_norm_weight,
            gdn_norm_bias,
            num_k_heads,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            key_dim,
            value_dim,
            kv_group_size,
            gdn_layer_idx,
            rms_norm_eps: config.rms_norm_eps,
            scale,
        })
    }

    pub fn forward(
        &self,
        xs: &Tensor,
        mamba_cache: &mut MambaCache,
        input_metadata: &InputMetadata,
        seq_slots: &Tensor,
    ) -> Result<Tensor> {
        let slot_count = seq_slots.dim(0)?;
        if slot_count == 0 {
            candle_core::bail!("Linear attention requires non-empty sequence slots");
        }
        let (token_count, _hidden) = xs.dims2()?;
        let is_prefill = input_metadata.is_prefill;

        let (q, k, v, z, b, a) = self.project_inputs(xs)?;
        let mixed_qkv = Tensor::cat(&[&q, &k, &v], 1)?;

        let (kv_conv, prefill_conv_state) = if is_prefill {
            let mut conv_state = mamba_cache.get_batch_conv_state(self.gdn_layer_idx, seq_slots)?;
            let cu_seqlens = input_metadata
                .cu_seqlens_q
                .as_ref()
                .expect("cu_seqlens_q must be present in prefill!");

            let out = gdn::causal_conv1d_fwd(
                &mixed_qkv,
                &self.conv_weight,
                self.conv_bias.as_ref(),
                &mut conv_state,
                Some(cu_seqlens),
                true, // SiLU activation
            )?;
            (out, Some(conv_state))
        } else {
            if token_count != slot_count {
                candle_core::bail!(
                    "Linear attention decode mismatch: {} tokens vs {} sequence slots",
                    token_count,
                    slot_count
                );
            }
            let out = gdn::causal_conv1d_update_slots(
                &mixed_qkv,
                &self.conv_weight,
                self.conv_bias.as_ref(),
                mamba_cache.conv_state_mut(self.gdn_layer_idx),
                seq_slots,
                true,
            )?;
            (out, None)
        };
        if let Some(conv_state) = prefill_conv_state {
            mamba_cache.set_batch_conv_state(self.gdn_layer_idx, seq_slots, &conv_state)?;
        }

        // Split convolved output back into q', k', v'
        let q_conv = kv_conv.narrow(1, 0, self.key_dim)?;
        let k_conv = kv_conv.narrow(1, self.key_dim, self.key_dim)?;
        let v_conv = kv_conv.narrow(1, self.key_dim * 2, self.value_dim)?;

        // Fused GDN gating
        let (a_expanded, b_expanded) = (a.unsqueeze(0)?, b.unsqueeze(0)?); // [1, seq_len, num_heads]
        let (g, beta) =
            gdn::fused_gdn_gating(&self.a_log, &a_expanded, &b_expanded, &self.dt_bias)?;
        let (g, beta) = (g.squeeze(0)?, beta.squeeze(0)?);

        let q = q_conv.reshape((token_count, self.num_k_heads, self.head_k_dim))?;
        let k = k_conv.reshape((token_count, self.num_k_heads, self.head_k_dim))?;
        let v = v_conv.reshape((token_count, self.num_v_heads, self.head_v_dim))?;
        let q = gdn::l2_norm_last_dim(&q, 1e-6)?;
        let k = gdn::l2_norm_last_dim(&k, 1e-6)?;
        let (q, k) = (self.repeat_kv_heads(q)?, self.repeat_kv_heads(k)?);

        let output = if is_prefill {
            // S1: Use batched varlen recurrence — one CUDA launch for all sequences
            let q_scaled = (&q * self.scale)?;

            let cu_seqlens = input_metadata
                .cu_seqlens_q
                .as_ref()
                .expect("cu_seqlens_q must be present in prefill!");

            // Get mutable reference to global state for in-place update (optimized prefill)
            let global_state = mamba_cache.recurrent_state_mut(self.gdn_layer_idx);

            gdn::gated_delta_rule_recurrence_varlen(
                &q_scaled,
                &k,
                &v,
                &g,
                &beta,
                global_state,
                seq_slots,
                &cu_seqlens,
            )?
        } else {
            let batch = slot_count;
            let q_b = (q.reshape((batch, self.num_v_heads, self.head_k_dim))? * self.scale)?;
            let k_b = k.reshape((batch, self.num_v_heads, self.head_k_dim))?;
            let v_b = v.reshape((batch, self.num_v_heads, self.head_v_dim))?;
            let g_b = g.reshape((batch, self.num_v_heads))?;
            let beta_b = beta.reshape((batch, self.num_v_heads))?;
            let global_state = mamba_cache.recurrent_state_mut(self.gdn_layer_idx);
            gdn::gated_delta_rule_decode_slots(
                &q_b,
                &k_b,
                &v_b,
                &g_b,
                &beta_b,
                global_state,
                seq_slots,
            )?
        };

        // output: [seq_len, num_v_heads, head_v_dim] -> [seq_len, value_dim]
        let output = output.reshape((token_count, self.value_dim))?;

        // Gated RMSNorm: norm(output) * silu(z) via fused kernel
        let gated_output = gdn::gated_rmsnorm_silu_mul(
            &output,
            &z,
            &self.gdn_norm_weight,
            self.gdn_norm_bias.as_ref(),
            self.rms_norm_eps,
            self.head_v_dim,
        )?;

        // Output projection
        self.out_proj.forward(&gated_output.to_dtype(xs.dtype())?)
    }
}
