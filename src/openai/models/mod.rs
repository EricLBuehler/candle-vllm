pub mod llama;
pub mod phi3;

#[derive(Debug, Clone)]
pub struct Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub use_flash_attn: bool,
    pub rms_norm_eps: f64,
    pub rope_theta: f32,
    pub bos_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    pub max_seq_len: usize,
    pub sliding_window: Option<usize>,
    pub hidden_act: Option<candle_nn::Activation>,
}

impl Config {
    pub fn get_head_size(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}
