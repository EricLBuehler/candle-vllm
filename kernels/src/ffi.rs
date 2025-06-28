use core::ffi::{c_int, c_long, c_void};
#[allow(dead_code)]
extern "C" {
    pub fn call_reshape_and_cache(
        key: *const c_void,
        value: *const c_void,
        key_cache: *const c_void,
        value_cache: *const c_void,
        slot_mapping: *const c_long,

        num_tokens: c_int,
        num_heads: c_int,
        head_size: c_int,
        block_size: c_int,
        x: c_int,
        key_stride: c_int,
        value_stride: c_int,
        dtype: u32,
        stream: i64,
    );

    pub fn paged_attention_v1(
        out: *const c_void,
        query: *const c_void,
        key_cache: *const c_void,
        value_cache: *const c_void,
        num_kv_heads: c_int,
        scale: f32,
        block_tables: *const c_int,
        context_lens: *const c_int,
        block_size: c_int,
        max_context_len: c_int,

        num_seqs: c_int,
        num_heads: c_int,
        head_size: c_int,
        max_num_blocks_per_seq: c_int,
        q_stride: c_int,
        kv_block_stride: c_int,
        kv_head_stride: c_int,

        dtype: u32,
        softscapping: f32,
        stream: i64,
    );

    pub fn paged_attention_v2(
        out: *const c_void,
        exp_sums: *const f32,
        max_logits: *const f32,
        tmp_out: *const c_void,
        query: *const c_void,
        key_cache: *const c_void,
        value_cache: *const c_void,
        num_kv_heads: c_int,
        scale: f32,
        block_tables: *const c_int,
        context_lens: *const c_int,
        block_size: c_int,
        max_context_len: c_int,

        num_seqs: c_int,
        num_heads: c_int,
        head_size: c_int,
        max_num_blocks_per_seq: c_int,
        q_stride: c_int,
        kv_block_stride: c_int,
        kv_head_stride: c_int,

        dtype: u32,
        softscapping: f32,
        stream: i64,
    );

    pub fn marlin_4bit_f16(
        inputs: *const c_void,
        weight: *const c_int,
        scales: *const c_void,
        zeros: *const c_void,
        g_idx: *const c_void,
        out: *mut c_void,
        m: c_int,
        k: c_int,
        n: c_int,
        workspace: *const c_void,
        groupsize: c_int,
        stream: i64,
    );

    pub fn marlin_4bit_bf16(
        inputs: *const c_void,
        weight: *const c_int,
        scales: *const c_void,
        zeros: *const c_void,
        g_idx: *const c_void,
        out: *mut c_void,
        m: c_int,
        k: c_int,
        n: c_int,
        workspace: *const c_void,
        groupsize: c_int,
        stream: i64,
    );

    pub fn marlin_awq_4bit_f16(
        inputs: *const c_void,
        weight: *const c_int,
        scales: *const c_void,
        zeros: *const c_void,
        g_idx: *const c_void,
        out: *mut c_void,
        m: c_int,
        k: c_int,
        n: c_int,
        workspace: *const c_void,
        groupsize: c_int,
        stream: i64,
    );

    pub fn marlin_awq_4bit_bf16(
        inputs: *const c_void,
        weight: *const c_int,
        scales: *const c_void,
        zeros: *const c_void,
        g_idx: *const c_void,
        out: *mut c_void,
        m: c_int,
        k: c_int,
        n: c_int,
        workspace: *const c_void,
        groupsize: c_int,
        stream: i64,
    );
    pub fn gptq_repack(
        weight: *const c_void,
        result: *const c_void,
        m: c_int,
        n: c_int,
        stream: i64,
    );

    pub fn awq_repack(
        weight: *const c_void,
        result: *const c_void,
        k: c_int,
        n: c_int,
        bits: c_int,
        stream: i64,
    );

    pub fn gemm_half_q_half_alt(
        a: *const c_void,
        weight: *const u32,
        qzeros: *const u32,
        scales: *const c_void,
        g_idx: *const i32,
        out: *mut c_void,
        m: i32,
        n: i32,
        k: i32,
        bit: i32,
        stream: i64,
    );

    pub fn count_nonzero_bf16(d_in: *const c_void, N: u32) -> u32;
    pub fn count_nonzero_f16(d_in: *const c_void, N: u32) -> u32;
    pub fn count_nonzero_f32(d_in: *const c_void, N: u32) -> u32;
    pub fn count_nonzero_f64(d_in: *const c_void, N: u32) -> u32;
    pub fn count_nonzero_u8(d_in: *const c_void, N: u32) -> u32;
    pub fn count_nonzero_u32(d_in: *const c_void, N: u32) -> u32;
    pub fn count_nonzero_i16(d_in: *const c_void, N: u32) -> u32;
    pub fn count_nonzero_i64(d_in: *const c_void, N: u32) -> u32;
    pub fn count_nonzero_i32(d_in: *const c_void, N: u32) -> u32;
    pub fn nonzero_bf16(
        d_in: *const c_void,
        N: u32,
        num_nonzero: u32,
        dims: *const c_void,
        num_dims: u32,
        d_out: *mut c_void,
    );
    pub fn nonzero_f16(
        d_in: *const c_void,
        N: u32,
        num_nonzero: u32,
        dims: *const c_void,
        num_dims: u32,
        d_out: *mut c_void,
    );
    pub fn nonzero_f32(
        d_in: *const c_void,
        N: u32,
        num_nonzero: u32,
        dims: *const c_void,
        num_dims: u32,
        d_out: *mut c_void,
    );
    pub fn nonzero_f64(
        d_in: *const c_void,
        N: u32,
        num_nonzero: u32,
        dims: *const c_void,
        num_dims: u32,
        d_out: *mut c_void,
    );
    pub fn nonzero_u8(
        d_in: *const c_void,
        N: u32,
        num_nonzero: u32,
        dims: *const c_void,
        num_dims: u32,
        d_out: *mut c_void,
    );
    pub fn nonzero_u32(
        d_in: *const c_void,
        N: u32,
        num_nonzero: u32,
        dims: *const c_void,
        num_dims: u32,
        d_out: *mut c_void,
    );
    pub fn nonzero_i64(
        d_in: *const c_void,
        N: u32,
        num_nonzero: u32,
        dims: *const c_void,
        num_dims: u32,
        d_out: *mut c_void,
    );
    pub fn nonzero_i16(
        d_in: *const c_void,
        N: u32,
        num_nonzero: u32,
        dims: *const c_void,
        num_dims: u32,
        d_out: *mut c_void,
    );
    pub fn nonzero_i32(
        d_in: *const c_void,
        N: u32,
        num_nonzero: u32,
        dims: *const c_void,
        num_dims: u32,
        d_out: *mut c_void,
    );

    pub fn bitwise_and_u8(d_in1: *const c_void, d_in2: *const c_void, d_out: *mut c_void, N: u32);
    pub fn bitwise_and_u32(d_in1: *const c_void, d_in2: *const c_void, d_out: *mut c_void, N: u32);
    pub fn bitwise_and_i64(d_in1: *const c_void, d_in2: *const c_void, d_out: *mut c_void, N: u32);
    pub fn bitwise_and_i32(d_in1: *const c_void, d_in2: *const c_void, d_out: *mut c_void, N: u32);
    pub fn bitwise_or_u8(d_in1: *const c_void, d_in2: *const c_void, d_out: *mut c_void, N: u32);
    pub fn bitwise_or_u32(d_in1: *const c_void, d_in2: *const c_void, d_out: *mut c_void, N: u32);
    pub fn bitwise_or_i64(d_in1: *const c_void, d_in2: *const c_void, d_out: *mut c_void, N: u32);
    pub fn bitwise_or_i32(d_in1: *const c_void, d_in2: *const c_void, d_out: *mut c_void, N: u32);
    pub fn bitwise_xor_u8(d_in1: *const c_void, d_in2: *const c_void, d_out: *mut c_void, N: u32);
    pub fn bitwise_xor_u32(d_in1: *const c_void, d_in2: *const c_void, d_out: *mut c_void, N: u32);
    pub fn bitwise_xor_i64(d_in1: *const c_void, d_in2: *const c_void, d_out: *mut c_void, N: u32);
    pub fn bitwise_xor_i32(d_in1: *const c_void, d_in2: *const c_void, d_out: *mut c_void, N: u32);
    // Linked to in mistralrs-quant
    pub fn leftshift_u8(d_in1: *const c_void, d_out: *mut c_void, N: u32, k: i32);
    pub fn leftshift_u32(d_in1: *const c_void, d_out: *mut c_void, N: u32, k: i32);
    pub fn leftshift_i64(d_in1: *const c_void, d_out: *mut c_void, N: u32, k: i32);
    pub fn leftshift_i32(d_in1: *const c_void, d_out: *mut c_void, N: u32, k: i32);

    pub fn asort_asc_f32(
        x: *const c_void,
        dst: *mut c_void,
        nrows: i32,
        ncols: i32,
        inplace: bool,
        stream: i64,
    );
    pub fn asort_asc_f16(
        x: *const c_void,
        dst: *mut c_void,
        nrows: i32,
        ncols: i32,
        inplace: bool,
        stream: i64,
    );
    pub fn asort_asc_bf16(
        x: *const c_void,
        dst: *const c_void,
        nrows: i32,
        ncols: i32,
        inplace: bool,
        stream: i64,
    );
    pub fn asort_asc_f64(
        x: *const c_void,
        dst: *mut c_void,
        nrows: i32,
        ncols: i32,
        inplace: bool,
        stream: i64,
    );
    pub fn asort_asc_u8(
        x: *const c_void,
        dst: *mut c_void,
        nrows: i32,
        ncols: i32,
        inplace: bool,
        stream: i64,
    );
    pub fn asort_asc_u32(
        x: *const c_void,
        dst: *mut c_void,
        nrows: i32,
        ncols: i32,
        inplace: bool,
        stream: i64,
    );
    pub fn asort_asc_i64(
        x: *const c_void,
        dst: *mut c_void,
        nrows: i32,
        ncols: i32,
        inplace: bool,
        stream: i64,
    );
    pub fn asort_desc_f32(
        x: *const c_void,
        dst: *mut c_void,
        nrows: i32,
        ncols: i32,
        inplace: bool,
        stream: i64,
    );
    pub fn asort_desc_f16(
        x: *const c_void,
        dst: *mut c_void,
        nrows: i32,
        ncols: i32,
        inplace: bool,
        stream: i64,
    );
    pub fn asort_desc_bf16(
        x: *const c_void,
        dst: *mut c_void,
        nrows: i32,
        ncols: i32,
        inplace: bool,
        stream: i64,
    );
    pub fn asort_desc_f64(
        x: *const c_void,
        dst: *mut c_void,
        nrows: i32,
        ncols: i32,
        inplace: bool,
        stream: i64,
    );
    pub fn asort_desc_u8(
        x: *const c_void,
        dst: *mut c_void,
        nrows: i32,
        ncols: i32,
        inplace: bool,
        stream: i64,
    );
    pub fn asort_desc_u32(
        x: *const c_void,
        dst: *mut c_void,
        nrows: i32,
        ncols: i32,
        inplace: bool,
        stream: i64,
    );
    pub fn asort_desc_i64(
        x: *const c_void,
        dst: *mut c_void,
        nrows: i32,
        ncols: i32,
        inplace: bool,
        stream: i64,
    );

    pub fn copy_blocks_bf16(
        key_cache_ptrs: *mut c_void,
        value_cache_ptrs: *mut c_void,
        block_mapping: *const c_void,
        num_layers: i32,
        num_pairs: i32,
        numel_per_block: i32,
        stream: i64,
    );

    pub fn copy_blocks_f16(
        key_cache_ptrs: *mut c_void,
        value_cache_ptrs: *mut c_void,
        block_mapping: *const c_void,
        num_layers: i32,
        num_pairs: i32,
        numel_per_block: i32,
        stream: i64,
    );

    pub fn copy_blocks_f32(
        key_cache_ptrs: *mut c_void,
        value_cache_ptrs: *mut c_void,
        block_mapping: *const c_void,
        num_layers: i32,
        num_pairs: i32,
        numel_per_block: i32,
        stream: i64,
    );
}
