use candle::backend::BackendStorage;
#[cfg(feature = "cuda")]
use candle::CudaStorage;
use candle::{CpuStorage, DType, Layout, Result, Shape, Storage, Tensor};
use candle_core as candle;
use half::{bf16, f16};
#[cfg(feature = "cuda")]
use kernels::ffi::{
    awq_repack, gemm_half_q_half_alt, gptq_repack, marlin_4bit_bf16, marlin_4bit_f16,
    marlin_awq_4bit_bf16, marlin_awq_4bit_f16,
};

struct GPTQMatMul {
    qzeros: Option<Tensor>,
    g_idx: Option<Tensor>,
    workspace: Option<Tensor>,
    bits: i32,
    group_size: i32,
    is_awq: bool,
}

impl GPTQMatMul {
    #[cfg(feature = "cuda")]
    fn cuda_fwd_t<
        T: candle::cuda_backend::CudaDType + candle::cuda_backend::cudarc::driver::DeviceRepr,
    >(
        &self,
        x: &CudaStorage,
        x_l: &Layout,
        qweight: &CudaStorage,
        qweight_l: &Layout,
        scale: &CudaStorage,
        scale_l: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        use candle::cuda_backend::cudarc::driver::DevicePtr;
        use candle::cuda_backend::WrapErr;
        use std::ffi::c_void;
        let dev = qweight.device();
        let x_shape = x_l.dims();
        let weight_shape = qweight_l.dims();
        // let zero_shape = self.qzeros.shape().dims();
        // let scale_shape = scale_l.dims();

        let pack_factor: usize = 32 / self.bits as usize;
        let marlin_format = self.workspace.is_some();
        let size_k = weight_shape[0] * pack_factor * if marlin_format { 2 } else { 1 }; //marlin format
        let size_n = weight_shape[1] / if marlin_format { 2 } else { 1 }; //marlin format

        let mut out_shape: Vec<usize> = x_shape.to_vec();
        out_shape[x_shape.len() - 1] = size_n;
        let oshape: Shape = out_shape.into();

        // Get cuda slices for all tensors
        let input = x.as_cuda_slice::<T>()?;
        let qw = qweight.as_cuda_slice::<u32>()?;
        let qs = scale.as_cuda_slice::<T>()?;

        // Get cuda views for all tensors
        let input = input.slice(x_l.start_offset()..);
        let qw = qw.slice(qweight_l.start_offset()..);
        let qs = qs.slice(scale_l.start_offset()..);

        let elem_count = oshape.elem_count();
        let out = unsafe { dev.alloc::<T>(elem_count) }.w()?;

        let out_ptr = *out.device_ptr() as *mut c_void;
        let in_ptr = *input.device_ptr() as *const c_void;
        let qw_ptr = *qw.device_ptr() as *const c_void;
        let qs_ptr = *qs.device_ptr() as *const c_void;

        let qzeros_ptr = if self.qzeros.is_some() {
            let (qzeros, qzeros_l) = self.qzeros.as_ref().unwrap().storage_and_layout();
            let qzeros = match &*qzeros {
                Storage::Cuda(p) => p,
                _ => candle::bail!("qzeros must be a cuda tensor"),
            };
            let qzeros_ = qzeros.as_cuda_slice::<u32>()?;
            let qzeros_ = qzeros_.slice(qzeros_l.start_offset()..);
            *qzeros_.device_ptr() as *const c_void
        } else {
            std::ptr::null()
        };

        let g_idx_ptr = if self.g_idx.is_some() {
            let (g_idx, g_idx_l) = self.g_idx.as_ref().unwrap().storage_and_layout();
            let g_idx = match &*g_idx {
                Storage::Cuda(p) => p,
                _ => candle::bail!("g_idx must be a cuda tensor"),
            };
            let g_idx_ = g_idx.as_cuda_slice::<u32>()?;
            let g_idx_ = g_idx_.slice(g_idx_l.start_offset()..);
            *g_idx_.device_ptr() as *const c_void
        } else {
            std::ptr::null()
        };

        unsafe {
            if marlin_format {
                let workspace_ptr = if self.workspace.is_some() {
                    let (workspace, workspace_l) =
                        self.workspace.as_ref().unwrap().storage_and_layout();
                    let workspace = match &*workspace {
                        Storage::Cuda(p) => p,
                        _ => candle::bail!("workspace must be a cuda tensor"),
                    };
                    let workspace_ = workspace.as_cuda_slice::<u32>()?;
                    let workspace_ = workspace_.slice(workspace_l.start_offset()..);
                    *workspace_.device_ptr() as *const c_void
                } else {
                    candle::bail!("workspace is required for marlin matmul!")
                };

                if x.dtype() == DType::F16 {
                    if self.is_awq {
                        marlin_awq_4bit_f16(
                            in_ptr,
                            qw_ptr as *const i32,
                            qs_ptr,
                            qzeros_ptr,
                            g_idx_ptr,
                            out_ptr,
                            (x_shape[0] * x_shape[1]) as i32,
                            size_k as i32,
                            size_n as i32,
                            workspace_ptr,
                            self.group_size,
                            *dev.cu_stream() as i64,
                        );
                    } else {
                        marlin_4bit_f16(
                            in_ptr,
                            qw_ptr as *const i32,
                            qs_ptr,
                            qzeros_ptr,
                            g_idx_ptr,
                            out_ptr,
                            (x_shape[0] * x_shape[1]) as i32, //m
                            size_k as i32,                    //k
                            size_n as i32,                    //n
                            workspace_ptr,
                            self.group_size,
                            *dev.cu_stream() as i64,
                        );
                    }
                } else if x.dtype() == DType::BF16 {
                    if self.is_awq {
                        marlin_awq_4bit_bf16(
                            in_ptr,
                            qw_ptr as *const i32,
                            qs_ptr,
                            qzeros_ptr,
                            g_idx_ptr,
                            out_ptr,
                            (x_shape[0] * x_shape[1]) as i32,
                            size_k as i32,
                            size_n as i32,
                            workspace_ptr,
                            self.group_size,
                            *dev.cu_stream() as i64,
                        );
                    } else {
                        marlin_4bit_bf16(
                            in_ptr,
                            qw_ptr as *const i32,
                            qs_ptr,
                            qzeros_ptr,
                            g_idx_ptr,
                            out_ptr,
                            (x_shape[0] * x_shape[1]) as i32, //m
                            size_k as i32,                    //k
                            size_n as i32,                    //n
                            workspace_ptr,
                            self.group_size,
                            *dev.cu_stream() as i64,
                        );
                    }
                }
            } else {
                if x.dtype() == DType::F16 {
                    gemm_half_q_half_alt(
                        in_ptr,
                        qw_ptr as *const u32,
                        qzeros_ptr as *const u32,
                        qs_ptr,
                        g_idx_ptr as *const i32,
                        out_ptr,
                        (x_shape[0] * x_shape[1]) as i32,
                        size_n as i32,
                        size_k as i32,
                        self.bits,
                        *dev.cu_stream() as i64,
                    )
                } else {
                    candle::bail!("GPTQMatMul is only supported for f16 non-marlin matmul. Use '--dtype f16' parameter instead.");
                }
            }
        }

        let out = CudaStorage::wrap_cuda_slice(out, dev.clone());
        Ok((out, oshape))
    }
}

impl candle::CustomOp3 for GPTQMatMul {
    fn name(&self) -> &'static str {
        "GPTQMatMul"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle::bail!("no cpu support for GPTQMatMul")
    }
    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        x: &CudaStorage,
        x_l: &Layout,
        qweight: &CudaStorage,
        qweight_l: &Layout,
        scale: &CudaStorage,
        scale_l: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        match x.dtype() {
            DType::F16 => self.cuda_fwd_t::<f16>(x, x_l, qweight, qweight_l, scale, scale_l),
            DType::BF16 => self.cuda_fwd_t::<bf16>(x, x_l, qweight, qweight_l, scale, scale_l),
            dt => candle::bail!("GPTQMatMul is only supported for f16 and bf16 ({dt:?})"),
        }
    }
}

pub fn gptq_matmul(
    x: &Tensor,
    qweight: &Tensor,
    scale: &Tensor,
    qzeros: &Option<Tensor>,
    g_idx: &Option<Tensor>,
    workspace: &Option<Tensor>,
    bits: i32,
    group_size: i32,
    is_awq: bool,
) -> Result<Tensor> {
    let op = GPTQMatMul {
        qzeros: qzeros.to_owned(),
        g_idx: g_idx.to_owned(),
        workspace: workspace.to_owned(),
        bits,
        group_size,
        is_awq,
    };
    x.apply_op3(qweight, scale, op)
}

#[allow(dead_code)]
struct MarlinRepack {
    bits: i32,
    is_awq: bool, //awq or gptq
}

impl MarlinRepack {
    #[cfg(feature = "cuda")]
    fn cuda_fwd_t<
        T: candle::cuda_backend::CudaDType + candle::cuda_backend::cudarc::driver::DeviceRepr,
    >(
        &self,
        qweight: &CudaStorage,
        qweight_l: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        use candle::cuda_backend::cudarc::driver::DevicePtr;
        use candle::cuda_backend::WrapErr;
        let dev = qweight.device();
        let q_shape = qweight_l.dims();
        let mut out_shape: Vec<usize> = q_shape.to_vec();
        let pack_factor = (32 / self.bits) as usize;
        if self.is_awq {
            //in_dim 4096, out_dim 1024 (/pack_factor)
            //ws shape [4096, 128]
            //out_shape [256, 2048]
            out_shape[0] = q_shape[0] / pack_factor / 2;
            out_shape[1] = q_shape[1] * pack_factor * 2;
        } else {
            //in_dim 4096 (/pack_factor), out_dim 1024
            //ws shape [512, 1024]
            //out_shape [256, 2048]
            out_shape[0] = q_shape[0] / 2;
            out_shape[1] = q_shape[1] * 2;
        }

        let oshape: Shape = out_shape.into();

        // Get cuda slices for all tensors
        let q = qweight.as_cuda_slice::<u32>()?;

        // Get cuda views for all tensors
        let q = q.slice(qweight_l.start_offset()..);

        let elem_count = oshape.elem_count();
        let out = unsafe { dev.alloc::<u32>(elem_count) }.w()?;

        let out_ptr = *out.device_ptr() as *const core::ffi::c_void;
        let q_ptr = *q.device_ptr() as *const core::ffi::c_void;

        unsafe {
            if self.is_awq {
                awq_repack(
                    q_ptr,
                    out_ptr,
                    q_shape[0] as i32,
                    q_shape[1] as i32,
                    self.bits,
                    *dev.cu_stream() as i64,
                )
            } else {
                gptq_repack(
                    q_ptr,
                    out_ptr,
                    q_shape[0] as i32,
                    q_shape[1] as i32,
                    *dev.cu_stream() as i64,
                )
            }
        }

        let out = CudaStorage::wrap_cuda_slice(out, dev.clone());
        Ok((out, oshape))
    }
}

impl candle::CustomOp1 for MarlinRepack {
    fn name(&self) -> &'static str {
        "MarlinRepack"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle::bail!("no cpu support for MarlinRepack")
    }
    #[cfg(feature = "cuda")]
    fn cuda_fwd(&self, qweight: &CudaStorage, qweight_l: &Layout) -> Result<(CudaStorage, Shape)> {
        match qweight.dtype() {
            DType::U32 => self.cuda_fwd_t::<u32>(qweight, qweight_l),
            dt => candle::bail!("MarlinRepack is only supported for i32/u32 weight ({dt:?})"),
        }
    }
}

pub fn marlin_weight_repack(qweight: &Tensor, bits: i32, is_awq: bool) -> Result<Tensor> {
    let op = MarlinRepack { bits, is_awq };
    qweight.apply_op1(op)
}
