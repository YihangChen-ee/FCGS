#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


torch::Tensor calculate_cdf_cu(
    const torch::Tensor& mean,
    const torch::Tensor& scale,
    const torch::Tensor& Q,
    const int min_value,
    const int max_value
);

std::tuple<torch::Tensor, torch::Tensor> arithmetic_encode_cu(
    const torch::Tensor& sym,
    const torch::Tensor& cdf,
    const int chunk_size,
    const int N,
    const int Lp
);

torch::Tensor arithmetic_decode_cu(
    const torch::Tensor& cdf,
    const torch::Tensor& in_cache_all,
    const torch::Tensor& in_cnt_all,
    const int chunk_size,
    const int N,
    const int Lp
);