#include <torch/extension.h>
#include "utils.h"

torch::Tensor calculate_cdf(
    const torch::Tensor& mean,
    const torch::Tensor& scale,
    const torch::Tensor& Q,
    const int min_value,
    const int max_value
){
    CHECK_INPUT(mean);
    CHECK_INPUT(scale);
    CHECK_INPUT(Q);
    return calculate_cdf_cu(mean, scale, Q, min_value, max_value);
}


std::tuple<torch::Tensor, torch::Tensor> arithmetic_encode(
    const torch::Tensor& sym,
    const torch::Tensor& cdf,  // int 16
    const int chunk_size,
    const int N,
    const int Lp
){
    CHECK_INPUT(sym);
    CHECK_INPUT(cdf);
    return arithmetic_encode_cu(sym, cdf, chunk_size, N, Lp);
}


torch::Tensor arithmetic_decode(
    const torch::Tensor& cdf,
    const torch::Tensor& in_cache_all,
    const torch::Tensor& in_cnt_all,
    const int chunk_size,
    const int N,
    const int Lp
){
    CHECK_INPUT(cdf);
    CHECK_INPUT(in_cache_all);
    CHECK_INPUT(in_cnt_all);
    return arithmetic_decode_cu(cdf, in_cache_all, in_cnt_all, chunk_size, N, Lp);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("calculate_cdf", &calculate_cdf);
    m.def("arithmetic_encode", &arithmetic_encode);
    m.def("arithmetic_decode", &arithmetic_decode);
}