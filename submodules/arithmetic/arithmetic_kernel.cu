#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>

__device__ float gaussian_cdf(float x, float mean, float scale) {
    return 0.5 * erfc(-(x - mean) / (scale * sqrtf(2.0)));
}

__global__ void calculate_cdf_kernel(
    const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> mean,
    const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> scale,
    const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> Q,
    const int min_value,
    const int max_value,
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> lower
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= mean.size(0)) return;

    float scale_clamped = max(scale[idx], 1e-9);

    for (int i = 0; i < max_value - min_value + 2; ++i) {
        float sample_value = (min_value + i - 0.5) * Q[idx];
        lower[idx][i] = gaussian_cdf(sample_value, mean[idx], scale_clamped);
    }
}

torch::Tensor calculate_cdf_cu(
    const torch::Tensor& mean,
    const torch::Tensor& scale,
    const torch::Tensor& Q,
    const int min_value,
    const int max_value
) {
    int N = mean.size(0);
    torch::Tensor lower = torch::zeros({N, max_value - min_value + 2}, mean.options());

    int threads_per_block = 128;
    int blocks = (N + threads_per_block - 1) / threads_per_block;

    calculate_cdf_kernel<<<blocks, threads_per_block>>>(
        mean.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
        scale.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
        Q.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
        min_value,
        max_value,
        lower.packed_accessor32<float, 2, torch::RestrictPtrTraits>()
    );

    cudaDeviceSynchronize();
    return lower;
}


/** Class to save output bit by bit to a byte string */
class OutCacheString {
private:
    uint8_t cache = 0;
    uint8_t count = 0;

public:
    __device__ void append(const int bit, uint8_t* out, int32_t& length_cnt) {
        cache <<= 1;
        cache |= bit;
        count += 1;
        if (count == 8) {
            out[length_cnt] = static_cast<uint8_t>(cache);
            count = 0;
            length_cnt += 1;
        }
    }

    __device__ void flush(uint8_t* out, int32_t& length_cnt) {
        if (count > 0) {
            for (int i = count; i < 8; ++i) {
                append(0, out, length_cnt);
            }
            assert(count == 0);
        }
    }

    __device__ void append_bit_and_pending(const int bit, uint64_t& pending_bits, uint8_t* out, int32_t& length_cnt) {
        append(bit, out, length_cnt);
        while (pending_bits > 0) {
            append(!bit, out, length_cnt);
            pending_bits -= 1;
        }
    }
};


__global__ void encode_arithmetic_kernel(
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> cdf,
    const torch::PackedTensorAccessor32<int16_t, 1, torch::RestrictPtrTraits> sym,
    torch::PackedTensorAccessor32<uint8_t, 2, torch::RestrictPtrTraits> out_cache_all,
    torch::PackedTensorAccessor32<int32_t, 1, torch::RestrictPtrTraits> out_cnt_all,
    const int chunk_size,
    const int chunk_num,
    const int Lp,
    const int max_symbol,
    const int precision
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= chunk_num) return;

    const int dim0_offset = idx * chunk_size;
    uint8_t* out = out_cache_all[idx].data();
    int32_t& length_cnt = out_cnt_all[idx];

    OutCacheString out_cache;

    uint32_t low = 0;
    uint32_t high = 0xFFFFFFFFU;
    uint64_t pending_bits = 0;
    float new_max_value = (1 << precision) - (Lp - 1);
    const int current_chunk_size = min(chunk_size, sym.size(0) - dim0_offset);

    for (int i = 0; i < current_chunk_size; ++i) {
        const int16_t sym_i = sym[dim0_offset + i];
        const uint64_t span = static_cast<uint64_t>(high) - static_cast<uint64_t>(low) + 1;

        const uint32_t c_low = __float2int_rn(cdf[dim0_offset + i][sym_i] * new_max_value) + sym_i;
        const uint32_t c_high = sym_i == max_symbol ? 0x10000U : __float2int_rn(cdf[dim0_offset + i][sym_i + 1] * new_max_value) + sym_i + 1;

        high = (low - 1) + ((span * static_cast<uint64_t>(c_high)) >> precision);
        low =  (low)     + ((span * static_cast<uint64_t>(c_low))  >> precision);

        while (true) {
            if (high < 0x80000000U) {
                out_cache.append_bit_and_pending(0, pending_bits, out, length_cnt);
                low <<= 1;
                high <<= 1;
                high |= 1;
            } else if (low >= 0x80000000U) {
                out_cache.append_bit_and_pending(1, pending_bits, out, length_cnt);
                low <<= 1;
                high <<= 1;
                high |= 1;
            } else if (low >= 0x40000000U && high < 0xC0000000U) {
                pending_bits++;
                low <<= 1;
                low &= 0x7FFFFFFF;
                high <<= 1;
                high |= 0x80000001;
            } else {
                break;
            }
        }
    }

    pending_bits += 1;
    if (pending_bits) {
        if (low < 0x40000000U) {
            out_cache.append_bit_and_pending(0, pending_bits, out, length_cnt);
        } else {
            out_cache.append_bit_and_pending(1, pending_bits, out, length_cnt);
        }
    }

    out_cache.flush(out, length_cnt);
}


__global__ void merge_chunks_kernel(
    const torch::PackedTensorAccessor32<uint8_t, 2, torch::RestrictPtrTraits> out_cache_all,
    const torch::PackedTensorAccessor32<int32_t, 1, torch::RestrictPtrTraits> out_cnt_all,
    torch::PackedTensorAccessor32<uint8_t, 1, torch::RestrictPtrTraits> out_final,
    const int chunk_num
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= chunk_num) return;

    int start_idx = 0;
    for (int i = 0; i < idx; ++i) {
        start_idx += out_cnt_all[i];
    }

    for (int i = 0; i < out_cnt_all[idx]; ++i) {
        out_final[start_idx + i] = out_cache_all[idx][i];
    }
}


std::tuple<torch::Tensor, torch::Tensor> arithmetic_encode_cu(
    const torch::Tensor& sym,
    const torch::Tensor& cdf,
    const int chunk_size,
    const int N,
    const int Lp
) {
    const int chunk_num = (N + chunk_size - 1) / chunk_size;
    const int max_symbol = Lp - 2;

    torch::Tensor out_cache_all = torch::zeros({chunk_num, chunk_size*4}, torch::TensorOptions().dtype(torch::kUInt8).device(sym.device()));
    torch::Tensor out_cnt_all = torch::zeros({chunk_num}, torch::TensorOptions().dtype(torch::kInt32).device(sym.device()));
    TORCH_CHECK(sym.dim() == 1, "Expected sym to have 1 dimension, but got ", sym.dim());
    TORCH_CHECK(cdf.dim() == 2, "Expected cdf to have 2 dimensions, but got ", cdf.dim());

    int threads_per_block = 1;
    int blocks = (chunk_num + threads_per_block - 1) / threads_per_block;

    encode_arithmetic_kernel<<<blocks, threads_per_block>>>(
        cdf.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        sym.packed_accessor32<int16_t, 1, torch::RestrictPtrTraits>(),
        out_cache_all.packed_accessor32<uint8_t, 2, torch::RestrictPtrTraits>(),
        out_cnt_all.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(),
        chunk_size,
        chunk_num,
        Lp,
        max_symbol,
        16
    );

    cudaDeviceSynchronize();

    int total_length = out_cnt_all.sum().item<int>();

    torch::Tensor out_final = torch::zeros({total_length}, out_cache_all.options());

    merge_chunks_kernel<<<blocks, threads_per_block>>>(
        out_cache_all.packed_accessor32<uint8_t, 2, torch::RestrictPtrTraits>(),
        out_cnt_all.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(),
        out_final.packed_accessor32<uint8_t, 1, torch::RestrictPtrTraits>(),
        chunk_num
    );

    cudaDeviceSynchronize();

    return std::make_tuple(out_final, out_cnt_all);
}

/************************* decoding *************************/


class InCacheString {
private:
    uint8_t cache=0;
    uint8_t cached_bits=0;
    int32_t in_ptr=0;

public:
    __device__ void get(uint32_t& value, uint8_t* in_, const int32_t in_length) {
        if (cached_bits == 0) {
            if (in_ptr == in_length) {
                value <<= 1;
                return;
            }
            cache = in_[in_ptr];
            in_ptr++;
            cached_bits = 8;
        }
        value <<= 1;
        value |= (cache >> (cached_bits - 1)) & 1;
        cached_bits--;
    }
    __device__ void initialize(uint32_t& value, uint8_t* in_, const int32_t in_length) {
        for (int i = 0; i < 32; ++i) {
            get(value, in_, in_length);
        }
    }
};

__device__ int16_t binsearch(
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> cdf,
    const uint16_t target,
    int max_sym,
    const int row_index,
    const float new_max_value
) {
    int left = 0;
    int right = max_sym + 1;

    while (left + 1 < right) {
        const uint16_t m = (left + right) / 2;
        const uint16_t v = __float2int_rn(cdf[row_index][m] * new_max_value) + m;
        if (v < target) {
            left = m;
        } else if (v > target) {
            right = m;
        } else {
            return m;
        }
    }
    return left;
}


__global__ void decode_arithmetic_kernel(
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> cdf,
    const torch::PackedTensorAccessor32<uint8_t, 1, torch::RestrictPtrTraits> in_cache_all,
    const torch::PackedTensorAccessor32<int32_t, 1, torch::RestrictPtrTraits> in_cnt_cum_all,
    const torch::PackedTensorAccessor32<int32_t, 1, torch::RestrictPtrTraits> in_cnt_all,
    torch::PackedTensorAccessor32<int16_t, 1, torch::RestrictPtrTraits> decoded_sym,
    const int chunk_size,
    const int chunk_num,
    const int Lp,
    const int max_symbol,
    const int precision
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= chunk_num) return;

    const int dim0_offset = idx * chunk_size;
    uint8_t* in_ = in_cache_all.data() + in_cnt_cum_all[idx];
    int32_t length_cnt = in_cnt_all[idx];

    InCacheString in_cache;
    uint32_t low = 0;
    uint32_t high = 0xFFFFFFFFU;
    uint32_t value = 0;
    const float new_max_value = (1 << precision) - (Lp - 1);
    const uint64_t c_count = (1 << precision);
    in_cache.initialize(value, in_, length_cnt);
    const int current_chunk_size = min(chunk_size, decoded_sym.size(0) - dim0_offset);

    for (int i = 0; i < current_chunk_size; ++i) {
        const uint64_t span = static_cast<uint64_t>(high) - static_cast<uint64_t>(low) + 1;
        const uint16_t count = ((static_cast<uint64_t>(value) - static_cast<uint64_t>(low) + 1) * c_count - 1) / span;

        int16_t sym_i = binsearch(
            cdf,
            count,
            max_symbol,
            dim0_offset + i,
            new_max_value
        );

        decoded_sym[dim0_offset + i] = sym_i;

        const uint32_t c_low = __float2int_rn(cdf[dim0_offset + i][sym_i] * new_max_value) + sym_i;
        const uint32_t c_high = sym_i == max_symbol ? 0x10000U : __float2int_rn(cdf[dim0_offset + i][sym_i + 1] * new_max_value) + sym_i + 1;

        high = (low - 1) + ((span * static_cast<uint64_t>(c_high)) >> precision);
        low = (low) + ((span * static_cast<uint64_t>(c_low)) >> precision);

        while (true) {
            if (low >= 0x80000000U || high < 0x80000000U) {
                low <<= 1;
                high <<= 1;
                high |= 1;
                in_cache.get(value, in_, length_cnt);
            } else if (low >= 0x40000000U && high < 0xC0000000U) {
                low <<= 1;
                low &= 0x7FFFFFFFU;
                high <<= 1;
                high |= 0x80000001U;
                value -= 0x40000000U;
                in_cache.get(value, in_, length_cnt);
            } else {
                break;
            }
        }
    }
}

torch::Tensor compute_cumsum(const torch::Tensor& in_cnt_all) {
    TORCH_CHECK(in_cnt_all.dim() == 1, "Expected in_cnt_all to be 1-dimensional");

    torch::Tensor in_cnt_cum_all = torch::cat({torch::zeros({1}, in_cnt_all.options()), torch::cumsum(in_cnt_all, 0)});
    return in_cnt_cum_all.to(torch::kInt32);
}


torch::Tensor arithmetic_decode_cu(
    const torch::Tensor& cdf,
    const torch::Tensor& in_cache_all,
    const torch::Tensor& in_cnt_all,
    const int chunk_size,
    const int N,
    const int Lp
) {
    const int chunk_num = (N + chunk_size - 1) / chunk_size;
    const int max_symbol = Lp - 2;

    torch::Tensor in_cnt_cum_all = compute_cumsum(in_cnt_all);

    torch::Tensor decoded_sym = torch::zeros({N}, torch::TensorOptions().dtype(torch::kInt16).device(cdf.device()));
    TORCH_CHECK(cdf.dim() == 2, "Expected cdf to have 2 dimensions, but got ", cdf.dim());
    TORCH_CHECK(in_cache_all.dim() == 1, "Expected in_cache_all to have 1 dimension, but got ", in_cache_all.dim());
    TORCH_CHECK(in_cnt_cum_all.dim() == 1, "Expected in_cnt_cum_all to have 1 dimension, but got ", in_cnt_cum_all.dim());
    TORCH_CHECK(in_cnt_all.dim() == 1, "Expected in_cnt_all to have 1 dimension, but got ", in_cnt_all.dim());

    int threads_per_block = 1;
    int blocks = (chunk_num + threads_per_block - 1) / threads_per_block;

    decode_arithmetic_kernel<<<blocks, threads_per_block>>>(
        cdf.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        in_cache_all.packed_accessor32<uint8_t, 1, torch::RestrictPtrTraits>(),
        in_cnt_cum_all.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(),
        in_cnt_all.packed_accessor32<int32_t, 1, torch::RestrictPtrTraits>(),
        decoded_sym.packed_accessor32<int16_t, 1, torch::RestrictPtrTraits>(),
        chunk_size,
        chunk_num,
        Lp,
        max_symbol,
        16
    );

    cudaDeviceSynchronize();
    return decoded_sym;
}
