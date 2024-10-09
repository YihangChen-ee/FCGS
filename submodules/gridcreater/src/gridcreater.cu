#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <ATen/cuda/CUDAContext.h>
#include <torch/torch.h>

#include <algorithm>
#include <stdexcept>

#include <stdint.h>
#include <cstdio>


#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be a contiguous tensor")
#define CHECK_IS_INT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Int, #x " must be an int tensor")
#define CHECK_IS_FLOATING(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float || x.scalar_type() == at::ScalarType::Half || x.scalar_type() == at::ScalarType::Double, #x " must be a floating tensor")


// just for compatability of half precision in AT_DISPATCH_FLOATING_TYPES_AND_HALF... program will never reach here!
 __device__ inline at::Half atomicAdd(at::Half *address, at::Half val) {
  // requires CUDA >= 10 and ARCH >= 70
  // this is very slow compared to float or __half2, never use it.
  //return atomicAdd(reinterpret_cast<__half*>(address), val);
}


template <typename T>
__host__ __device__ inline T div_round_up(T val, T divisor) {
    return (val + divisor - 1) / divisor;
}

template <typename T>
__device__ inline T smoothstep(T val) {
	return val*val*(3.0f - 2.0f * val);
}

template <typename T>
__device__ inline T smoothstep_derivative(T val) {
	return 6*val*(1.0f - val);
}


template <uint32_t num_dim>
__device__ uint32_t fast_hash(const uint32_t pos_grid[num_dim]) {

    // coherent type of hashing
    constexpr uint32_t primes[7] = { 1u, 2654435761u, 805459861u, 3674653429u, 2097192037u, 1434869437u, 2165219737u };

    uint32_t result = 0;
    #pragma unroll
    for (uint32_t i = 0; i < num_dim; ++i) {
        result ^= pos_grid[i] * primes[i];
    }

    return result;
}


template <uint32_t num_dim, uint32_t n_features>
// gridtype, 0, hashmap_size, resolution, pos_grid_local
__device__ uint32_t get_grid_index(const uint32_t gridtype, const uint32_t ch, const uint32_t hashmap_size, const uint32_t resolution, const uint32_t pos_grid[num_dim]) {
    uint32_t stride = 1;
    uint32_t index = 0;

    #pragma unroll

    // if level is small, hashtable is long enough, then no hash trick is needed
    // final index = pos_grid[0] + pos_grid[1] * resolution + pos_grid[2] * resolution ^ 2
    // This is to get the ordered index (eg: idx = W*H+w for 2D case)
    for (uint32_t d = 0; d < num_dim && stride <= hashmap_size; d++) {
        // pos_grid[0] -> pos_grid[0] + pos_grid[1] * resolution -> pos_grid[0] + pos_grid[1] * resolution + pos_grid[2] * resolution ^ 2
        index += pos_grid[d] * stride;
        // resolution -> resolution^2 -> resolution^3
        stride *= resolution;
    }

    // NOTE: for NeRF, the hash is in fact not necessary. Check https://github.com/NVlabs/instant-ngp/issues/97.
    // gridtype: 0 == hash, 1 == tiled
    if (gridtype == 0 && stride > hashmap_size) {
        index = fast_hash<num_dim>(pos_grid);
    }

    // (index % hashmap_size) to avoid overflow
    // notice: here n_features is multipled
    return (index % hashmap_size);
}


// N: N_rays
// n_levels: level
// S: log2(per_level_scale)
// H: base_resolution
////// One CPU kernel calls one GPU grid, one GPU grid contains several blocks, one block contains several threads
template <typename scalar_t, uint32_t num_dim, uint32_t n_features>  // <scalar_t, num_dim, 2 // num_dim: coords input_dim = 3
__global__ void kernel_grid(
    const float * __restrict__ input_norm_xyz,  //  has been mapped to [0, 1]  [N, num_dim]
    const scalar_t * __restrict__ input_feature,  // [N, n_features]
    scalar_t * __restrict__ outputs,   // [offsets_list[-1], n_features+1]
    scalar_t * __restrict__ weights,   // [offsets_list[-1], n_features+1]
    const int * __restrict__ offsets_list,
    const int * __restrict__ resolutions_list,
    const uint32_t N, const uint32_t n_levels
    ) {
    // grid > block > thread
    // blockIdx.x is idx of the block along x axis, blockDim.x is th block width, threadIdx.x is idx along the width of thread
    // get the place of corresponding parallel computing point
    // get b: the index of [0, N_rays)
    const uint32_t b = blockIdx.x * blockDim.x + threadIdx.x;  // parallel along N_rays axis
    if (b >= N) return;  // deprecate not needed threads
    const uint32_t level = blockIdx.y;

    input_norm_xyz += b * num_dim;
    input_feature += b * n_features;
    outputs += (uint32_t)offsets_list[level] * n_features;
    weights += (uint32_t)offsets_list[level];

    // check input range (should be in [0, 1])
    bool flag_oob = false;
    #pragma unroll
    for (uint32_t d = 0; d < num_dim; d++) {  // traverse each input_dim
        if (input_norm_xyz[d] < 0 || input_norm_xyz[d] > 1) {
            flag_oob = true;
            return;
        }
    }

    // calculate coordinate (always use float for precision!)
    float pos[num_dim];  // fractional part
    uint32_t pos_grid[num_dim];
    const uint32_t hashmap_size = offsets_list[level + 1] - offsets_list[level];
    const uint32_t resolution = (uint32_t)resolutions_list[level];

    #pragma unroll
    for (uint32_t d = 0; d < num_dim; d++) {  // traverse each input_dim
        pos[d] = input_norm_xyz[d] * float(resolution - 2) + 0.5; // resolution = 6: 0->0.5, 1->4.5
        pos_grid[d] = (uint32_t)floorf(pos[d]);
        pos[d] -= (float)pos_grid[d];
    }

    uint32_t index_list[1 << num_dim] = {0};

    #pragma unroll
    // idx = {0, 1, 2, 3, 4, 5, 6, 7}
    // here loop_num==8 is because there are 8 vertextes for interp 8=2**num_dim
    for (uint32_t idx = 0; idx < (1 << num_dim); idx++) { // why parallel is not applied for this loop?
        float w = 1;  // define weight for triblinear interp for different vertexes
        uint32_t pos_grid_local[num_dim];

        #pragma unroll
        for (uint32_t d = 0; d < num_dim; d++) {
            if ((idx & (1 << d)) == 0) {  // (1 << d) = {1, 2, 4}
                w *= 1 - pos[d];
                pos_grid_local[d] = pos_grid[d];
            } else {
                w *= pos[d];
                pos_grid_local[d] = min(pos_grid[d] + 1, resolution - 1);
            }
        }
        w += 1e-9;  // avoid cases input coordinates are right at voxel lines

        uint32_t index = get_grid_index<num_dim, n_features>(0, 0, hashmap_size, resolution, pos_grid_local);
        uint32_t index_m_nfeatures = index * n_features;

        #pragma unroll
        for (uint32_t ch = 0; ch < n_features; ch++) {
            atomicAdd(&outputs[index_m_nfeatures+ch], w*input_feature[ch]);
        }
        atomicAdd(&weights[index], w);
    }
}


// input_norm_xyz, grad, weights, grad_feature, offsets_list, resolutions_list, N, n_levels
// grad, inputs, embeddings, offsets_list, grad_embeddings, N, num_dim, n_features, n_levels, max_level, S, H, dy_dx, grad_inputs, gridtype, align_corners, interpolation
// template <typename scalar_t, uint32_t num_dim, uint32_t n_features, uint32_t n_features_per_thread>  // n_features_per_thread is n_features_per_thread
template <typename scalar_t, uint32_t num_dim, uint32_t n_features>  // n_features_per_thread is n_features_per_thread
__global__ void kernel_grid_backward(
    const float * __restrict__ input_norm_xyz,  // [N, 3]
    const scalar_t * __restrict__ grad,  // [offsets_list[-1], n_features]
    const scalar_t * __restrict__ weights,  // [offsets_list[-1], 1]
    scalar_t * __restrict__ grad_feature,   // [N, n_features]
    const int * __restrict__ offsets_list,
    const int * __restrict__ resolutions_list,
    const uint32_t N, const uint32_t n_levels
    ) {

    const uint32_t b = blockIdx.x * blockDim.x + threadIdx.x;  // parallel along N_rays axis
    if (b >= N) return;
    const uint32_t level = blockIdx.y;

    input_norm_xyz += b * num_dim;
    grad_feature += b * n_features;
    grad += (uint32_t)offsets_list[level] * n_features;
    weights += (uint32_t)offsets_list[level];

    // check input range (should be in [0, 1])
    #pragma unroll
    for (uint32_t d = 0; d < num_dim; d++) {
        if (input_norm_xyz[d] < 0 || input_norm_xyz[d] > 1) {
            return; // grad is init as 0, so we simply return.
        }
    }

        // calculate coordinate (always use float for precision!)
    float pos[num_dim];  // fractional part
    uint32_t pos_grid[num_dim];
    const uint32_t hashmap_size = offsets_list[level + 1] - offsets_list[level];
    const uint32_t resolution = (uint32_t)resolutions_list[level];
    #pragma unroll
    for (uint32_t d = 0; d < num_dim; d++) {  // traverse each input_dim
        pos[d] = input_norm_xyz[d] * float(resolution - 2) + 0.5; // resolution = 6: 0->0.5, 1->4.5
        pos_grid[d] = (uint32_t)floorf(pos[d]);
        pos[d] -= (float)pos_grid[d];
    }

    #pragma unroll
    // idx = {0, 1, 2, 3, 4, 5, 6, 7}
    // here loop_num==8 is because there are 8 vertextes for interp 8=2**num_dim
    for (uint32_t idx = 0; idx < (1 << num_dim); idx++) { // why parallel is not applied for this loop?
        float w = 1;  // define weight for triblinear interp for different vertexes
        uint32_t pos_grid_local[num_dim];

        #pragma unroll
        for (uint32_t d = 0; d < num_dim; d++) {
            if ((idx & (1 << d)) == 0) {  // (1 << d) = {1, 2, 4}
                w *= 1 - pos[d];
                pos_grid_local[d] = pos_grid[d];
            } else {
                w *= pos[d];
                pos_grid_local[d] = min(pos_grid[d] + 1, resolution - 1);
            }
        }
        w += 1e-9;  // avoid cases input coordinates are right at voxel lines

        uint32_t index = get_grid_index<num_dim, n_features>(0, 0, hashmap_size, resolution, pos_grid_local);
        uint32_t index_m_nfeatures = index * n_features;

        #pragma unroll
        for (uint32_t ch = 0; ch < n_features; ch++) {
            atomicAdd(&grad_feature[ch], w/weights[index]*grad[index_m_nfeatures+ch]);
        }
    }

}



// -----------------------------


template <typename scalar_t, uint32_t num_dim>
void kernel_grid_wrapper(
    const float *input_norm_xyz,
    const scalar_t *input_feature,
    scalar_t *outputs,
    scalar_t *weights,
    const int *offsets_list,
    const int *resolutions_list,
    const uint32_t N, const uint32_t n_features, const uint32_t n_levels
    ) {
    // blocks and threads are defined here
    static constexpr uint32_t N_THREAD = 512;  // in fact (521, 1, 1)
    // div_round_up is (N + N_THREAD - 1) / N_THREAD
    const dim3 blocks_hashgrid = { div_round_up(N, N_THREAD), n_levels, 1 };
    switch (n_features) {
        // at the input of function, there might have "packed_accessor". it is used to transform data types.
        case 1: kernel_grid<scalar_t, num_dim, 1><<<blocks_hashgrid, N_THREAD>>>(input_norm_xyz, input_feature, outputs, weights, offsets_list, resolutions_list, N, n_levels); break;
        case 2: kernel_grid<scalar_t, num_dim, 2><<<blocks_hashgrid, N_THREAD>>>(input_norm_xyz, input_feature, outputs, weights, offsets_list, resolutions_list, N, n_levels); break;
        case 4: kernel_grid<scalar_t, num_dim, 4><<<blocks_hashgrid, N_THREAD>>>(input_norm_xyz, input_feature, outputs, weights, offsets_list, resolutions_list, N, n_levels); break;
        case 8: kernel_grid<scalar_t, num_dim, 8><<<blocks_hashgrid, N_THREAD>>>(input_norm_xyz, input_feature, outputs, weights, offsets_list, resolutions_list, N, n_levels); break;
        case 16: kernel_grid<scalar_t, num_dim, 16><<<blocks_hashgrid, N_THREAD>>>(input_norm_xyz, input_feature, outputs, weights, offsets_list, resolutions_list, N, n_levels); break;
        case 32: kernel_grid<scalar_t, num_dim, 32><<<blocks_hashgrid, N_THREAD>>>(input_norm_xyz, input_feature, outputs, weights, offsets_list, resolutions_list, N, n_levels); break;
        case 48: kernel_grid<scalar_t, num_dim, 48><<<blocks_hashgrid, N_THREAD>>>(input_norm_xyz, input_feature, outputs, weights, offsets_list, resolutions_list, N, n_levels); break;
        case 64: kernel_grid<scalar_t, num_dim, 64><<<blocks_hashgrid, N_THREAD>>>(input_norm_xyz, input_feature, outputs, weights, offsets_list, resolutions_list, N, n_levels); break;
        case 128: kernel_grid<scalar_t, num_dim, 128><<<blocks_hashgrid, N_THREAD>>>(input_norm_xyz, input_feature, outputs, weights, offsets_list, resolutions_list, N, n_levels); break;
        case 256: kernel_grid<scalar_t, num_dim, 256><<<blocks_hashgrid, N_THREAD>>>(input_norm_xyz, input_feature, outputs, weights, offsets_list, resolutions_list, N, n_levels); break;
        default: throw std::runtime_error{"GridEncoding: n_features must be 1, 2, 4, 8, 16 or 32."};
    }
}

// inputs: [N, num_dim], float, in [0, 1]
// embeddings: [sO, n_features], float
// offsets_list: [n_levels + 1], uint32_t
// outputs: [n_levels, N, n_features], float (n_levels first, so only one level of hashmap needs to fit into cache at a time.)
// H: base resolution
// dy_dx: [N, n_levels * num_dim * n_features]
template <typename scalar_t>
void grid_creater_forward_cuda(
    const float *input_norm_xyz,
    const scalar_t * input_feature,
    scalar_t *outputs,
    scalar_t *weights,
    const int *offsets_list,
    const int *resolutions_list,
    const uint32_t N, const uint32_t num_dim, const uint32_t n_features, const uint32_t n_levels
    ) {
    switch (num_dim) {
        case 1: kernel_grid_wrapper<scalar_t, 1>(input_norm_xyz, input_feature, outputs, weights, offsets_list, resolutions_list, N, n_features, n_levels); break;
        case 2: kernel_grid_wrapper<scalar_t, 2>(input_norm_xyz, input_feature, outputs, weights, offsets_list, resolutions_list, N, n_features, n_levels); break;
        case 3: kernel_grid_wrapper<scalar_t, 3>(input_norm_xyz, input_feature, outputs, weights, offsets_list, resolutions_list, N, n_features, n_levels); break;
        default: throw std::runtime_error{"GridEncoding: num_dim must be 1, 2, 3."};
    }
}

template <typename scalar_t, uint32_t num_dim>
void kernel_grid_backward_wrapper(
    const float *input_norm_xyz,
    const scalar_t *grad,
    const scalar_t *weights,
    scalar_t *grad_feature,
    const int *offsets_list,
    const int *resolutions_list,
    const uint32_t N, const uint32_t n_features, const uint32_t n_levels
    ) {
    // static constexpr uint32_t N_THREAD = 512;
    // const uint32_t n_features_per_thread = std::min(2u, n_features); // n_features_per_thread
    // const dim3 blocks_hashgrid = { div_round_up(N * n_features / n_features_per_thread, N_THREAD), n_levels, 1 };
    static constexpr uint32_t N_THREAD = 512;  // in fact (521, 1, 1)
    // div_round_up is (N + N_THREAD - 1) / N_THREAD
    const dim3 blocks_hashgrid = { div_round_up(N, N_THREAD), n_levels, 1 };

    switch (n_features) {
        case 1: kernel_grid_backward<scalar_t, num_dim, 1><<<blocks_hashgrid, N_THREAD>>>(input_norm_xyz, grad, weights, grad_feature, offsets_list, resolutions_list, N, n_levels); break;
        case 2: kernel_grid_backward<scalar_t, num_dim, 2><<<blocks_hashgrid, N_THREAD>>>(input_norm_xyz, grad, weights, grad_feature, offsets_list, resolutions_list, N, n_levels); break;
        case 4: kernel_grid_backward<scalar_t, num_dim, 4><<<blocks_hashgrid, N_THREAD>>>(input_norm_xyz, grad, weights, grad_feature, offsets_list, resolutions_list, N, n_levels); break;
        case 8: kernel_grid_backward<scalar_t, num_dim, 8><<<blocks_hashgrid, N_THREAD>>>(input_norm_xyz, grad, weights, grad_feature, offsets_list, resolutions_list, N, n_levels); break;
        case 16: kernel_grid_backward<scalar_t, num_dim, 16><<<blocks_hashgrid, N_THREAD>>>(input_norm_xyz, grad, weights, grad_feature, offsets_list, resolutions_list, N, n_levels); break;
        case 32: kernel_grid_backward<scalar_t, num_dim, 32><<<blocks_hashgrid, N_THREAD>>>(input_norm_xyz, grad, weights, grad_feature, offsets_list, resolutions_list, N, n_levels); break;
        case 48: kernel_grid_backward<scalar_t, num_dim, 48><<<blocks_hashgrid, N_THREAD>>>(input_norm_xyz, grad, weights, grad_feature, offsets_list, resolutions_list, N, n_levels); break;
        case 64: kernel_grid_backward<scalar_t, num_dim, 64><<<blocks_hashgrid, N_THREAD>>>(input_norm_xyz, grad, weights, grad_feature, offsets_list, resolutions_list, N, n_levels); break;
        case 128: kernel_grid_backward<scalar_t, num_dim, 128><<<blocks_hashgrid, N_THREAD>>>(input_norm_xyz, grad, weights, grad_feature, offsets_list, resolutions_list, N, n_levels); break;
        case 256: kernel_grid_backward<scalar_t, num_dim, 256><<<blocks_hashgrid, N_THREAD>>>(input_norm_xyz, grad, weights, grad_feature, offsets_list, resolutions_list, N, n_levels); break;
        default: throw std::runtime_error{"GridEncoding: n_features must be 1, 2, 4, 8, 16 or 32."};
    }
}


// grad: [n_levels, N, n_features], float
// inputs: [N, num_dim], float, in [0, 1]
// embeddings: [sO, n_features], float
// offsets_list: [n_levels + 1], uint32_t
// grad_embeddings: [sO, n_features]
// H: base resolution

template <typename scalar_t>
void grid_creater_backward_cuda(
    const float *input_norm_xyz,
    const scalar_t *grad,
    const scalar_t *weights,
    scalar_t *grad_feature,
    const int *offsets_list,
    const int *resolutions_list,
    const uint32_t N, const uint32_t num_dim, const uint32_t n_features, const uint32_t n_levels
    ) {
    switch (num_dim) {
        case 1: kernel_grid_backward_wrapper<scalar_t, 1>(input_norm_xyz, grad, weights, grad_feature, offsets_list, resolutions_list, N, n_features, n_levels); break;
        case 2: kernel_grid_backward_wrapper<scalar_t, 2>(input_norm_xyz, grad, weights, grad_feature, offsets_list, resolutions_list, N, n_features, n_levels); break;
        case 3: kernel_grid_backward_wrapper<scalar_t, 3>(input_norm_xyz, grad, weights, grad_feature, offsets_list, resolutions_list, N, n_features, n_levels); break;
        default: throw std::runtime_error{"GridEncoding: num_dim must be 1, 2, 3."};
    }
}



void grid_creater_forward(
    const at::Tensor input_norm_xyz,
    const at::Tensor input_feature,
    at::Tensor outputs,
    at::Tensor weights,
    const at::Tensor offsets_list,
    const at::Tensor resolutions_list,
    const uint32_t N, const uint32_t num_dim, const uint32_t n_features, const uint32_t n_levels
    ) {

    CHECK_CUDA(input_norm_xyz);
    CHECK_CUDA(input_feature);
    CHECK_CUDA(outputs);
    CHECK_CUDA(weights);
    CHECK_CUDA(offsets_list);
    CHECK_CUDA(resolutions_list);

    CHECK_CONTIGUOUS(input_norm_xyz);
    CHECK_CONTIGUOUS(input_feature);
    CHECK_CONTIGUOUS(outputs);
    CHECK_CONTIGUOUS(weights);
    CHECK_CONTIGUOUS(offsets_list);
    CHECK_CONTIGUOUS(resolutions_list);

    CHECK_IS_FLOATING(input_norm_xyz);
    CHECK_IS_FLOATING(input_feature);
    CHECK_IS_FLOATING(outputs);
    CHECK_IS_FLOATING(weights);
    CHECK_IS_INT(offsets_list);
    CHECK_IS_INT(resolutions_list);

    // AT_DISPATCH_FLOATING_TYPES_AND_HALF is the action to lunch a kernal
    // input_feature.scalar_type() indicates the type of data, to decide the type of lunched kernal
    // "grid_creater_forward" indicates the name in traceback, when reporting error...
    // grid_creater_forward_cuda name of function
    // process: first generate an output (already done), and fill data into it.
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    input_feature.scalar_type(), "grid_creater_forward",
    ([&] {
        grid_creater_forward_cuda<scalar_t>(
            input_norm_xyz.data_ptr<float>(),
            input_feature.data_ptr<scalar_t>(),
            outputs.data_ptr<scalar_t>(),
            weights.data_ptr<scalar_t>(),
            offsets_list.data_ptr<int>(),
            resolutions_list.data_ptr<int>(),
            N, num_dim, n_features, n_levels
            );
    })
    );
}

void grid_creater_backward(
    const at::Tensor input_norm_xyz,
    const at::Tensor grad,
    const at::Tensor weights,
    at::Tensor grad_feature,
    const at::Tensor offsets_list,
    const at::Tensor resolutions_list,
    const uint32_t N, const uint32_t num_dim, const uint32_t n_features, const uint32_t n_levels
    ) {

    CHECK_CUDA(input_norm_xyz);
    CHECK_CUDA(grad);
    CHECK_CUDA(weights);
    CHECK_CUDA(grad_feature);
    CHECK_CUDA(offsets_list);
    CHECK_CUDA(resolutions_list);

    CHECK_CONTIGUOUS(input_norm_xyz);
    CHECK_CONTIGUOUS(grad);
    CHECK_CONTIGUOUS(weights);
    CHECK_CONTIGUOUS(grad_feature);
    CHECK_CONTIGUOUS(offsets_list);
    CHECK_CONTIGUOUS(resolutions_list);

    CHECK_IS_FLOATING(input_norm_xyz);
    CHECK_IS_FLOATING(grad);
    CHECK_IS_FLOATING(weights);
    CHECK_IS_FLOATING(grad_feature);
    CHECK_IS_INT(offsets_list);
    CHECK_IS_INT(resolutions_list);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    grad.scalar_type(), "grid_creater_backward", ([&] {
        grid_creater_backward_cuda<scalar_t>(
            input_norm_xyz.data_ptr<float>(),
            grad.data_ptr<scalar_t>(),
            weights.data_ptr<scalar_t>(),
            grad_feature.data_ptr<scalar_t>(),
            offsets_list.data_ptr<int>(),
            resolutions_list.data_ptr<int>(),
            N, num_dim, n_features, n_levels
            );
    }));

}



//------------------

template <typename scalar_t, uint32_t num_dim, uint32_t n_features>  // <scalar_t, num_dim, 2 // num_dim: coords input_dim = 3
__global__ void kernel_grid_determ(
    const float * __restrict__ input_norm_xyz,  //  has been mapped to [0, 1]  [N, num_dim]
    const scalar_t * __restrict__ input_feature,  // [N, n_features]
    int32_t * __restrict__ outputs,   // [offsets_list[-1], n_features+1] (using int32_t now)
    int32_t * __restrict__ weights,   // [offsets_list[-1], n_features+1] (using int32_t now)
    const int * __restrict__ offsets_list,
    const int * __restrict__ resolutions_list,
    const uint32_t N, const uint32_t n_levels
) {
    const uint32_t b = blockIdx.x * blockDim.x + threadIdx.x;  // parallel along N_rays axis
    if (b >= N) return;  // skip unnecessary threads
    const uint32_t level = blockIdx.y;

    input_norm_xyz += b * num_dim;
    input_feature += b * n_features;
    outputs += (uint32_t)offsets_list[level] * n_features;
    weights += (uint32_t)offsets_list[level];

    // check input range (should be in [0, 1])
    bool flag_oob = false;
    #pragma unroll
    for (uint32_t d = 0; d < num_dim; d++) {  // traverse each input_dim
        if (input_norm_xyz[d] < 0 || input_norm_xyz[d] > 1) {
            flag_oob = true;
            return;
        }
    }

    // calculate coordinate (always use float for precision!)
    float pos[num_dim];  // fractional part
    uint32_t pos_grid[num_dim];
    const uint32_t hashmap_size = offsets_list[level + 1] - offsets_list[level];
    const uint32_t resolution = (uint32_t)resolutions_list[level];
    const float scale_factor = 1e4;  // scaling factor to convert float to int32_t

    #pragma unroll
    for (uint32_t d = 0; d < num_dim; d++) {  // traverse each input_dim
        pos[d] = input_norm_xyz[d] * float(resolution - 2) + 0.5f; // resolution = 6: 0->0.5, 1->4.5
        pos_grid[d] = (uint32_t)floorf(pos[d]);
        pos[d] -= (float)pos_grid[d];
    }

    uint32_t index_list[1 << num_dim] = {0};

    #pragma unroll
    for (uint32_t idx = 0; idx < (1 << num_dim); idx++) {  // traverse vertices for trilinear interpolation
        float w = 1.0f;  // weight for interpolation
        uint32_t pos_grid_local[num_dim];

        #pragma unroll
        for (uint32_t d = 0; d < num_dim; d++) {
            if ((idx & (1 << d)) == 0) {  // (1 << d) = {1, 2, 4}
                w *= 1.0f - pos[d];
                pos_grid_local[d] = pos_grid[d];
            } else {
                w *= pos[d];
                pos_grid_local[d] = min(pos_grid[d] + 1, resolution - 1);
            }
        }
        w += 1e-9f;  // avoid edge cases where coordinates lie exactly on voxel boundaries

        uint32_t index = get_grid_index<num_dim, n_features>(0, 0, hashmap_size, resolution, pos_grid_local);
        uint32_t index_m_nfeatures = index * n_features;

        #pragma unroll
        for (uint32_t ch = 0; ch < n_features; ch++) {
            atomicAdd(&outputs[index_m_nfeatures + ch], (int32_t)(w * input_feature[ch] * scale_factor));
        }
        atomicAdd(&weights[index], (int32_t)(w * scale_factor));
    }
}

template <typename scalar_t, uint32_t num_dim>
void kernel_grid_wrapper_determ(
    const float *input_norm_xyz,
    const scalar_t *input_feature,
    int32_t *outputs,
    int32_t *weights,
    const int *offsets_list,
    const int *resolutions_list,
    const uint32_t N, const uint32_t n_features, const uint32_t n_levels
    ) {
    // blocks and threads are defined here
    static constexpr uint32_t N_THREAD = 512;  // in fact (521, 1, 1)
    // div_round_up is (N + N_THREAD - 1) / N_THREAD
    const dim3 blocks_hashgrid = { div_round_up(N, N_THREAD), n_levels, 1 };
    switch (n_features) {
        // at the input of function, there might have "packed_accessor". it is used to transform data types.
        case 1: kernel_grid_determ<scalar_t, num_dim, 1><<<blocks_hashgrid, N_THREAD>>>(input_norm_xyz, input_feature, outputs, weights, offsets_list, resolutions_list, N, n_levels); break;
        case 2: kernel_grid_determ<scalar_t, num_dim, 2><<<blocks_hashgrid, N_THREAD>>>(input_norm_xyz, input_feature, outputs, weights, offsets_list, resolutions_list, N, n_levels); break;
        case 4: kernel_grid_determ<scalar_t, num_dim, 4><<<blocks_hashgrid, N_THREAD>>>(input_norm_xyz, input_feature, outputs, weights, offsets_list, resolutions_list, N, n_levels); break;
        case 8: kernel_grid_determ<scalar_t, num_dim, 8><<<blocks_hashgrid, N_THREAD>>>(input_norm_xyz, input_feature, outputs, weights, offsets_list, resolutions_list, N, n_levels); break;
        case 16: kernel_grid_determ<scalar_t, num_dim, 16><<<blocks_hashgrid, N_THREAD>>>(input_norm_xyz, input_feature, outputs, weights, offsets_list, resolutions_list, N, n_levels); break;
        case 32: kernel_grid_determ<scalar_t, num_dim, 32><<<blocks_hashgrid, N_THREAD>>>(input_norm_xyz, input_feature, outputs, weights, offsets_list, resolutions_list, N, n_levels); break;
        case 48: kernel_grid_determ<scalar_t, num_dim, 48><<<blocks_hashgrid, N_THREAD>>>(input_norm_xyz, input_feature, outputs, weights, offsets_list, resolutions_list, N, n_levels); break;
        case 64: kernel_grid_determ<scalar_t, num_dim, 64><<<blocks_hashgrid, N_THREAD>>>(input_norm_xyz, input_feature, outputs, weights, offsets_list, resolutions_list, N, n_levels); break;
        case 128: kernel_grid_determ<scalar_t, num_dim, 128><<<blocks_hashgrid, N_THREAD>>>(input_norm_xyz, input_feature, outputs, weights, offsets_list, resolutions_list, N, n_levels); break;
        case 256: kernel_grid_determ<scalar_t, num_dim, 256><<<blocks_hashgrid, N_THREAD>>>(input_norm_xyz, input_feature, outputs, weights, offsets_list, resolutions_list, N, n_levels); break;
        default: throw std::runtime_error{"GridEncoding: n_features must be 1, 2, 4, 8, 16 or 32."};
    }
}

template <typename scalar_t>
void grid_creater_forward_cuda_determ(
    const float *input_norm_xyz,
    const scalar_t * input_feature,
    int32_t *outputs,
    int32_t *weights,
    const int *offsets_list,
    const int *resolutions_list,
    const uint32_t N, const uint32_t num_dim, const uint32_t n_features, const uint32_t n_levels
    ) {
    switch (num_dim) {
        case 1: kernel_grid_wrapper_determ<scalar_t, 1>(input_norm_xyz, input_feature, outputs, weights, offsets_list, resolutions_list, N, n_features, n_levels); break;
        case 2: kernel_grid_wrapper_determ<scalar_t, 2>(input_norm_xyz, input_feature, outputs, weights, offsets_list, resolutions_list, N, n_features, n_levels); break;
        case 3: kernel_grid_wrapper_determ<scalar_t, 3>(input_norm_xyz, input_feature, outputs, weights, offsets_list, resolutions_list, N, n_features, n_levels); break;
        default: throw std::runtime_error{"GridEncoding: num_dim must be 1, 2, 3."};
    }
}

void grid_creater_forward_determ(
    const at::Tensor input_norm_xyz,
    const at::Tensor input_feature,
    at::Tensor outputs,
    at::Tensor weights,
    const at::Tensor offsets_list,
    const at::Tensor resolutions_list,
    const uint32_t N, const uint32_t num_dim, const uint32_t n_features, const uint32_t n_levels
    ) {

    CHECK_CUDA(input_norm_xyz);
    CHECK_CUDA(input_feature);
    CHECK_CUDA(outputs);
    CHECK_CUDA(weights);
    CHECK_CUDA(offsets_list);
    CHECK_CUDA(resolutions_list);

    CHECK_CONTIGUOUS(input_norm_xyz);
    CHECK_CONTIGUOUS(input_feature);
    CHECK_CONTIGUOUS(outputs);
    CHECK_CONTIGUOUS(weights);
    CHECK_CONTIGUOUS(offsets_list);
    CHECK_CONTIGUOUS(resolutions_list);

    CHECK_IS_FLOATING(input_norm_xyz);
    CHECK_IS_FLOATING(input_feature);
    CHECK_IS_INT(outputs);
    CHECK_IS_INT(weights);
    CHECK_IS_INT(offsets_list);
    CHECK_IS_INT(resolutions_list);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    input_feature.scalar_type(), "grid_creater_forward",
    ([&] {
        grid_creater_forward_cuda_determ<scalar_t>(
            input_norm_xyz.data_ptr<float>(),
            input_feature.data_ptr<scalar_t>(),
            outputs.data_ptr<int32_t>(),
            weights.data_ptr<int32_t>(),
            offsets_list.data_ptr<int>(),
            resolutions_list.data_ptr<int>(),
            N, num_dim, n_features, n_levels
            );
    })
    );
}