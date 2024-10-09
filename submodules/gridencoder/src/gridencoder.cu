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
    return (index % hashmap_size) * n_features + ch;
}

// kernel_grid<scalar_t, num_dim, 2><<<blocks_hashgrid, N_THREAD>>>
// (inputs, embeddings, offsets_list, outputs, N, n_levels, S, H, dy_dx, gridtype, align_corners, interp);
// N: N_rays
// n_levels: level
// S: log2(per_level_scale)
// H: base_resolution
////// One CPU kernel calls one GPU grid, one GPU grid contains several blocks, one block contains several threads
template <typename scalar_t, uint32_t num_dim, uint32_t n_features>  // <scalar_t, num_dim, 2 // num_dim: coords input_dim = 3
// __global__ means called by CPU and conducted by GPU
// always no return, so always void
__global__ void kernel_grid(
    const float * __restrict__ inputs,  //  has been mapped to [0, 1]  [N, num_dim]
    const scalar_t * __restrict__ grid,  // here grid means hashgrid not processing grid in GPU shape:[offset*n_features]?
    const int * __restrict__ offsets_list,
    const int * __restrict__ resolutions_list,
    scalar_t * __restrict__ outputs,   // non-constant
    const uint32_t N, const uint32_t n_levels
    ) {
    // grid > block > thread
    // blockIdx.x is idx of the block along x axis, blockDim.x is th block width, threadIdx.x is idx along the width of thread
    // get the place of corresponding parallel computing point
    // get b: the index of [0, N_rays)
    const uint32_t b = blockIdx.x * blockDim.x + threadIdx.x;  // parallel along N_rays axis

    if (b >= N) return;  // deprecate not needed threads

    const uint32_t level = blockIdx.y;

    // locate  why these variables are changed? because they are pointers when defined? --> get the locate of the data to be processed
    grid += (uint32_t)offsets_list[level] * n_features;
    inputs += b * num_dim;
    outputs += level * N * n_features + b * n_features;

    // check input range (should be in [0, 1])
    bool flag_oob = false;
    #pragma unroll
    for (uint32_t d = 0; d < num_dim; d++) {  // traverse each input_dim
        if (inputs[d] < 0 || inputs[d] > 1) {
            flag_oob = true;
        }
    }
    // if input out of bound, just set output to 0
    if (flag_oob) {
        #pragma unroll
        for (uint32_t ch = 0; ch < n_features; ch++) {  // traverse each feature_dim
            outputs[ch] = 0;
        }
        return;
    }

    const uint32_t hashmap_size = offsets_list[level + 1] - offsets_list[level];
    // exp2f(level * S) = 2 ^ (level*S) = 2 ^ (level*log2(per_level_scale)) = 2 ^ (log2(per_level_scale)*level) = per_level_scale ^ level
    // const uint32_t resolution = (uint32_t)ceil(exp2f(level * S) * H);
    const uint32_t resolution = (uint32_t)resolutions_list[level];
    /*  resolution = 6:
          (n,0)->|       |<-(n,1)
                0 0 0 0 0 0
                0 1 1 1 1 0
                0 1 1 1 1 0
                0 1 1 1 1 0
                0 1 1 1 1 0
                0 0 0 0 0 0
    */

    // calculate coordinate (always use float for precision!)
    float pos[num_dim];
    float pos_deriv[num_dim];
    uint32_t pos_grid[num_dim];

    #pragma unroll
    for (uint32_t d = 0; d < num_dim; d++) {  // traverse each input_dim

        pos[d] = inputs[d] * float(resolution - 2) + 0.5; // resolution = 6: 0->0.5, 1->4.5
        pos_grid[d] = (uint32_t)floorf(pos[d]);
        pos[d] -= (float)pos_grid[d];
        pos_deriv[d] = 1.0f;
    }

    // interpolate
    scalar_t results[n_features] = {0}; // temp results in register

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

        #pragma unroll
        for (uint32_t ch = 0; ch < n_features; ch++) {
            results[ch] += w * grid[index + ch];  // index is already multipled by C.
        }
    }
    // writing to global memory (slow)
    #pragma unroll
    for (uint32_t ch = 0; ch < n_features; ch++) {
        outputs[ch] = results[ch];
    }
}


// grad, inputs, embeddings, offsets_list, grad_embeddings, N, num_dim, n_features, n_levels, max_level, S, H, dy_dx, grad_inputs, gridtype, align_corners, interpolation
template <typename scalar_t, uint32_t num_dim, uint32_t n_features, uint32_t n_features_per_thread>  // n_features_per_thread is n_features_per_thread
__global__ void kernel_grid_backward(
    const scalar_t * __restrict__ grad,  // grad is the gradient from loss
    const float * __restrict__ inputs,
    const scalar_t * __restrict__ grid,
    const int * __restrict__ offsets_list,
    const int * __restrict__ resolutions_list,
    scalar_t * __restrict__ grad_grid,   // same type as grad
    const uint32_t N, const uint32_t n_levels
    ) {
    const uint32_t b = (blockIdx.x * blockDim.x + threadIdx.x) * n_features_per_thread / n_features;
    if (b >= N) return;
    const uint32_t level = blockIdx.y;
    const uint32_t ch = (blockIdx.x * blockDim.x + threadIdx.x) * n_features_per_thread - b * n_features;

    // locate
    grad_grid += offsets_list[level] * n_features;
    inputs += b * num_dim;
    grad += blockIdx.y * N * n_features + b * n_features + ch; // n_levels, N, n_features

    const uint32_t hashmap_size = offsets_list[level + 1] - offsets_list[level];
    const uint32_t resolution = (uint32_t)resolutions_list[level];

    // check input range (should be in [0, 1])
    #pragma unroll
    for (uint32_t d = 0; d < num_dim; d++) {
        if (inputs[d] < 0 || inputs[d] > 1) {
            return; // grad is init as 0, so we simply return.
        }
    }

    // calculate coordinate
    float pos[num_dim];
    uint32_t pos_grid[num_dim];

    // same as forward process
    #pragma unroll
    for (uint32_t d = 0; d < num_dim; d++) {  // traverse each input_dim
        pos[d] = inputs[d] * float(resolution - 2) + 0.5; // resolution = 6: 0->0.5, 1->4.5
        pos_grid[d] = (uint32_t)floorf(pos[d]);
        pos[d] -= (float)pos_grid[d];
    }

    scalar_t grad_cur[n_features_per_thread] = {0}; // fetch to register
    #pragma unroll
    for (uint32_t c = 0; c < n_features_per_thread; c++) {
        grad_cur[c] = grad[c];
    }

    #pragma unroll
    for (uint32_t idx = 0; idx < (1 << num_dim); idx++) {
        float w = 1;
        uint32_t pos_grid_local[num_dim];

        #pragma unroll
        for (uint32_t d = 0; d < num_dim; d++) {
            if ((idx & (1 << d)) == 0) {
                w *= 1 - pos[d];
                pos_grid_local[d] = pos_grid[d];
            } else {
                w *= pos[d];
                pos_grid_local[d] = min(pos_grid[d] + 1, resolution - 1);
            }
        }
        w += 1e-9;  // avoid cases input coordinates are right at voxel lines

        uint32_t index = get_grid_index<num_dim, n_features>(0, ch, hashmap_size, resolution, pos_grid_local);

        // for atomicAdd, see https://www.cnblogs.com/biglucky/p/4283476.html. It is plus operation
        // atomicAdd for __half is slow (especially for large values), so we use __half2 if n_features_per_thread % 2 == 0
        // TODO: use float which is better than __half, if n_features_per_thread % 2 != 0
        if (std::is_same<scalar_t, at::Half>::value && n_features_per_thread % 2 == 0) {  // in this code it should be in this line
            #pragma unroll
            for (uint32_t c = 0; c < n_features_per_thread; c += 2) {
                // process two __half at once (by interpreting as a __half2)
                __half2 v = {(__half)(w * grad_cur[c]), (__half)(w * grad_cur[c + 1])};
                atomicAdd((__half2*)&grad_grid[index + c], v);
            }
        // float, or __half when n_features_per_thread % 2 != 0 (which means C == 1)
        } else {
            #pragma unroll
            for (uint32_t c = 0; c < n_features_per_thread; c++) {
                atomicAdd(&grad_grid[index + c], w * grad_cur[c]);
            }
        }
    }
}


template <typename scalar_t, uint32_t num_dim, uint32_t n_features>
__global__ void kernel_input_backward(
    const scalar_t * __restrict__ grad,
    const scalar_t * __restrict__ dy_dx,
    scalar_t * __restrict__ grad_inputs,
    uint32_t N, uint32_t n_levels
    ) {
    const uint32_t t = threadIdx.x + blockIdx.x * blockDim.x;
    if (t >= N * num_dim) return;

    const uint32_t b = t / num_dim;
    const uint32_t d = t - b * num_dim;

    dy_dx += b * n_levels * num_dim * n_features;

    scalar_t result = 0;

    # pragma unroll
    for (int l = 0; l < n_levels; l++) {
        # pragma unroll
        for (int ch = 0; ch < n_features; ch++) {
            result += grad[l * N * n_features + b * n_features + ch] * dy_dx[l * num_dim * n_features + d * n_features + ch];
        }
    }

    grad_inputs[t] = result;
}


template <typename scalar_t, uint32_t num_dim>
void kernel_grid_wrapper(
    const float *inputs,
    const scalar_t *embeddings,
    const int *offsets_list,
    const int *resolutions_list,
    scalar_t *outputs,
    const uint32_t N, const uint32_t n_features, const uint32_t n_levels
    ) {
    // blocks and threads are defined here
    static constexpr uint32_t N_THREAD = 512;
    // div_round_up is (N + N_THREAD - 1) / N_THREAD
    const dim3 blocks_hashgrid = { div_round_up(N, N_THREAD), n_levels, 1 };
    switch (n_features) {
        // at the input of function, there might have "packed_accessor". it is used to transform data types.
        case 1: kernel_grid<scalar_t, num_dim, 1><<<blocks_hashgrid, N_THREAD>>>(inputs, embeddings, offsets_list, resolutions_list, outputs, N, n_levels); break;
        case 2: kernel_grid<scalar_t, num_dim, 2><<<blocks_hashgrid, N_THREAD>>>(inputs, embeddings, offsets_list, resolutions_list, outputs, N, n_levels); break;
        case 4: kernel_grid<scalar_t, num_dim, 4><<<blocks_hashgrid, N_THREAD>>>(inputs, embeddings, offsets_list, resolutions_list, outputs, N, n_levels); break;
        case 8: kernel_grid<scalar_t, num_dim, 8><<<blocks_hashgrid, N_THREAD>>>(inputs, embeddings, offsets_list, resolutions_list, outputs, N, n_levels); break;
        case 16: kernel_grid<scalar_t, num_dim, 16><<<blocks_hashgrid, N_THREAD>>>(inputs, embeddings, offsets_list, resolutions_list, outputs, N, n_levels); break;
        case 32: kernel_grid<scalar_t, num_dim, 32><<<blocks_hashgrid, N_THREAD>>>(inputs, embeddings, offsets_list, resolutions_list, outputs, N, n_levels); break;
        case 48: kernel_grid<scalar_t, num_dim, 48><<<blocks_hashgrid, N_THREAD>>>(inputs, embeddings, offsets_list, resolutions_list, outputs, N, n_levels); break;
        case 64: kernel_grid<scalar_t, num_dim, 64><<<blocks_hashgrid, N_THREAD>>>(inputs, embeddings, offsets_list, resolutions_list, outputs, N, n_levels); break;
        case 128: kernel_grid<scalar_t, num_dim, 128><<<blocks_hashgrid, N_THREAD>>>(inputs, embeddings, offsets_list, resolutions_list, outputs, N, n_levels); break;
        case 256: kernel_grid<scalar_t, num_dim, 256><<<blocks_hashgrid, N_THREAD>>>(inputs, embeddings, offsets_list, resolutions_list, outputs, N, n_levels); break;
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
void grid_encode_forward_cuda(
    const float *inputs,
    const scalar_t *embeddings,
    const int *offsets_list,
    const int *resolutions_list,
    scalar_t *outputs,
    const uint32_t N, const uint32_t num_dim, const uint32_t n_features, const uint32_t n_levels
    ) {
    switch (num_dim) {
        case 1: kernel_grid_wrapper<scalar_t, 1>(inputs, embeddings, offsets_list, resolutions_list, outputs, N, n_features, n_levels); break;
        case 2: kernel_grid_wrapper<scalar_t, 2>(inputs, embeddings, offsets_list, resolutions_list, outputs, N, n_features, n_levels); break;
        case 3: kernel_grid_wrapper<scalar_t, 3>(inputs, embeddings, offsets_list, resolutions_list, outputs, N, n_features, n_levels); break;
        default: throw std::runtime_error{"GridEncoding: num_dim must be 1, 2, 3."};
    }
}

template <typename scalar_t, uint32_t num_dim>
void kernel_grid_backward_wrapper(
    const scalar_t *grad,
    const float *inputs,
    const scalar_t *embeddings,
    const int *offsets_list,
    const int *resolutions_list,
    scalar_t *grad_embeddings,
    const uint32_t N, const uint32_t n_features, const uint32_t n_levels
    ) {
    static constexpr uint32_t N_THREAD = 256;
    const uint32_t n_features_per_thread = std::min(2u, n_features); // n_features_per_thread
    const dim3 blocks_hashgrid = { div_round_up(N * n_features / n_features_per_thread, N_THREAD), n_levels, 1 };
    switch (n_features) {
        case 1: kernel_grid_backward<scalar_t, num_dim, 1, 1><<<blocks_hashgrid, N_THREAD>>>(grad, inputs, embeddings, offsets_list, resolutions_list, grad_embeddings, N, n_levels); break;
        case 2: kernel_grid_backward<scalar_t, num_dim, 2, 2><<<blocks_hashgrid, N_THREAD>>>(grad, inputs, embeddings, offsets_list, resolutions_list, grad_embeddings, N, n_levels); break;
        case 4: kernel_grid_backward<scalar_t, num_dim, 4, 2><<<blocks_hashgrid, N_THREAD>>>(grad, inputs, embeddings, offsets_list, resolutions_list, grad_embeddings, N, n_levels); break;
        case 8: kernel_grid_backward<scalar_t, num_dim, 8, 2><<<blocks_hashgrid, N_THREAD>>>(grad, inputs, embeddings, offsets_list, resolutions_list, grad_embeddings, N, n_levels); break;
        case 16: kernel_grid_backward<scalar_t, num_dim, 16, 2><<<blocks_hashgrid, N_THREAD>>>(grad, inputs, embeddings, offsets_list, resolutions_list, grad_embeddings, N, n_levels); break;
        case 32: kernel_grid_backward<scalar_t, num_dim, 32, 2><<<blocks_hashgrid, N_THREAD>>>(grad, inputs, embeddings, offsets_list, resolutions_list, grad_embeddings, N, n_levels); break;
        case 48: kernel_grid_backward<scalar_t, num_dim, 48, 2><<<blocks_hashgrid, N_THREAD>>>(grad, inputs, embeddings, offsets_list, resolutions_list, grad_embeddings, N, n_levels); break;
        case 64: kernel_grid_backward<scalar_t, num_dim, 64, 2><<<blocks_hashgrid, N_THREAD>>>(grad, inputs, embeddings, offsets_list, resolutions_list, grad_embeddings, N, n_levels); break;
        case 128: kernel_grid_backward<scalar_t, num_dim, 128, 2><<<blocks_hashgrid, N_THREAD>>>(grad, inputs, embeddings, offsets_list, resolutions_list, grad_embeddings, N, n_levels); break;
        case 256: kernel_grid_backward<scalar_t, num_dim, 256, 2><<<blocks_hashgrid, N_THREAD>>>(grad, inputs, embeddings, offsets_list, resolutions_list, grad_embeddings, N, n_levels); break;
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
void grid_encode_backward_cuda(
    const scalar_t *grad,
    const float *inputs,
    const scalar_t *embeddings,
    const int *offsets_list,
    const int *resolutions_list,
    scalar_t *grad_embeddings,
    const uint32_t N, const uint32_t num_dim, const uint32_t n_features, const uint32_t n_levels
    ) {
    switch (num_dim) {
        case 1: kernel_grid_backward_wrapper<scalar_t, 1>(grad, inputs, embeddings, offsets_list, resolutions_list, grad_embeddings, N, n_features, n_levels); break;
        case 2: kernel_grid_backward_wrapper<scalar_t, 2>(grad, inputs, embeddings, offsets_list, resolutions_list, grad_embeddings, N, n_features, n_levels); break;
        case 3: kernel_grid_backward_wrapper<scalar_t, 3>(grad, inputs, embeddings, offsets_list, resolutions_list, grad_embeddings, N, n_features, n_levels); break;
        default: throw std::runtime_error{"GridEncoding: num_dim must be 1, 2, 3."};
    }
}



void grid_encode_forward(
    const at::Tensor inputs,
    const at::Tensor embeddings,
    const at::Tensor offsets_list,
    const at::Tensor resolutions_list,
    at::Tensor outputs,
    const uint32_t N, const uint32_t num_dim, const uint32_t n_features, const uint32_t n_levels
    ) {

    CHECK_CUDA(inputs);
    CHECK_CUDA(embeddings);
    CHECK_CUDA(offsets_list);
    CHECK_CUDA(resolutions_list);
    CHECK_CUDA(outputs);

    CHECK_CONTIGUOUS(inputs);
    CHECK_CONTIGUOUS(embeddings);
    CHECK_CONTIGUOUS(offsets_list);
    CHECK_CONTIGUOUS(resolutions_list);
    CHECK_CONTIGUOUS(outputs);

    CHECK_IS_FLOATING(inputs);
    CHECK_IS_FLOATING(embeddings);
    CHECK_IS_INT(offsets_list);
    CHECK_IS_INT(resolutions_list);
    CHECK_IS_FLOATING(outputs);

    // AT_DISPATCH_FLOATING_TYPES_AND_HALF is the action to lunch a kernal
    // embeddings.scalar_type() indicates the type of data, to decide the type of lunched kernal
    // "grid_encode_forward" indicates the name in traceback, when reporting error...
    // grid_encode_forward_cuda name of function
    // process: first generate an output (already done), and fill data into it.
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    embeddings.scalar_type(), "grid_encode_forward",
    ([&] {
        grid_encode_forward_cuda<scalar_t>(
            inputs.data_ptr<float>(),
            embeddings.data_ptr<scalar_t>(),
            offsets_list.data_ptr<int>(),
            resolutions_list.data_ptr<int>(),
            outputs.data_ptr<scalar_t>(),
            N, num_dim, n_features, n_levels
            );
    })
    );
}

void grid_encode_backward(
    const at::Tensor grad,
    const at::Tensor inputs,
    const at::Tensor embeddings,
    const at::Tensor offsets_list,
    const at::Tensor resolutions_list,
    at::Tensor grad_embeddings,
    const uint32_t N, const uint32_t num_dim, const uint32_t n_features, const uint32_t n_levels
    ) {

    CHECK_CUDA(grad);
    CHECK_CUDA(inputs);
    CHECK_CUDA(embeddings);
    CHECK_CUDA(offsets_list);
    CHECK_CUDA(resolutions_list);
    CHECK_CUDA(grad_embeddings);

    CHECK_CONTIGUOUS(grad);
    CHECK_CONTIGUOUS(inputs);
    CHECK_CONTIGUOUS(embeddings);
    CHECK_CONTIGUOUS(offsets_list);
    CHECK_CONTIGUOUS(resolutions_list);
    CHECK_CONTIGUOUS(grad_embeddings);

    CHECK_IS_FLOATING(grad);
    CHECK_IS_FLOATING(inputs);
    CHECK_IS_FLOATING(embeddings);
    CHECK_IS_INT(offsets_list);
    CHECK_IS_INT(resolutions_list);
    CHECK_IS_FLOATING(grad_embeddings);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    grad.scalar_type(), "grid_encode_backward", ([&] {
        grid_encode_backward_cuda<scalar_t>(
            grad.data_ptr<scalar_t>(),
            inputs.data_ptr<float>(),
            embeddings.data_ptr<scalar_t>(),
            offsets_list.data_ptr<int>(),
            resolutions_list.data_ptr<int>(),
            grad_embeddings.data_ptr<scalar_t>(),
            N, num_dim, n_features, n_levels
            );
    }));

}
