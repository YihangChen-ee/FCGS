#ifndef _HASH_ENCODE_H
#define _HASH_ENCODE_H

#include <stdint.h>
#include <torch/torch.h>

// inputs: [N, num_dim], float, in [0, 1]
// embeddings: [offsets[-1], n_features], float
// offsets: [n_levels + 1], uint32_t
// outputs: [N, n_levels * n_features], float

void grid_encode_forward(
    const at::Tensor inputs, 
    const at::Tensor embeddings, 
    const at::Tensor offsets_list,
    const at::Tensor resolutions_list, 
    at::Tensor outputs, 
    const uint32_t N, const uint32_t num_dim, const uint32_t n_features, const uint32_t n_levels
    );

void grid_encode_backward(
    const at::Tensor grad, 
    const at::Tensor inputs, 
    const at::Tensor embeddings, 
    const at::Tensor offsets_list,
    const at::Tensor resolutions_list,
    at::Tensor grad_embeddings,
    const uint32_t N, const uint32_t num_dim, const uint32_t n_features, const uint32_t n_levels
    );

#endif