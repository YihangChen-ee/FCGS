#ifndef _HASH_ENCODE_H
#define _HASH_ENCODE_H

#include <stdint.h>
#include <torch/torch.h>

// inputs: [N, num_dim], float, in [0, 1]
// embeddings: [offsets[-1], n_features], float
// offsets: [n_levels + 1], uint32_t
// outputs: [N, n_levels * n_features], float

void grid_creater_forward_determ(
    const at::Tensor input_norm_xyz,
    const at::Tensor input_feature,
    at::Tensor outputs,
    at::Tensor weights,
    const at::Tensor offsets_list,
    const at::Tensor resolutions_list,
    const uint32_t N, const uint32_t num_dim, const uint32_t n_features, const uint32_t n_levels
    );

void grid_creater_forward(
    const at::Tensor input_norm_xyz,
    const at::Tensor input_feature,
    at::Tensor outputs,
    at::Tensor weights,
    const at::Tensor offsets_list,
    const at::Tensor resolutions_list,
    const uint32_t N, const uint32_t num_dim, const uint32_t n_features, const uint32_t n_levels
    );

void grid_creater_backward(
    const at::Tensor input_norm_xyz,
    const at::Tensor grad,
    const at::Tensor weights,
    at::Tensor grad_feature,
    const at::Tensor offsets_list,
    const at::Tensor resolutions_list,
    const uint32_t N, const uint32_t num_dim, const uint32_t n_features, const uint32_t n_levels
    );

#endif