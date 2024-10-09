import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.cuda.amp import custom_bwd, custom_fwd

try:
    import _gridencoder as _backend
except ImportError:
    from .backend import _backend

_gridtype_to_id = {
    'hash': 0,
    'tiled': 1,
}

_interp_to_id = {
    'linear': 0,
    'smoothstep': 1,
}

class STE_binary(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        # out = torch.sign(input)
        p = (input >= 0) * (+1.0)
        n = (input < 0) * (-1.0)
        out = p + n
        return out
    @staticmethod
    def backward(ctx, grad_output):
        # mask: to ensure x belongs to (-1, 1)
        input, = ctx.saved_tensors
        i2 = input.clone().detach()
        i3 = torch.clamp(i2, -1, 1)
        mask = (i3 == i2) + 0.0
        return grad_output * mask


class STE_multistep(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, Q):
        return torch.round(input/Q)*Q
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class _grid_encode(Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, inputs, embeddings, offsets_list, resolutions_list, calc_grad_inputs=False, max_level=None):
        # inputs: [N, num_dim], float in [0, 1]
        # embeddings: [sO, n_features], float. self.embeddings = nn.Parameter(torch.empty(offset, n_features))
        # offsets_list: [n_levels + 1], int
        # RETURN: [N, F], float

        inputs = inputs.contiguous()

        N, num_dim = inputs.shape # batch size, coord dim # N_rays, 3
        n_levels = offsets_list.shape[0] - 1 # level # 层数=16
        n_features = embeddings.shape[1] # embedding dim for each level # 就是channel数=2

        max_level = n_levels if max_level is None else min(max_level, n_levels)

        # manually handle autocast (only use half precision embeddings, inputs must be float for enough precision)
        # if n_features % 2 != 0, force float, since half for atomicAdd is very slow.
        if torch.is_autocast_enabled() and n_features % 2 == 0:
            embeddings = embeddings.to(torch.half)

        # n_levels first, optimize cache for cuda kernel, but needs an extra permute later
        outputs = torch.empty(n_levels, N, n_features, device=inputs.device, dtype=embeddings.dtype)  # 创建一个buffer给cuda填充
        # outputs = [hash层数=16, N_rays, channels=2]

        # zero init if we only calculate partial levels
        if max_level < n_levels: outputs.zero_()

        if calc_grad_inputs:  # inputs.requires_grad
            dy_dx = torch.empty(N, n_levels * num_dim * n_features, device=inputs.device, dtype=embeddings.dtype)
            if max_level < n_levels: dy_dx.zero_()
        else:
            dy_dx = None

        _backend.grid_encode_forward(
            inputs,
            embeddings,
            offsets_list,
            resolutions_list,
            outputs,
            N, num_dim, n_features, n_levels, max_level,
            dy_dx
            )

        # permute back to [N, n_levels * n_features]  # [N_rays, hash层数=16 * channels=2]
        outputs = outputs.permute(1, 0, 2).reshape(N, n_levels * n_features)

        ctx.save_for_backward(inputs, embeddings, offsets_list, resolutions_list, dy_dx)
        ctx.dims = [N, num_dim, n_features, n_levels, max_level]

        return outputs

    @staticmethod
    #@once_differentiable
    @custom_bwd
    def backward(ctx, grad):

        inputs, embeddings, offsets_list, resolutions_list, dy_dx = ctx.saved_tensors
        N, num_dim, n_features, n_levels, max_level = ctx.dims

        # grad: [N, n_levels * n_features] --> [n_levels, N, n_features]
        grad = grad.view(N, n_levels, n_features).permute(1, 0, 2).contiguous()

        # 是梯度的占位变量，和本体的形状相同，因为代码里是直接加原始值的，所以这里得定义为全0
        grad_embeddings = torch.zeros_like(embeddings)

        if dy_dx is not None:
            grad_inputs = torch.zeros_like(inputs, dtype=embeddings.dtype)
        else:
            grad_inputs = None

        _backend.grid_encode_backward(
            grad,
            inputs,
            embeddings,
            offsets_list,
            resolutions_list,
            grad_embeddings,
            N, num_dim, n_features, n_levels, max_level,
            dy_dx,
            grad_inputs
            )

        if dy_dx is not None:
            grad_inputs = grad_inputs.to(inputs.dtype)

        return grad_inputs, grad_embeddings, None, None, None, None


grid_encode = _grid_encode.apply


class GridEncoder(nn.Module):
    def __init__(self,
                 num_dim=3,
                 n_features=2,
                 resolutions_list=(16, 23, 32, 46, 64, 92, 128, 184, 256, 368, 512, 736),
                 log2_hashmap_size=19,
                 ste_binary=False,
                 ):
        super().__init__()

        resolutions_list = torch.tensor(resolutions_list).to(torch.int)
        n_levels = resolutions_list.numel()

        self.num_dim = num_dim # coord dims, 2 or 3
        self.n_levels = n_levels # num levels, each level multiply resolution by 2
        self.n_features = n_features # encode channels per level
        self.log2_hashmap_size = log2_hashmap_size
        self.output_dim = n_levels * n_features
        self.ste_binary = ste_binary

        # allocate parameters
        offsets_list = []  # 每层hashtable长度的cumsum
        offset = 0  # 用于统计所有层加起来一共需要多少长度的hashtable
        self.max_params = 2 ** log2_hashmap_size  # 按论文算法的每层的hashtable长度上限
        for i in range(n_levels):
            resolution = resolutions_list[i].item()
            params_in_level = min(self.max_params, resolution ** num_dim) # limit max number
            params_in_level = int(np.ceil(params_in_level / 8) * 8) # make divisible
            offsets_list.append(offset)
            offset += params_in_level
        offsets_list.append(offset)
        offsets_list = torch.from_numpy(np.array(offsets_list, dtype=np.int32))
        self.register_buffer('offsets_list', offsets_list)
        self.register_buffer('resolutions_list', resolutions_list)

        self.n_params = offsets_list[-1] * n_features  # 所有的params的数量

        # parameters
        self.embeddings = nn.Parameter(torch.empty(offset, n_features))

        self.reset_parameters()

        self.n_output_dims = n_levels * n_features

    def reset_parameters(self):
        std = 1e-4
        self.embeddings.data.uniform_(-std, std)

    def __repr__(self):
        return f"GridEncoder: num_dim={self.num_dim} n_levels={self.n_levels} n_features={self.n_features} resolution={self.base_resolution} -> {int(round(self.base_resolution * self.per_level_scale ** (self.n_levels - 1)))} per_level_scale={self.per_level_scale:.4f} params={tuple(self.embeddings.shape)} gridtype={self.gridtype} align_corners={self.align_corners} interpolation={self.interpolation}"

    def forward(self, inputs, max_level=None):
        # inputs: [..., num_dim], normalized real world positions in [-1, 1]
        # max_level: only calculate first max_level levels (None will use all levels)
        # return: [..., n_levels * n_features]

        #print('inputs', inputs.shape, inputs.dtype, inputs.min().item(), inputs.max().item())

        prefix_shape = list(inputs.shape[:-1])
        inputs = inputs.view(-1, self.num_dim)

        if self.ste_binary:
            embeddings = STE_binary.apply(self.embeddings)
        else:
            embeddings = self.embeddings
        outputs = grid_encode(inputs, embeddings, self.offsets_list, self.resolutions_list, inputs.requires_grad, max_level)
        outputs = outputs.view(prefix_shape + [self.output_dim])

        #print('outputs', outputs.shape, outputs.dtype, outputs.min().item(), outputs.max().item())

        return outputs