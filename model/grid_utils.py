import torch
import torch.nn as nn
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd
import _gridcreater as gc
import _gridencoder as ge
import _freqencoder as fe

def normalize_xyz(xyz_orig, K=3, means=None, stds=None):
    if means == None:
        xyz_orig = xyz_orig.detach()
        means = torch.mean(xyz_orig, dim=0, keepdim=True)
        stds = torch.std(xyz_orig, dim=0, keepdim=True)

    lower_bound = means - K * stds
    upper_bound = means + K * stds

    norm_xyz = (xyz_orig - lower_bound) / (upper_bound - lower_bound)
    norm_xyz_clamp = torch.clamp(norm_xyz, min=0, max=1)
    mask_xyz = torch.all((norm_xyz == norm_xyz_clamp) + 0.0, dim=1) + 0.0
    # mask_xyz: [xyz_orig.shape[0]]
    return norm_xyz, norm_xyz_clamp, mask_xyz


class _grid_creater(Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input_norm_xyz, input_feature, resolutions_list, offsets_list, determ=False):
        # input_norm_xyz: [N, num_dim], float in [0, 1]
        # input_feature: [N, num_feature]
        # resolutions_list: [16, ..., 128]
        # offsets_list: [16, ..., 128]
        input_norm_xyz = input_norm_xyz.contiguous()
        input_feature = input_feature.contiguous()

        N, num_dim = input_norm_xyz.shape # batch size, coord dim # N_rays, 3
        n_levels = offsets_list.shape[0] - 1 # level # 层数=16
        n_features = input_feature.shape[1] # embedding dim for each level # 就是channel数=2

        if determ:
            outputs0 = torch.zeros(size=[offsets_list[-1].item(), n_features], device=input_feature.device, dtype=torch.int32)
            weights0 = torch.zeros(size=[offsets_list[-1].item(), 1], device=input_feature.device, dtype=torch.int32)
            outputs = torch.zeros(size=[offsets_list[-1].item(), n_features], device=input_feature.device, dtype=input_feature.dtype)
            weights = torch.zeros(size=[offsets_list[-1].item(), 1], device=input_feature.device, dtype=input_feature.dtype)
            gc.grid_creater_forward_determ(
                input_norm_xyz,
                input_feature,
                outputs0,
                weights0,
                offsets_list,
                resolutions_list,
                N, num_dim, n_features, n_levels
                )
            outputs[...] = outputs0.to(torch.float32) * 1e-4
            weights[...] = weights0.to(torch.float32) * 1e-4
            outputs = outputs.contiguous()
            weights = weights.contiguous()

        else:
            outputs = torch.zeros(size=[offsets_list[-1].item(), n_features], device=input_feature.device, dtype=input_feature.dtype)
            weights = torch.zeros(size=[offsets_list[-1].item(), 1], device=input_feature.device, dtype=input_feature.dtype)
            gc.grid_creater_forward(
                input_norm_xyz,
                input_feature,
                outputs,
                weights,
                offsets_list,
                resolutions_list,
                N, num_dim, n_features, n_levels
                )

        # outputs_cat_weights = torch.cat([outputs, weights], dim=-1)
        outputs_div_weights = outputs / (weights + 1e-9)
        ctx.save_for_backward(input_norm_xyz, weights, offsets_list, resolutions_list)
        ctx.dims = [N, num_dim, n_features, n_levels]

        return outputs_div_weights

    @staticmethod
    @custom_bwd
    def backward(ctx, grad):
        # grad: [offsets_list[-1], n_features]
        grad = grad.contiguous()

        input_norm_xyz, weights, offsets_list, resolutions_list = ctx.saved_tensors
        N, num_dim, n_features, n_levels = ctx.dims

        grad_feature = torch.zeros(size=[N, n_features], device=input_norm_xyz.device, dtype=grad.dtype)

        gc.grid_creater_backward(
            input_norm_xyz,
            grad,
            weights,
            grad_feature,
            offsets_list,
            resolutions_list,
            N, num_dim, n_features, n_levels
            )

        return None, grad_feature, None, None, None

class _grid_encoder(Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, inputs, embeddings, offsets_list, resolutions_list):
        # inputs: [N, num_dim], float in [0, 1]
        # embeddings: [offsets_list[-1], n_features]
        inputs = inputs.contiguous()
        N, num_dim = inputs.shape  # batch size, coord dim # N_rays, 3
        n_levels = offsets_list.shape[0] - 1  # level # 层数=16
        n_features = embeddings.shape[1]  # embedding dim for each level # 就是channel数=2

        outputs = torch.empty(n_levels, N, n_features, device=inputs.device, dtype=embeddings.dtype)

        ge.grid_encode_forward(
            inputs,
            embeddings,
            offsets_list,
            resolutions_list,
            outputs,
            N, num_dim, n_features, n_levels
        )

        outputs = outputs.permute(1, 0, 2).reshape(N, n_levels * n_features)

        ctx.save_for_backward(inputs, embeddings, offsets_list, resolutions_list)
        ctx.dims = [N, num_dim, n_features, n_levels]

        return outputs

    @staticmethod
    @custom_bwd
    def backward(ctx, grad):

        inputs, embeddings, offsets_list, resolutions_list = ctx.saved_tensors
        N, num_dim, n_features, n_levels = ctx.dims

        # grad: [N, n_levels * n_features] --> [n_levels, N, n_features]
        grad = grad.view(N, n_levels, n_features).permute(1, 0, 2).contiguous()

        grad_embeddings = torch.zeros_like(embeddings)

        ge.grid_encode_backward(
            grad,
            inputs,
            embeddings,
            offsets_list,
            resolutions_list,
            grad_embeddings,
            N, num_dim, n_features, n_levels
            )
        return None, grad_embeddings, None, None


class _freq_encoder(Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)  # force float32 for better precision
    def forward(ctx, inputs, degree, output_dim):
        # inputs: [B, input_dim], float
        # RETURN: [B, F], float

        if not inputs.is_cuda: inputs = inputs.cuda()
        inputs = inputs.contiguous()

        B, input_dim = inputs.shape  # batch size, coord dim

        outputs = torch.empty(B, output_dim, dtype=inputs.dtype, device=inputs.device)

        fe.freq_encode_forward(inputs, B, input_dim, degree, output_dim, outputs)

        ctx.save_for_backward(inputs, outputs)
        ctx.dims = [B, input_dim, degree, output_dim]

        return outputs

    @staticmethod
    # @once_differentiable
    @custom_bwd
    def backward(ctx, grad):
        # grad: [B, C * C]

        grad = grad.contiguous()
        inputs, outputs = ctx.saved_tensors
        B, input_dim, degree, output_dim = ctx.dims

        grad_inputs = torch.zeros_like(inputs)
        fe.freq_encode_backward(grad, outputs, B, input_dim, degree, output_dim, grad_inputs)

        return grad_inputs, None, None


freq_encode = _freq_encoder.apply

class FreqEncoder(nn.Module):
    def __init__(self, input_dim=3, degree=4):
        super().__init__()

        self.input_dim = input_dim
        self.degree = degree
        self.output_dim = input_dim + input_dim * 2 * degree

    def __repr__(self):
        return f"FreqEncoder: input_dim={self.input_dim} degree={self.degree} output_dim={self.output_dim}"

    def forward(self, inputs, **kwargs):
        # inputs: [..., input_dim]
        # return: [..., ]

        prefix_shape = list(inputs.shape[:-1])
        inputs = inputs.reshape(-1, self.input_dim)

        outputs = freq_encode(inputs, self.degree, self.output_dim)

        outputs = outputs.reshape(prefix_shape + [self.output_dim])

        return outputs
