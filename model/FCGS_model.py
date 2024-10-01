import os.path
import time

import torch
import torch.nn as nn
import torch.nn.functional as nnf
import math
from model.entropy_models import (Entropy_gaussian, Entropy_gaussian_mix_prob_3,
                                  Entropy_gaussian_mix_prob_2,
                                  Entropy_factorized,
                                  )
from model.grid_utils import normalize_xyz, _grid_creater, _grid_encoder, FreqEncoder
from model.gpcc_utils import sorted_voxels, sorted_orig_voxels, compress_gaussian_params, decompress_gaussian_params
from model.encodings_cuda import (STE_multistep, encoder, decoder,
                             encoder_gaussian_mixed_chunk, decoder_gaussian_mixed_chunk,
                             encoder_factorized_chunk, decoder_factorized_chunk,
                             )

b2M = 8*1024*1024

def get_time():
    torch.cuda.synchronize()
    return time.time()

class Channel_CTX_fea(nn.Module):
    def __init__(self):
        super().__init__()
        self.mean_d0 = nn.Parameter(torch.zeros(size=[1, 64]))
        self.scale_d0 = nn.Parameter(torch.zeros(size=[1, 64]))
        self.prob_d0 = nn.Parameter(torch.zeros(size=[1, 64]))
        self.MLP_d0 = nn.Sequential(
            nn.Linear(64, 64*3),
            nn.LeakyReLU(inplace=True),
            nn.Linear(64*3, 64*3),
            nn.LeakyReLU(inplace=True),
            nn.Linear(64*3, 64*3),
        )
        self.MLP_d1 = nn.Sequential(
            nn.Linear(64*2, 64*3),
            nn.LeakyReLU(inplace=True),
            nn.Linear(64*3, 64*3),
            nn.LeakyReLU(inplace=True),
            nn.Linear(64*3, 64*3),
        )
        self.MLP_d2 = nn.Sequential(
            nn.Linear(64*3, 64*3),
            nn.LeakyReLU(inplace=True),
            nn.Linear(64*3, 64*3),
            nn.LeakyReLU(inplace=True),
            nn.Linear(64*3, 64*3),
        )

    def forward(self, fea_q, to_dec=-1):
        # fea_q: [N, 256]
        NN = fea_q.shape[0]
        d0, d1, d2, d3 = torch.split(fea_q, split_size_or_sections=[64, 64, 64, 64], dim=-1)
        # mean_d0, scale_d0, prob_d0 = torch.zeros_like(d0), torch.zeros_like(d0), torch.zeros_like(d0)  # [N, 64] * 3
        mean_d0, scale_d0, prob_d0 = self.mean_d0.repeat(NN, 1), self.scale_d0.repeat(NN, 1), self.prob_d0.repeat(NN, 1)
        mean_d1, scale_d1, prob_d1 = torch.chunk(self.MLP_d0(d0), chunks=3, dim=-1)  # [N, 64*3] -> [N, 64] * 3
        mean_d2, scale_d2, prob_d2 = torch.chunk(self.MLP_d1(torch.cat([d0, d1], dim=-1)), chunks=3, dim=-1)  # [N, 64*3] -> [N, 64] * 3
        mean_d3, scale_d3, prob_d3 = torch.chunk(self.MLP_d2(torch.cat([d0, d1, d2], dim=-1)), chunks=3, dim=-1)  # [N, 64*3] -> [N, 64] * 3
        mean = torch.cat([mean_d0, mean_d1, mean_d2, mean_d3], dim=-1)
        scale = torch.cat([scale_d0, scale_d1, scale_d2, scale_d3], dim=-1)
        prob = torch.cat([prob_d0, prob_d1, prob_d2, prob_d3], dim=-1)
        if to_dec == 0:
            return mean_d0, scale_d0, prob_d0
        if to_dec == 1:
            return mean_d1, scale_d1, prob_d1
        if to_dec == 2:
            return mean_d2, scale_d2, prob_d2
        if to_dec == 3:
            return mean_d3, scale_d3, prob_d3
        return mean, scale, prob  # [N, 64*4], [N, 64*4], [N, 64*4]

class Channel_CTX_feq(nn.Module):
    def __init__(self):
        super().__init__()
        # assume 3
        # channel-wise context: (max_sh_degree + 1) ** 2
        # 0: 1 R/G/B.   1
        # 1: 4          3
        # 2: 9          5
        # 3: 16         7
        self.mean_d0 = nn.Parameter(torch.zeros(size=[1, 16]))
        self.scale_d0 = nn.Parameter(torch.zeros(size=[1, 16]))
        self.prob_d0 = nn.Parameter(torch.zeros(size=[1, 16]))
        self.MLP_d0 = nn.Sequential(
            nn.Linear(16, 16*3),
            nn.LeakyReLU(inplace=True),
            nn.Linear(16*3, 16*3),
            nn.LeakyReLU(inplace=True),
            nn.Linear(16*3, 16*3),
        )
        self.MLP_d1 = nn.Sequential(
            nn.Linear(16*2, 16*3),
            nn.LeakyReLU(inplace=True),
            nn.Linear(16*3, 16*3),
            nn.LeakyReLU(inplace=True),
            nn.Linear(16*3, 16*3),
        )

    def forward(self, shs_q, to_dec=-1):
        # shs_q: [N, 48]
        NN = shs_q.shape[0]
        shs_q = shs_q.view(NN, 16, 3)
        d0, d1, d2 = shs_q[..., 0], shs_q[..., 1], shs_q[..., 2]  # [N, 16], [N, 16], [N, 16]

        # mean_d0, scale_d0, prob_d0 = torch.zeros_like(d0), torch.zeros_like(d0), torch.zeros_like(d0)
        mean_d0, scale_d0, prob_d0 = self.mean_d0.repeat(NN, 1), self.scale_d0.repeat(NN, 1), self.prob_d0.repeat(NN, 1)
        mean_d1, scale_d1, prob_d1 = torch.chunk(self.MLP_d0(d0), chunks=3, dim=-1)  # [N, 16] * 3
        mean_d2, scale_d2, prob_d2 = torch.chunk(self.MLP_d1(torch.cat([d0, d1], dim=-1)), chunks=3, dim=-1)  # [N, 16] * 3
        mean = torch.stack([mean_d0, mean_d1, mean_d2], dim=-1).view(NN, -1)  # [N, 16, 3] ->[N, 48]
        scale = torch.stack([scale_d0, scale_d1, scale_d2], dim=-1).view(NN, -1)  # [N, 16, 3] ->[N, 48]
        prob = torch.stack([prob_d0, prob_d1, prob_d2], dim=-1).view(NN, -1)  # [N, 16, 3] ->[N, 48]
        if to_dec == 0:
            return mean_d0, scale_d0, prob_d0
        if to_dec == 1:
            return mean_d1, scale_d1, prob_d1
        if to_dec == 2:
            return mean_d2, scale_d2, prob_d2
        return mean, scale, prob

class Spatial_CTX(nn.Module):
    def __init__(self, reso_3D, off_3D, reso_2D, off_2D):
        super().__init__()
        self.reso_3D = reso_3D
        self.off_3D = off_3D
        self.reso_2D = reso_2D
        self.off_2D = off_2D
    def forward(self, xyz_for_creater, xyz_for_interp, feature, determ=False, return_all=False):
        assert xyz_for_creater.shape[0] == feature.shape[0]
        grid_3D = _grid_creater.apply(xyz_for_creater, feature, self.reso_3D, self.off_3D, determ)  # [offsets_list_3D[-1], 48]
        grid_xy = _grid_creater.apply(xyz_for_creater[:, 0:2], feature, self.reso_2D, self.off_2D, determ)  # [offsets_list[-1], 48]
        grid_xz = _grid_creater.apply(xyz_for_creater[:, 0::2], feature, self.reso_2D, self.off_2D, determ)  # [offsets_list[-1], 48]
        grid_yz = _grid_creater.apply(xyz_for_creater[:, 1:3], feature, self.reso_2D, self.off_2D, determ)  # [offsets_list[-1], 48]

        context_info_3D = _grid_encoder.apply(xyz_for_interp, grid_3D, self.off_3D, self.reso_3D)  # [N_choose, 48*n_levels]
        context_info_xy = _grid_encoder.apply(xyz_for_interp[:, 0:2], grid_xy, self.off_2D, self.reso_2D)  # [N_choose, 48*n_levels]
        context_info_xz = _grid_encoder.apply(xyz_for_interp[:, 0::2], grid_xz, self.off_2D, self.reso_2D)  # [N_choose, 48*n_levels]
        context_info_yz = _grid_encoder.apply(xyz_for_interp[:, 1:3], grid_yz, self.off_2D, self.reso_2D)  # [N_choose, 48*n_levels]

        context_info = torch.cat([context_info_3D, context_info_xy, context_info_xz, context_info_yz], dim=-1)  # [N_choose, 48*n_levels*4]
        if return_all:
            return context_info, (xyz_for_creater, xyz_for_interp, feature, grid_3D, grid_xy, grid_xz, grid_yz, context_info_3D, context_info_xy, context_info_xz, context_info_yz, self.reso_3D, self.off_3D)
        return context_info

class FCGS(nn.Module):
    def __init__(self,
                 fea_dim=56, hidden=256, lat_dim=256, grid_dim=48,
                 Q=1, Q_fe=0.001, Q_op=0.001, Q_sc=0.01, Q_ro=0.00001,
                 resolutions_list=[300, 400, 500],
                 resolutions_list_3D=[60, 80, 100],
                 num_dim=3,
                 norm_radius=3,
                 binary=0,
                 ):
        super().__init__()
        self.norm_radius = norm_radius
        self.binary = binary
        self.freq_enc = FreqEncoder(3, 4)
        assert len(resolutions_list) == len(resolutions_list_3D)
        n_levels = len(resolutions_list)
        #

        self.Encoder_mask = nn.Sequential(
            nn.Linear(fea_dim, hidden),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )
        self.Encoder_fea = nn.Sequential(
            nn.Linear(fea_dim, hidden),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden, lat_dim),
        )

        self.Decoder_fea = nn.Sequential(
            nn.Linear(lat_dim, hidden),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden, hidden),
        )
        self.head_f_dc = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden, 3)
        )
        nn.init.constant_(self.head_f_dc[-1].weight, 0)
        nn.init.constant_(self.head_f_dc[-1].bias, 0)
        self.head_f_rst = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden, 45)
        )
        nn.init.constant_(self.head_f_rst[-1].weight, 0)
        nn.init.constant_(self.head_f_rst[-1].bias, 0)

        self.latdim_2_griddim_fea = nn.Sequential(
            nn.Linear(lat_dim, grid_dim),
        )

        self.resolutions_list, self.offsets_list = self.get_offsets(resolutions_list, dim=2)
        self.resolutions_list_3D, self.offsets_list_3D = self.get_offsets(resolutions_list_3D, dim=3)

        self.cafea_indim = 48*(len(resolutions_list)*3+len(resolutions_list_3D)) + self.freq_enc.output_dim
        self.context_analyzer_fea = nn.Sequential(
            nn.Linear(self.cafea_indim, 256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 256*3),  # [1 mean, 1 scale, 2 prob]
        )
        self.cafeq_indim = 48*(len(resolutions_list)*3+len(resolutions_list_3D)) + self.freq_enc.output_dim
        self.context_analyzer_feq = nn.Sequential(
            nn.Linear(self.cafeq_indim, 48*12),
            nn.LeakyReLU(inplace=True),
            nn.Linear(48*12, 48*12),
            nn.LeakyReLU(inplace=True),
            nn.Linear(48*12, 48*3),  # [1 mean, 1 scale, 2 prob]
        )
        self.cageo_indim = 8*(len(resolutions_list)*3+len(resolutions_list_3D)) + self.freq_enc.output_dim
        self.context_analyzer_geo = nn.Sequential(
            nn.Linear(self.cageo_indim, 8*12),
            nn.LeakyReLU(inplace=True),
            nn.Linear(8*12, 8*12),
            nn.LeakyReLU(inplace=True),
            nn.Linear(8*12, 8*3),  # [1 mean, 1 scale, 2 prob]
        )

        self.feq_channel_ctx = Channel_CTX_feq()
        self.fea_channel_ctx = Channel_CTX_fea()
        self.feq_spatial_ctx = Spatial_CTX(
            self.resolutions_list_3D,
            self.offsets_list_3D,
            self.resolutions_list,
            self.offsets_list,
        )

        self.Encoder_fea_hyp = nn.Sequential(
            nn.Linear(lat_dim, lat_dim//2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(lat_dim//2, lat_dim//2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(lat_dim//2, lat_dim//4),
        )
        self.Decoder_fea_hyp = nn.Sequential(
            nn.Linear(lat_dim//4, lat_dim//2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(lat_dim//2, lat_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(lat_dim, lat_dim*3),
        )

        self.Encoder_feq_hyp = nn.Sequential(
            nn.Linear(48, 48//2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(48//2, 48//2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(48//2, 48//2),
        )
        self.Decoder_feq_hyp = nn.Sequential(
            nn.Linear(48//2, 48//2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(48//2, 48),
            nn.LeakyReLU(inplace=True),
            nn.Linear(48, 48*3),
        )

        self.Encoder_geo_hyp = nn.Sequential(
            nn.Linear(8, 16),
            nn.LeakyReLU(inplace=True),
            nn.Linear(16, 16),
            nn.LeakyReLU(inplace=True),
            nn.Linear(16, 16),
        )
        self.Decoder_geo_hyp = nn.Sequential(
            nn.Linear(16, 16),
            nn.LeakyReLU(inplace=True),
            nn.Linear(16, 16),
            nn.LeakyReLU(inplace=True),
            nn.Linear(16, 8*3),
        )

        self.Q = Q
        self.Q_fe = Q_fe
        self.Q_op = Q_op
        self.Q_sc = Q_sc
        self.Q_ro = Q_ro

        self.EF_fea = Entropy_factorized(lat_dim//4, Q=Q)
        self.EF_feq = Entropy_factorized(48//2, Q=Q)
        self.EF_geo = Entropy_factorized(16, Q=Q)
        self.EF_op = Entropy_factorized(1, Q=Q)
        self.EF_sc = Entropy_factorized(3, Q=Q)
        self.EF_ro = Entropy_factorized(4, Q=Q)
        self.EG = Entropy_gaussian(Q=Q)
        self.EG_mix_prob_2 = Entropy_gaussian_mix_prob_2(Q=Q)
        self.EG_mix_prob_3 = Entropy_gaussian_mix_prob_3(Q=Q)

        self.ad_fe = nn.Parameter(torch.tensor(data=[1.0, 0.0, 0.0]).unsqueeze(0))  # mul, add, tanh # [1, 3]
        self.ad_op = nn.Parameter(torch.tensor(data=[1.0, 0.0, 0.0]).unsqueeze(0))  # mul, add, tanh # [1, 3]
        self.ad_sc = nn.Parameter(torch.tensor(data=[1.0, 0.0, 0.0]).unsqueeze(0))  # mul, add, tanh # [1, 3]
        self.ad_ro = nn.Parameter(torch.tensor(data=[1.0, 0.0, 0.0]).unsqueeze(0))  # mul, add, tanh # [1, 3]

    def get_offsets(self, resolutions_list, dim=3):
        offsets_list = [0]
        offsets = 0
        for resolution in resolutions_list:
            offset = resolution ** dim
            offsets_list.append(offsets + offset)
            offsets += offset
        offsets_list = torch.tensor(offsets_list, device='cuda', dtype=torch.int)
        resolutions_list = torch.tensor(resolutions_list, device='cuda', dtype=torch.int)
        return resolutions_list, offsets_list

    def clamp(self, x, Q):
        x_mean = x.mean().detach()
        x_min = x_mean - 15_000 * Q
        x_max = x_mean + 15_000 * Q
        x = torch.clamp(x, min=x_min.detach(), max=x_max.detach())
        return x

    def quantize(self, x, Q, testing):
        if not testing:
            x_q = x + torch.empty_like(x).uniform_(-0.5, 0.5) * Q
        else:
            x_q = STE_multistep.apply(x, Q)
        x_q = self.clamp(x_q, Q)
        return x_q

    def compress_only(self, g_xyz, g_fea, means=None, stds=None, testing=True, root_path='./', chunk_size_list=(), feqonly=False, random_seed=1):
        c_size_fea, c_size_feq, c_size_geo = chunk_size_list
        g_xyz, g_fea = sorted_orig_voxels(g_xyz, g_fea)  # to morton order
        # doing compression
        bits_xyz = compress_gaussian_params(
            gaussian_params=g_xyz,
            bin_path=os.path.join(root_path, 'xyz_gpcc.bin')
        )[-1]['file_size']['total'] * 8 * 1024 * 1024
        torch.manual_seed(random_seed)
        shuffled_indices = torch.randperm(g_xyz.size(0))
        g_xyz = g_xyz[shuffled_indices]  # [N_g, 3]
        g_fea = g_fea[shuffled_indices]  # [N_g, 56]
        fe, op, sc, ro = torch.split(g_fea, split_size_or_sections=[3 + 45, 1, 3, 4], dim=-1)  # [N_g, x] for each
        norm_xyz, norm_xyz_clamp, mask_xyz = normalize_xyz(g_xyz, K=self.norm_radius, means=means, stds=stds)   # [N_g, 3]
        freq_enc_xyz = self.freq_enc(norm_xyz_clamp)  # [N_g, freq_output]
        N_g = g_xyz.shape[0]  # N_g
        mask_sig = self.Encoder_mask(g_fea)  # [N_g, 1]
        mask = ((mask_sig > 0.01).float() - mask_sig).detach() + mask_sig  # [N_g, 1]
        mask_fea = mask.detach()[:, 0].to(torch.bool)  # [N_g]
        mask_feq = torch.logical_not(mask_fea)  # [N_g]

        # used for shape0=N_g
        s0 = 0
        s1 = N_g // 6 * 1
        s2 = N_g // 6 * 2
        s3 = N_g // 6 * 4
        s4 = N_g
        sn = [s0, s1, s2, s3, s4]

        # used for shape0=N_fea
        k0 = 0
        k1 = int(mask_fea[s0:s1].sum().item())
        k2 = int(mask_fea[s0:s2].sum().item())
        k3 = int(mask_fea[s0:s3].sum().item())
        k4 = int(mask_fea[s0:s4].sum().item())
        kn = [k0, k1, k2, k3, k4]

        # used for shape0=N_feq
        t0 = 0
        t1 = int(mask_feq[s0:s1].sum().item())
        t2 = int(mask_feq[s0:s2].sum().item())
        t3 = int(mask_feq[s0:s3].sum().item())
        t4 = int(mask_feq[s0:s4].sum().item())
        tn = [t0, t1, t2, t3, t4]

        flag_remask = False  # if shape len = 0, then do remask
        if k1==k0 or k2==k1 or k3==k2 or k4==k3:
            mask = torch.zeros_like(mask)
            flag_remask = True
        elif t1==t0 or t2==t1 or t3==t2 or t4==t3:
            mask = torch.ones_like(mask)
            flag_remask = True
        if feqonly:
            mask = torch.zeros_like(mask)
            flag_remask = True
        if flag_remask:
            mask_fea = mask.detach()[:, 0].to(torch.bool)  # [N_g]
            mask_feq = torch.logical_not(mask_fea)  # [N_g]
            # used for shape0=N_fea
            k0 = 0
            k1 = int(mask_fea[s0:s1].sum().item())
            k2 = int(mask_fea[s0:s2].sum().item())
            k3 = int(mask_fea[s0:s3].sum().item())
            k4 = int(mask_fea[s0:s4].sum().item())
            kn = [k0, k1, k2, k3, k4]
            # used for shape0=N_feq
            t0 = 0
            t1 = int(mask_feq[s0:s1].sum().item())
            t2 = int(mask_feq[s0:s2].sum().item())
            t3 = int(mask_feq[s0:s3].sum().item())
            t4 = int(mask_feq[s0:s4].sum().item())
            tn = [t0, t1, t2, t3, t4]

        bits_mask = encoder(
            x=mask,
            file_name=os.path.join(root_path, 'mask.b')
        )

        g_fea_enc = self.Encoder_fea(g_fea[mask_fea])  # [N_fea, 256]
        g_fea_enc_q = self.quantize(g_fea_enc, self.Q, testing)  # [N_fea, 256]
        g_fea_out = self.Decoder_fea(g_fea_enc_q)  # [N_fea, 256]
        fe_dec = torch.cat([self.head_f_dc(g_fea_out), self.head_f_rst(g_fea_out)], dim=-1)  # [N_fea, 48]

        Q_fe = (self.Q_fe * self.ad_fe[:, 0:1] + self.ad_fe[:, 1:2]) * (1 + torch.tanh(self.ad_fe[:, 2:3]))  # [1, 1]
        Q_op = (self.Q_op * self.ad_op[:, 0:1] + self.ad_op[:, 1:2]) * (1 + torch.tanh(self.ad_op[:, 2:3]))  # [1, 1]
        Q_sc = (self.Q_sc * self.ad_sc[:, 0:1] + self.ad_sc[:, 1:2]) * (1 + torch.tanh(self.ad_sc[:, 2:3]))  # [1, 1]
        Q_ro = (self.Q_ro * self.ad_ro[:, 0:1] + self.ad_ro[:, 1:2]) * (1 + torch.tanh(self.ad_ro[:, 2:3]))  # [1, 1]
        Q_fe = Q_fe.repeat(mask_feq.sum(), 48)  # [N_feq, 48]
        Q_op = Q_op.repeat(N_g, 1)  # [N_g, 1]
        Q_sc = Q_sc.repeat(N_g, 3)  # [N_g, 3]
        Q_ro = Q_ro.repeat(N_g, 4)  # [N_g, 4]
        fe_q = self.quantize(fe[mask_feq], Q_fe, testing)  # [N_feq, 48]
        op_q = self.quantize(op, Q_op, testing)  # [N_g, 1]
        sc_q = self.quantize(sc, Q_sc, testing)  # [N_g, 3]
        ro_q = self.quantize(ro, Q_ro, testing)  # [N_g, 4]
        fe_final = torch.zeros([N_g, 48], dtype=torch.float32, device='cuda')  # [N_g, 48]
        fe_final[mask_fea] = fe_dec  # [N_g, 48]
        fe_final[mask_feq] = fe_q  # [N_g, 48]

        geo_q = torch.cat([op_q, sc_q, ro_q], dim=-1)  # [N_g, 8]
        Q_geo = torch.cat([Q_op, Q_sc, Q_ro], dim=-1)  # [N_g, 8]

        #-----#

        if mask_fea.sum() > 0:
            print('Start compressing fea...')

            g_fea_enc_q_hyp = self.Encoder_fea_hyp(g_fea_enc_q)  # [N_fea, 64]
            g_fea_enc_q_hyp_q = self.quantize(g_fea_enc_q_hyp, self.Q, testing)  # [N_fea, 64]
            fea_grid_feature = self.latdim_2_griddim_fea(g_fea_enc_q)  # [N_fea, 48]
            # norm_xyz_clamp: [N_g, 3], mask_fea: [N_g], fea_grid_feature: [N_fea, 48], norm_xyz_clamp[s1:s2][mask_fea[s1:s2]]: [k2-k1, 3]
            ctx_s2 = self.feq_spatial_ctx.forward(norm_xyz_clamp[s0:s1][mask_fea[s0:s1]], norm_xyz_clamp[s1:s2][mask_fea[s1:s2]], fea_grid_feature[k0:k1])  # [k2-k1, dim]
            ctx_s3 = self.feq_spatial_ctx.forward(norm_xyz_clamp[s0:s2][mask_fea[s0:s2]], norm_xyz_clamp[s2:s3][mask_fea[s2:s3]], fea_grid_feature[k0:k2])  # [k3-k2, dim]
            ctx_s4 = self.feq_spatial_ctx.forward(norm_xyz_clamp[s0:s3][mask_fea[s0:s3]], norm_xyz_clamp[s3:s4][mask_fea[s3:s4]], fea_grid_feature[k0:k3])  # [k4-k3, dim]
            ctx_s1 = torch.zeros(size=[k1-k0, ctx_s2.shape[-1]], dtype=ctx_s2.dtype, device=ctx_s2.device)  # [k1-k0, dim]
            ctx_s1234 = torch.cat([torch.cat([ctx_s1, ctx_s2, ctx_s3, ctx_s4], dim=0), freq_enc_xyz[mask_fea]], dim=-1)  # [N_fea, dim]
            mean_sp, scale_sp, prob_sp = torch.split(self.context_analyzer_fea(ctx_s1234), split_size_or_sections=[256, 256, 256], dim=-1)  # [N_fea, 256] for each

            mean_hp, scale_hp, prob_hp = torch.split(self.Decoder_fea_hyp(g_fea_enc_q_hyp_q), split_size_or_sections=[256, 256, 256], dim=-1)  # [N_fea, 256] for each
            mean_ch, scale_ch, prob_ch = self.fea_channel_ctx.forward(g_fea_enc_q)  # [N_fea, 256] for each
            probs = torch.stack([prob_sp, prob_hp, prob_ch], dim=-1)  # [N_fea, 256, 3]
            probs = torch.softmax(probs, dim=-1)  # [N_fea, 256, 3]
            prob_sp, prob_hp, prob_ch = probs[..., 0], probs[..., 1], probs[..., 2]  # [N_fea, 256] for each

            bits_fea_hyp = encoder_factorized_chunk(
                x=g_fea_enc_q_hyp_q,  # [N_fea, 64]
                lower_func=self.EF_fea._logits_cumulative,
                Q=self.Q,
                file_name=os.path.join(root_path, 'g_fea_enc_q_hyp_q.b')
            )

            bits_fea_main = 0
            for l_sp in range(4):
                k_st = kn[l_sp]
                k_ed = kn[l_sp+1]
                for l_ch in range(4):
                    c_st = l_ch*64
                    c_ed = l_ch*64+64
                    # from: mean_sp: [N_fea, 256], scale_sp: [N_fea, 256], prob_sp: [N_fea, 256]
                    # to: mean_sp_l: [k_len, 64], scale_sp_l: [k_len, 64], prob_sp_l: [k_len, 64]
                    mean_sp_l, scale_sp_l, prob_sp_l = mean_sp[k_st:k_ed, c_st:c_ed], scale_sp[k_st:k_ed, c_st:c_ed], prob_sp[k_st:k_ed, c_st:c_ed]
                    mean_hp_l, scale_hp_l, prob_hp_l = mean_hp[k_st:k_ed, c_st:c_ed], scale_hp[k_st:k_ed, c_st:c_ed], prob_hp[k_st:k_ed, c_st:c_ed]
                    mean_ch_l, scale_ch_l, prob_ch_l = mean_ch[k_st:k_ed, c_st:c_ed], scale_ch[k_st:k_ed, c_st:c_ed], prob_ch[k_st:k_ed, c_st:c_ed]
                    bits_fea_main_tmp = encoder_gaussian_mixed_chunk(
                        x=g_fea_enc_q[k_st:k_ed, c_st:c_ed].contiguous().view(-1),  # from: [N_fea, 256]
                        mean_list=[mean_sp_l.contiguous().view(-1), mean_hp_l.contiguous().view(-1), mean_ch_l.contiguous().view(-1)],
                        scale_list=[scale_sp_l.contiguous().view(-1), scale_hp_l.contiguous().view(-1), scale_ch_l.contiguous().view(-1)],
                        prob_list=[prob_sp_l.contiguous().view(-1), prob_hp_l.contiguous().view(-1), prob_ch_l.contiguous().view(-1)],
                        Q=self.Q,
                        file_name=os.path.join(root_path, f'g_fea_enc_q_sp{l_sp}_ch{l_ch}.b'),
                        chunk_size=c_size_fea,
                    )
                    bits_fea_main += bits_fea_main_tmp
            bits_fea = bits_fea_main + bits_fea_hyp
        else:
            bits_fea_main = 0
            bits_fea_hyp = 0
            bits_fea = 0


        if mask_feq.sum() > 0:
            print('Start compressing feq...')

            # ---------
            fe_q_hyp = self.Encoder_feq_hyp(fe_q)  # [N_feq, 24]
            fe_q_hyp_q = self.quantize(fe_q_hyp, self.Q, testing)  # [N_feq, 24]

            # norm_xyz_clamp: [N_g, 3], mask_feq: [N_g], fe_final: [N_g, 48], norm_xyz_clamp[s1:s2][mask_feq[s1:s2]]: [t2-t1, 3]
            ctx_s2 = self.feq_spatial_ctx.forward(norm_xyz_clamp[s0:s1], norm_xyz_clamp[s1:s2][mask_feq[s1:s2]], fe_final[s0:s1])  # [t2-t1, dim]
            ctx_s3 = self.feq_spatial_ctx.forward(norm_xyz_clamp[s0:s2], norm_xyz_clamp[s2:s3][mask_feq[s2:s3]], fe_final[s0:s2])  # [t3-t2, dim]
            ctx_s4 = self.feq_spatial_ctx.forward(norm_xyz_clamp[s0:s3], norm_xyz_clamp[s3:s4][mask_feq[s3:s4]], fe_final[s0:s3])  # [t4-t3, dim]
            ctx_s1 = torch.zeros(size=[mask_feq[s0:s1].sum(), ctx_s2.shape[-1]], dtype=ctx_s2.dtype, device=ctx_s2.device)  # [t1-t0, dim]
            ctx_s1234 = torch.cat([torch.cat([ctx_s1, ctx_s2, ctx_s3, ctx_s4], dim=0), freq_enc_xyz[mask_feq]], dim=-1)  # [N_feq, dim]
            mean_sp, scale_sp, prob_sp = torch.split(self.context_analyzer_feq(ctx_s1234), split_size_or_sections=[48, 48, 48], dim=-1)  # [N_feq, 48] for each


            mean_hp, scale_hp, prob_hp = torch.split(self.Decoder_feq_hyp(fe_q_hyp_q), split_size_or_sections=[48, 48, 48], dim=-1)  # [N_feq, 48] for each
            mean_ch, scale_ch, prob_ch = self.feq_channel_ctx.forward(fe_q)  # [N_feq, 48] for each
            probs = torch.stack([prob_sp, prob_hp, prob_ch], dim=-1)  # [N_feq, 48, 3]
            probs = torch.softmax(probs, dim=-1)  # [N_feq, 48, 3]
            prob_sp, prob_hp, prob_ch = probs[..., 0], probs[..., 1], probs[..., 2]  # [N_feq, 48] for each

            bits_feq_hyp = encoder_factorized_chunk(
                x=fe_q_hyp_q,  # [N_feq, 24]
                lower_func=self.EF_feq._logits_cumulative,
                Q=self.Q,
                file_name=os.path.join(root_path, 'fe_q_hyp_q.b')
            )

            bits_feq_main = 0
            for l_sp in range(4):
                t_st = tn[l_sp]
                t_ed = tn[l_sp+1]
                for l_ch in range(3):
                    # from: mean_sp: [N_feq, 48], scale_sp: [N_feq, 48], prob_sp: [N_feq, 48]
                    # to: mean_sp_l: [t_len, 16], scale_sp_l: [t_len, 16], prob_sp_l: [t_len, 16]
                    mean_sp_l, scale_sp_l, prob_sp_l = mean_sp[t_st:t_ed, l_ch::3], scale_sp[t_st:t_ed, l_ch::3], prob_sp[t_st:t_ed, l_ch::3]
                    mean_hp_l, scale_hp_l, prob_hp_l = mean_hp[t_st:t_ed, l_ch::3], scale_hp[t_st:t_ed, l_ch::3], prob_hp[t_st:t_ed, l_ch::3]
                    mean_ch_l, scale_ch_l, prob_ch_l = mean_ch[t_st:t_ed, l_ch::3], scale_ch[t_st:t_ed, l_ch::3], prob_ch[t_st:t_ed, l_ch::3]
                    bits_feq_main_tmp = encoder_gaussian_mixed_chunk(
                        x=fe_q[t_st:t_ed, l_ch::3].contiguous().view(-1),  # from: [N_feq, 48]
                        mean_list=[mean_sp_l.contiguous().view(-1), mean_hp_l.contiguous().view(-1), mean_ch_l.contiguous().view(-1)],
                        scale_list=[scale_sp_l.contiguous().view(-1), scale_hp_l.contiguous().view(-1), scale_ch_l.contiguous().view(-1)],
                        prob_list=[prob_sp_l.contiguous().view(-1), prob_hp_l.contiguous().view(-1), prob_ch_l.contiguous().view(-1)],
                        Q=Q_fe[t_st:t_ed, l_ch::3].contiguous().view(-1),  # from: [N_feq, 48]
                        file_name=os.path.join(root_path, f'fe_q_sp{l_sp}_ch{l_ch}.b'),
                        chunk_size=c_size_feq,
                    )
                    bits_feq_main += bits_feq_main_tmp
            bits_feq = bits_feq_main + bits_feq_hyp
        else:
            bits_feq_main = 0
            bits_feq_hyp = 0
            bits_feq = 0


        print('Start compressing geo...')

        # ---------
        geo_q_hyp = self.Encoder_geo_hyp(geo_q)  # [N_g, 8]
        geo_q_hyp_q = self.quantize(geo_q_hyp, self.Q, testing)  # [N_g, 8]

        # norm_xyz_clamp: [N_g, 3], geo_q: [N_g, 8],  norm_xyz_clamp[s1:s2]: [s2-s1, 3]
        ctx_s2 = self.feq_spatial_ctx.forward(norm_xyz_clamp[s0:s1], norm_xyz_clamp[s1:s2], geo_q[s0:s1])  # [s2-s1, dim]
        ctx_s3 = self.feq_spatial_ctx.forward(norm_xyz_clamp[s0:s2], norm_xyz_clamp[s2:s3], geo_q[s0:s2])  # [s3-s2, dim]
        ctx_s4 = self.feq_spatial_ctx.forward(norm_xyz_clamp[s0:s3], norm_xyz_clamp[s3:s4], geo_q[s0:s3])  # [s4-s3, dim]
        ctx_s1 = torch.zeros(size=[s1-s0, ctx_s2.shape[-1]], dtype=ctx_s2.dtype, device=ctx_s2.device)  # [s1-s0, dim]
        ctx_s1234 = torch.cat([torch.cat([ctx_s1, ctx_s2, ctx_s3, ctx_s4], dim=0), freq_enc_xyz], dim=-1)  # [N_g, dim]

        mean_sp, scale_sp, prob_sp = torch.split(self.context_analyzer_geo(ctx_s1234), split_size_or_sections=[8, 8, 8], dim=-1)  # [N_g, 8] for each
        mean_hp, scale_hp, prob_hp = torch.split(self.Decoder_geo_hyp(geo_q_hyp_q), split_size_or_sections=[8, 8, 8], dim=-1)  # [N_g, 8] for each
        probs = torch.stack([prob_sp, prob_hp], dim=-1)  # [N_g, 8, 2]
        probs = torch.softmax(probs, dim=-1)  # [N_g, 8, 2]
        prob_sp, prob_hp = probs[..., 0], probs[..., 1]  # [N_g, 8] for each

        bits_geo_hyp = encoder_factorized_chunk(
            x=geo_q_hyp_q,  # [N_g, 8]
            lower_func=self.EF_geo._logits_cumulative,
            Q=self.Q,
            file_name=os.path.join(root_path, 'geo_q_hyp_q.b')
        )

        bits_geo_main = 0
        for l_sp in range(4):
            s_st = sn[l_sp]
            s_ed = sn[l_sp+1]
            # from: mean_sp: [N_g, 8], scale_sp: [N_g, 8], prob_sp: [N_g, 8]
            # to: mean_sp_l: [s_len, 8], scale_sp_l: [s_len, 8], prob_sp_l: [s_len, 8]
            mean_sp_l, scale_sp_l, prob_sp_l = mean_sp[s_st:s_ed], scale_sp[s_st:s_ed], prob_sp[s_st:s_ed]
            mean_hp_l, scale_hp_l, prob_hp_l = mean_hp[s_st:s_ed], scale_hp[s_st:s_ed], prob_hp[s_st:s_ed]
            bits_geo_main_tmp = encoder_gaussian_mixed_chunk(
                x=geo_q[s_st:s_ed].contiguous().view(-1),  # from: [N_g, 8]
                mean_list=[mean_sp_l.contiguous().view(-1), mean_hp_l.contiguous().view(-1)],
                scale_list=[scale_sp_l.contiguous().view(-1), scale_hp_l.contiguous().view(-1)],
                prob_list=[prob_sp_l.contiguous().view(-1), prob_hp_l.contiguous().view(-1)],
                Q=Q_geo[s_st:s_ed].contiguous().view(-1),  # from: [N_g, 8]
                file_name=os.path.join(root_path, f'geo_q_sp{l_sp}.b'),
                chunk_size=c_size_geo,
            )
            bits_geo_main += bits_geo_main_tmp
        bits_geo = bits_geo_main + bits_geo_hyp

        bits = bits_fea + bits_feq + bits_geo + bits_xyz + bits_mask

        g_fea_out = torch.cat([fe_final, geo_q], dim=-1)  # [N_choose, 48+1+3+4]

        return (g_xyz, g_fea_out, mask,
                bits/b2M, bits_xyz/b2M, bits_mask/b2M,
                bits_fea/b2M, bits_fea_main/b2M, bits_fea_hyp/b2M,
                bits_feq/b2M, bits_feq_main/b2M, bits_feq_hyp/b2M,
                bits_geo/b2M, bits_geo_main/b2M, bits_geo_hyp/b2M,
                )

    def compress(self, g_xyz, g_fea, means=None, stds=None, testing=True, root_path='./', chunk_size_list=(), determ_codec=False):
        c_size_fea, c_size_feq, c_size_geo = chunk_size_list
        g_xyz, g_fea = sorted_orig_voxels(g_xyz, g_fea)  # to morton order
        # doing compression
        bits_xyz = compress_gaussian_params(
            gaussian_params=g_xyz,
            bin_path=os.path.join(root_path, 'xyz_gpcc.bin')
        )[-1]['file_size']['total'] * 8 * 1024 * 1024
        torch.manual_seed(1)
        shuffled_indices = torch.randperm(g_xyz.size(0))
        g_xyz = g_xyz[shuffled_indices]  # [N_g, 3]
        g_fea = g_fea[shuffled_indices]  # [N_g, 56]
        fe, op, sc, ro = torch.split(g_fea, split_size_or_sections=[3 + 45, 1, 3, 4], dim=-1)  # [N_g, x] for each
        norm_xyz, norm_xyz_clamp, mask_xyz = normalize_xyz(g_xyz, K=self.norm_radius, means=means, stds=stds)   # [N_g, 3]
        freq_enc_xyz = self.freq_enc(norm_xyz_clamp)  # [N_g, freq_output]
        N_g = g_xyz.shape[0]  # N_g
        mask_sig = self.Encoder_mask(g_fea)  # [N_g, 1]
        mask = ((mask_sig > 0.01).float() - mask_sig).detach() + mask_sig  # [N_g, 1]
        bits_mask = encoder(
            x=mask,
            file_name=os.path.join(root_path, 'mask.b')
        )
        mask_fea = mask.detach()[:, 0].to(torch.bool)  # [N_g]
        mask_feq = torch.logical_not(mask_fea)  # [N_g]

        g_fea_enc = self.Encoder_fea(g_fea[mask_fea])  # [N_fea, 256]
        g_fea_enc_q = self.quantize(g_fea_enc, self.Q, testing)  # [N_fea, 256]

        g_fea_out = self.Decoder_fea(g_fea_enc_q)  # [N_fea, 256]
        fe_dec = torch.cat([self.head_f_dc(g_fea_out), self.head_f_rst(g_fea_out)], dim=-1)  # [N_fea, 48]

        Q_fe = (self.Q_fe * self.ad_fe[:, 0:1] + self.ad_fe[:, 1:2]) * (1 + torch.tanh(self.ad_fe[:, 2:3]))  # [1, 1]
        Q_op = (self.Q_op * self.ad_op[:, 0:1] + self.ad_op[:, 1:2]) * (1 + torch.tanh(self.ad_op[:, 2:3]))  # [1, 1]
        Q_sc = (self.Q_sc * self.ad_sc[:, 0:1] + self.ad_sc[:, 1:2]) * (1 + torch.tanh(self.ad_sc[:, 2:3]))  # [1, 1]
        Q_ro = (self.Q_ro * self.ad_ro[:, 0:1] + self.ad_ro[:, 1:2]) * (1 + torch.tanh(self.ad_ro[:, 2:3]))  # [1, 1]
        Q_fe = Q_fe.repeat(mask_feq.sum(), 48)  # [N_feq, 48]
        Q_op = Q_op.repeat(N_g, 1)  # [N_g, 1]
        Q_sc = Q_sc.repeat(N_g, 3)  # [N_g, 3]
        Q_ro = Q_ro.repeat(N_g, 4)  # [N_g, 4]
        fe_q = self.quantize(fe[mask_feq], Q_fe, testing)  # [N_feq, 48]
        op_q = self.quantize(op, Q_op, testing)  # [N_g, 1]
        sc_q = self.quantize(sc, Q_sc, testing)  # [N_g, 3]
        ro_q = self.quantize(ro, Q_ro, testing)  # [N_g, 4]
        fe_final = torch.zeros([N_g, 48], dtype=torch.float32, device='cuda')  # [N_g, 48]
        fe_final[mask_fea] = fe_dec  # [N_g, 48]
        fe_final[mask_feq] = fe_q  # [N_g, 48]

        geo_q = torch.cat([op_q, sc_q, ro_q], dim=-1)  # [N_g, 8]
        Q_geo = torch.cat([Q_op, Q_sc, Q_ro], dim=-1)  # [N_g, 8]

        # used for shape0=N_g
        s0 = 0
        s1 = N_g // 6 * 1
        s2 = N_g // 6 * 2
        s3 = N_g // 6 * 4
        s4 = N_g
        sn = [s0, s1, s2, s3, s4]

        # used for shape0=N_fea
        k0 = 0
        k1 = int(mask_fea[s0:s1].sum().item())
        k2 = int(mask_fea[s0:s2].sum().item())
        k3 = int(mask_fea[s0:s3].sum().item())
        k4 = int(mask_fea[s0:s4].sum().item())
        kn = [k0, k1, k2, k3, k4]

        # used for shape0=N_feq
        t0 = 0
        t1 = int(mask_feq[s0:s1].sum().item())
        t2 = int(mask_feq[s0:s2].sum().item())
        t3 = int(mask_feq[s0:s3].sum().item())
        t4 = int(mask_feq[s0:s4].sum().item())
        tn = [t0, t1, t2, t3, t4]

        #-----#
        g_fea_enc_q_hyp = self.Encoder_fea_hyp(g_fea_enc_q)  # [N_fea, 64]
        g_fea_enc_q_hyp_q = self.quantize(g_fea_enc_q_hyp, self.Q, testing)  # [N_fea, 64]
        fea_grid_feature = self.latdim_2_griddim_fea(g_fea_enc_q)  # [N_fea, 48]
        # norm_xyz_clamp: [N_g, 3], mask_fea: [N_g], fea_grid_feature: [N_fea, 48], norm_xyz_clamp[s1:s2][mask_fea[s1:s2]]: [k2-k1, 3]
        ctx_s2 = self.feq_spatial_ctx.forward(norm_xyz_clamp[s0:s1][mask_fea[s0:s1]], norm_xyz_clamp[s1:s2][mask_fea[s1:s2]], fea_grid_feature[k0:k1], determ=determ_codec)  # [k2-k1, dim]
        ctx_s3 = self.feq_spatial_ctx.forward(norm_xyz_clamp[s0:s2][mask_fea[s0:s2]], norm_xyz_clamp[s2:s3][mask_fea[s2:s3]], fea_grid_feature[k0:k2], determ=determ_codec)  # [k3-k2, dim]
        ctx_s4 = self.feq_spatial_ctx.forward(norm_xyz_clamp[s0:s3][mask_fea[s0:s3]], norm_xyz_clamp[s3:s4][mask_fea[s3:s4]], fea_grid_feature[k0:k3], determ=determ_codec)  # [k4-k3, dim]
        ctx_s1 = torch.zeros(size=[k1-k0, ctx_s2.shape[-1]], dtype=ctx_s2.dtype, device=ctx_s2.device)  # [k1-k0, dim]
        ctx_s1234 = torch.cat([torch.cat([ctx_s1, ctx_s2, ctx_s3, ctx_s4], dim=0), freq_enc_xyz[mask_fea]], dim=-1)  # [N_fea, dim]

        # mean_sp, scale_sp, prob_sp = torch.split(self.context_analyzer_fea(ctx_s1234), split_size_or_sections=[256, 256, 256], dim=-1)  # [N_fea, 256] for each
        mean_sp_list = []
        scale_sp_list = []
        prob_sp_list = []
        for iii in range(4):
            mean_sp_k, scale_sp_k, prob_sp_k = torch.split(self.context_analyzer_fea(ctx_s1234[kn[iii]:kn[iii+1]]),
                                                           split_size_or_sections=[256, 256, 256],
                                                           dim=-1)  # [k_len, 256] for each
            mean_sp_list.append(mean_sp_k)
            scale_sp_list.append(scale_sp_k)
            prob_sp_list.append(prob_sp_k)
        mean_hp, scale_hp, prob_hp = torch.split(self.Decoder_fea_hyp(g_fea_enc_q_hyp_q), split_size_or_sections=[256, 256, 256], dim=-1)  # [N_fea, 256] for each
        mean_ch, scale_ch, prob_ch = self.fea_channel_ctx.forward(g_fea_enc_q)  # [N_fea, 256] for each

        print('Start compressing fea...')

        bits_fea_hyp = encoder_factorized_chunk(
            x=g_fea_enc_q_hyp_q,  # [N_fea, 64]
            lower_func=self.EF_fea._logits_cumulative,
            Q=self.Q,
            file_name=os.path.join(root_path, 'g_fea_enc_q_hyp_q.b')
        )

        bits_fea_main = 0
        for l_sp in range(4):
            k_st = kn[l_sp]
            k_ed = kn[l_sp+1]
            mean_sp = mean_sp_list[l_sp]
            scale_sp = scale_sp_list[l_sp]
            prob_sp = prob_sp_list[l_sp]
            for l_ch in range(4):
                c_st = l_ch*64
                c_ed = l_ch*64+64
                # from: mean_sp: [N_fea, 256], scale_sp: [N_fea, 256], prob_sp: [N_fea, 256]
                # to: mean_sp_l: [k_len, 64], scale_sp_l: [k_len, 64], prob_sp_l: [k_len, 64]
                mean_sp_l, scale_sp_l, prob_sp_l = mean_sp[:, c_st:c_ed], scale_sp[:, c_st:c_ed], prob_sp[:, c_st:c_ed]
                mean_hp_l, scale_hp_l, prob_hp_l = mean_hp[k_st:k_ed, c_st:c_ed], scale_hp[k_st:k_ed, c_st:c_ed], prob_hp[k_st:k_ed, c_st:c_ed]
                mean_ch_l, scale_ch_l, prob_ch_l = mean_ch[k_st:k_ed, c_st:c_ed], scale_ch[k_st:k_ed, c_st:c_ed], prob_ch[k_st:k_ed, c_st:c_ed]
                probs = torch.stack([prob_sp_l, prob_hp_l, prob_ch_l], dim=-1)  # [k_len, 256, 3]
                probs = torch.softmax(probs, dim=-1)  # [k_len, 256, 3]
                prob_sp_l, prob_hp_l, prob_ch_l = probs[..., 0], probs[..., 1], probs[..., 2]  # [k_len, 256] for each
                bits_fea_main_tmp = encoder_gaussian_mixed_chunk(
                    x=g_fea_enc_q[k_st:k_ed, c_st:c_ed].contiguous().view(-1),  # from: [N_fea, 256]
                    mean_list=[mean_sp_l.contiguous().view(-1), mean_hp_l.contiguous().view(-1), mean_ch_l.contiguous().view(-1)],
                    scale_list=[scale_sp_l.contiguous().view(-1), scale_hp_l.contiguous().view(-1), scale_ch_l.contiguous().view(-1)],
                    prob_list=[prob_sp_l.contiguous().view(-1), prob_hp_l.contiguous().view(-1), prob_ch_l.contiguous().view(-1)],
                    Q=self.Q,
                    file_name=os.path.join(root_path, f'g_fea_enc_q_sp{l_sp}_ch{l_ch}.b'),
                    chunk_size=c_size_fea,
                )
                bits_fea_main += bits_fea_main_tmp
        bits_fea = bits_fea_main + bits_fea_hyp

        print('Start compressing feq...')

        # ---------
        fe_q_hyp = self.Encoder_feq_hyp(fe_q)  # [N_feq, 24]
        fe_q_hyp_q = self.quantize(fe_q_hyp, self.Q, testing)  # [N_feq, 24]

        # norm_xyz_clamp: [N_g, 3], mask_feq: [N_g], fe_final: [N_g, 48], norm_xyz_clamp[s1:s2][mask_feq[s1:s2]]: [t2-t1, 3]
        ctx_s2 = self.feq_spatial_ctx.forward(norm_xyz_clamp[s0:s1], norm_xyz_clamp[s1:s2][mask_feq[s1:s2]], fe_final[s0:s1], determ=determ_codec)  # [t2-t1, dim]
        ctx_s3 = self.feq_spatial_ctx.forward(norm_xyz_clamp[s0:s2], norm_xyz_clamp[s2:s3][mask_feq[s2:s3]], fe_final[s0:s2], determ=determ_codec)  # [t3-t2, dim]
        ctx_s4 = self.feq_spatial_ctx.forward(norm_xyz_clamp[s0:s3], norm_xyz_clamp[s3:s4][mask_feq[s3:s4]], fe_final[s0:s3], determ=determ_codec)  # [t4-t3, dim]
        ctx_s1 = torch.zeros(size=[mask_feq[s0:s1].sum(), ctx_s2.shape[-1]], dtype=ctx_s2.dtype, device=ctx_s2.device)  # [t1-t0, dim]
        ctx_s1234 = torch.cat([torch.cat([ctx_s1, ctx_s2, ctx_s3, ctx_s4], dim=0), freq_enc_xyz[mask_feq]], dim=-1)  # [N_feq, dim]

        mean_sp_list = []
        scale_sp_list = []
        prob_sp_list = []
        for iii in range(4):
            mean_sp_k, scale_sp_k, prob_sp_k = torch.split(self.context_analyzer_feq(ctx_s1234[tn[iii]:tn[iii+1]]),
                                                           split_size_or_sections=[48, 48, 48],
                                                           dim=-1)  # [t_len, 48] for each
            mean_sp_list.append(mean_sp_k)
            scale_sp_list.append(scale_sp_k)
            prob_sp_list.append(prob_sp_k)
        mean_hp, scale_hp, prob_hp = torch.split(self.Decoder_feq_hyp(fe_q_hyp_q), split_size_or_sections=[48, 48, 48], dim=-1)  # [N_feq, 48] for each
        mean_ch, scale_ch, prob_ch = self.feq_channel_ctx.forward(fe_q)  # [N_feq, 48] for each

        bits_feq_hyp = encoder_factorized_chunk(
            x=fe_q_hyp_q,  # [N_feq, 24]
            lower_func=self.EF_feq._logits_cumulative,
            Q=self.Q,
            file_name=os.path.join(root_path, 'fe_q_hyp_q.b')
        )

        bits_feq_main = 0
        for l_sp in range(4):
            t_st = tn[l_sp]
            t_ed = tn[l_sp+1]
            mean_sp = mean_sp_list[l_sp]
            scale_sp = scale_sp_list[l_sp]
            prob_sp = prob_sp_list[l_sp]
            for l_ch in range(3):
                # from: mean_sp: [N_feq, 48], scale_sp: [N_feq, 48], prob_sp: [N_feq, 48]
                # to: mean_sp_l: [t_len, 16], scale_sp_l: [t_len, 16], prob_sp_l: [t_len, 16]
                mean_sp_l, scale_sp_l, prob_sp_l = mean_sp[:, l_ch::3], scale_sp[:, l_ch::3], prob_sp[:, l_ch::3]
                # mean_sp_l, scale_sp_l, prob_sp_l = mean_sp[t_st:t_ed, l_ch::3], scale_sp[t_st:t_ed, l_ch::3], prob_sp[t_st:t_ed, l_ch::3]
                mean_hp_l, scale_hp_l, prob_hp_l = mean_hp[t_st:t_ed, l_ch::3], scale_hp[t_st:t_ed, l_ch::3], prob_hp[t_st:t_ed, l_ch::3]
                mean_ch_l, scale_ch_l, prob_ch_l = mean_ch[t_st:t_ed, l_ch::3], scale_ch[t_st:t_ed, l_ch::3], prob_ch[t_st:t_ed, l_ch::3]
                probs = torch.stack([prob_sp_l, prob_hp_l, prob_ch_l], dim=-1)  # [t_len, 48, 3]
                probs = torch.softmax(probs, dim=-1)  # [t_len, 48, 3]
                prob_sp_l, prob_hp_l, prob_ch_l = probs[..., 0], probs[..., 1], probs[..., 2]  # [t_len, 48] for each
                bits_feq_main_tmp = encoder_gaussian_mixed_chunk(
                    x=fe_q[t_st:t_ed, l_ch::3].contiguous().view(-1),  # from: [N_feq, 48]
                    mean_list=[mean_sp_l.contiguous().view(-1), mean_hp_l.contiguous().view(-1), mean_ch_l.contiguous().view(-1)],
                    scale_list=[scale_sp_l.contiguous().view(-1), scale_hp_l.contiguous().view(-1), scale_ch_l.contiguous().view(-1)],
                    prob_list=[prob_sp_l.contiguous().view(-1), prob_hp_l.contiguous().view(-1), prob_ch_l.contiguous().view(-1)],
                    Q=Q_fe[t_st:t_ed, l_ch::3].contiguous().view(-1),  # from: [N_feq, 48]
                    file_name=os.path.join(root_path, f'fe_q_sp{l_sp}_ch{l_ch}.b'),
                    chunk_size=c_size_feq,
                )
                bits_feq_main += bits_feq_main_tmp
        bits_feq = bits_feq_main + bits_feq_hyp

        print('Start compressing geo...')

        # ---------
        geo_q_hyp = self.Encoder_geo_hyp(geo_q)  # [N_g, 16]
        geo_q_hyp_q = self.quantize(geo_q_hyp, self.Q, testing)  # [N_g, 16]

        # norm_xyz_clamp: [N_g, 3], geo_q: [N_g, 8],  norm_xyz_clamp[s1:s2]: [s2-s1, 3]
        ctx_s2 = self.feq_spatial_ctx.forward(norm_xyz_clamp[s0:s1], norm_xyz_clamp[s1:s2], geo_q[s0:s1], determ=determ_codec)  # [s2-s1, dim]
        ctx_s3 = self.feq_spatial_ctx.forward(norm_xyz_clamp[s0:s2], norm_xyz_clamp[s2:s3], geo_q[s0:s2], determ=determ_codec)  # [s3-s2, dim]
        ctx_s4 = self.feq_spatial_ctx.forward(norm_xyz_clamp[s0:s3], norm_xyz_clamp[s3:s4], geo_q[s0:s3], determ=determ_codec)  # [s4-s3, dim]
        ctx_s1 = torch.zeros(size=[s1-s0, ctx_s2.shape[-1]], dtype=ctx_s2.dtype, device=ctx_s2.device)  # [s1-s0, dim]
        ctx_s1234 = torch.cat([torch.cat([ctx_s1, ctx_s2, ctx_s3, ctx_s4], dim=0), freq_enc_xyz], dim=-1)  # [N_g, dim]

        # mean_sp, scale_sp, prob_sp = torch.split(self.context_analyzer_geo(ctx_s1234), split_size_or_sections=[8, 8, 8], dim=-1)  # [N_g, 8] for each

        mean_sp_list = []
        scale_sp_list = []
        prob_sp_list = []
        for iii in range(4):
            mean_sp_k, scale_sp_k, prob_sp_k = torch.split(self.context_analyzer_geo(ctx_s1234[sn[iii]:sn[iii+1]]),
                                                           split_size_or_sections=[8, 8, 8],
                                                           dim=-1)  # [N_len, 8] for each
            mean_sp_list.append(mean_sp_k)
            scale_sp_list.append(scale_sp_k)
            prob_sp_list.append(prob_sp_k)
        mean_hp, scale_hp, prob_hp = torch.split(self.Decoder_geo_hyp(geo_q_hyp_q), split_size_or_sections=[8, 8, 8], dim=-1)  # [N_g, 8] for each

        bits_geo_hyp = encoder_factorized_chunk(
            x=geo_q_hyp_q,  # [N_g, 16]
            lower_func=self.EF_geo._logits_cumulative,
            Q=self.Q,
            file_name=os.path.join(root_path, 'geo_q_hyp_q.b')
        )

        bits_geo_main = 0
        for l_sp in range(4):
            s_st = sn[l_sp]
            s_ed = sn[l_sp+1]
            mean_sp = mean_sp_list[l_sp]
            scale_sp = scale_sp_list[l_sp]
            prob_sp = prob_sp_list[l_sp]
            # from: mean_sp: [N_g, 8], scale_sp: [N_g, 8], prob_sp: [N_g, 8]
            # to: mean_sp_l: [s_len, 8], scale_sp_l: [s_len, 8], prob_sp_l: [s_len, 8]
            mean_sp_l, scale_sp_l, prob_sp_l = mean_sp, scale_sp, prob_sp
            # mean_sp_l, scale_sp_l, prob_sp_l = mean_sp[s_st:s_ed], scale_sp[s_st:s_ed], prob_sp[s_st:s_ed]
            mean_hp_l, scale_hp_l, prob_hp_l = mean_hp[s_st:s_ed], scale_hp[s_st:s_ed], prob_hp[s_st:s_ed]
            probs = torch.stack([prob_sp_l, prob_hp_l], dim=-1)  # [N_len, 8, 3]
            probs = torch.softmax(probs, dim=-1)  # [N_len, 8, 3]
            prob_sp_l, prob_hp_l = probs[..., 0], probs[..., 1]  # [N_len, 8] for each
            bits_geo_main_tmp = encoder_gaussian_mixed_chunk(
                x=geo_q[s_st:s_ed].contiguous().view(-1),  # from: [N_g, 8]
                mean_list=[mean_sp_l.contiguous().view(-1), mean_hp_l.contiguous().view(-1)],
                scale_list=[scale_sp_l.contiguous().view(-1), scale_hp_l.contiguous().view(-1)],
                prob_list=[prob_sp_l.contiguous().view(-1), prob_hp_l.contiguous().view(-1)],
                Q=Q_geo[s_st:s_ed].contiguous().view(-1),  # from: [N_g, 8]
                file_name=os.path.join(root_path, f'geo_q_sp{l_sp}.b'),
                chunk_size=c_size_geo,
            )
            bits_geo_main += bits_geo_main_tmp
        bits_geo = bits_geo_main + bits_geo_hyp

        bits = bits_fea + bits_feq + bits_geo + bits_xyz + bits_mask

        g_fea_out = torch.cat([fe_final, geo_q], dim=-1)  # [N_choose, 48+1+3+4]

        return (g_xyz, g_fea_out, mask,
                bits/b2M, bits_xyz/b2M, bits_mask/b2M,
                bits_fea/b2M, bits_fea_main/b2M, bits_fea_hyp/b2M,
                bits_feq/b2M, bits_feq_main/b2M, bits_feq_hyp/b2M,
                bits_geo/b2M, bits_geo_main/b2M, bits_geo_hyp/b2M,
                )

    def decomprss(self, means=None, stds=None, root_path='./', chunk_size_list=()):
        c_size_fea, c_size_feq, c_size_geo = chunk_size_list
        g_xyz = decompress_gaussian_params(
            bin_path=os.path.join(root_path, 'xyz_gpcc.bin'),
        )[0]  # [N_g, 3]
        torch.manual_seed(1)
        shuffled_indices = torch.randperm(g_xyz.size(0))
        g_xyz = g_xyz[shuffled_indices]  # [N_g, 3]
        norm_xyz, norm_xyz_clamp, mask_xyz = normalize_xyz(g_xyz, K=self.norm_radius, means=means, stds=stds)   # [N_g, 3]
        freq_enc_xyz = self.freq_enc(norm_xyz_clamp)  # [N_g, freq_output]
        N_g = g_xyz.shape[0]
        mask = decoder(
            N_len=N_g,
            file_name=os.path.join(root_path, 'mask.b')
        ).view(N_g, 1)  # [N_g, 1]
        mask_fea = mask.detach()[:, 0].to(torch.bool)  # [N_g]
        mask_feq = torch.logical_not(mask_fea)  # [N_g]

        Q_fe = (self.Q_fe * self.ad_fe[:, 0:1] + self.ad_fe[:, 1:2]) * (1 + torch.tanh(self.ad_fe[:, 2:3]))  # [1, 1]
        Q_op = (self.Q_op * self.ad_op[:, 0:1] + self.ad_op[:, 1:2]) * (1 + torch.tanh(self.ad_op[:, 2:3]))  # [1, 1]
        Q_sc = (self.Q_sc * self.ad_sc[:, 0:1] + self.ad_sc[:, 1:2]) * (1 + torch.tanh(self.ad_sc[:, 2:3]))  # [1, 1]
        Q_ro = (self.Q_ro * self.ad_ro[:, 0:1] + self.ad_ro[:, 1:2]) * (1 + torch.tanh(self.ad_ro[:, 2:3]))  # [1, 1]
        Q_fe = Q_fe.repeat(mask_feq.sum(), 48)  # [N_feq, 48]
        Q_op = Q_op.repeat(N_g, 1)  # [N_g, 1]
        Q_sc = Q_sc.repeat(N_g, 3)  # [N_g, 3]
        Q_ro = Q_ro.repeat(N_g, 4)  # [N_g, 4]

        Q_geo = torch.cat([Q_op, Q_sc, Q_ro], dim=-1)  # [N_g, 8]

        s0 = 0
        s1 = N_g // 6 * 1  # 6, 1
        s2 = N_g // 6 * 2  # 6, 2
        s3 = N_g // 6 * 4  # 6, 4
        s4 = N_g
        sn = [s0, s1, s2, s3, s4]

        k0 = 0
        k1 = int(mask_fea[s0:s1].sum().item())
        k2 = int(mask_fea[s0:s2].sum().item())
        k3 = int(mask_fea[s0:s3].sum().item())
        k4 = int(mask_fea[s0:s4].sum().item())
        kn = [k0, k1, k2, k3, k4]

        t0 = 0
        t1 = int(mask_feq[s0:s1].sum().item())
        t2 = int(mask_feq[s0:s2].sum().item())
        t3 = int(mask_feq[s0:s3].sum().item())
        t4 = int(mask_feq[s0:s4].sum().item())
        tn = [t0, t1, t2, t3, t4]

        print('Start decompressing fea...')

        g_fea_enc_q_hyp_q = decoder_factorized_chunk(
            lower_func=self.EF_fea._logits_cumulative,
            Q=self.Q,
            N_len=mask_fea.sum().item(),  # N_fea
            dim=64,
            file_name=os.path.join(root_path, 'g_fea_enc_q_hyp_q.b')
        )  # [N_fea, 64]
        mean_hp, scale_hp, prob_hp = torch.split(self.Decoder_fea_hyp(g_fea_enc_q_hyp_q), split_size_or_sections=[256, 256, 256], dim=-1)  # [N_fea, 256] for each

        g_fea_enc_q = torch.zeros(size=[mask_fea.sum(), 256], dtype=torch.float32, device='cuda')  # [N_fea, 256]
        for l_sp in range(4):
            k_st = kn[l_sp]
            k_ed = kn[l_sp+1]
            s_st = sn[l_sp]
            s_ed = sn[l_sp+1]
            if l_sp == 0:
                ctx_sn = torch.zeros(size=[k1-k0, self.cafea_indim-self.freq_enc.output_dim], dtype=torch.float32, device='cuda')  # [k_st-k0, 48]=[k1-k0, 48]
            else:
                fea_grid_feature_curr = self.latdim_2_griddim_fea(g_fea_enc_q[k0:k_st])  # [k_st-k0, 48]
                # norm_xyz_clamp: [N_g, 3], mask_fea: [N_g], fea_grid_feature_curr: [k_st-k0, 48], norm_xyz_clamp[s_st:s_ed][mask_fea[s_st:s_ed]]: [k_ed-k_st, 3]
                ctx_sn = self.feq_spatial_ctx.forward(norm_xyz_clamp[s0:s_st][mask_fea[s0:s_st]], norm_xyz_clamp[s_st:s_ed][mask_fea[s_st:s_ed]], fea_grid_feature_curr, determ=True)  # [k_ed-k_st, dim]
            ctx_sn = torch.cat([ctx_sn, freq_enc_xyz[s_st:s_ed][mask_fea[s_st:s_ed]]], dim=-1)  # [k_ed-k_st, dim]
            mean_sp, scale_sp, prob_sp = torch.split(self.context_analyzer_fea(ctx_sn), split_size_or_sections=[256, 256, 256], dim=-1)
            for l_ch in range(4):
                c_st = l_ch*64
                c_ed = l_ch*64+64
                mean_sp_l, scale_sp_l, prob_sp_l = mean_sp[:, c_st:c_ed], scale_sp[:, c_st:c_ed], prob_sp[:, c_st:c_ed]  # [k_ed-k_st, 48] for each
                mean_hp_l, scale_hp_l, prob_hp_l = mean_hp[k_st:k_ed, c_st:c_ed], scale_hp[k_st:k_ed, c_st:c_ed], prob_hp[k_st:k_ed, c_st:c_ed]  # [k_ed-k_st, 48] for each
                mean_ch_l, scale_ch_l, prob_ch_l = self.fea_channel_ctx.forward(g_fea_enc_q[k_st:k_ed], to_dec=l_ch)  # [k_ed-k_st, 48] for each
                probs = torch.stack([prob_sp_l, prob_hp_l, prob_ch_l], dim=-1)  # [N_fea, 48, 3]
                probs = torch.softmax(probs, dim=-1)  # [N_fea, 48, 3]
                prob_sp_l, prob_hp_l, prob_ch_l = probs[..., 0], probs[..., 1], probs[..., 2]  # [N_fea, 48] for each
                g_fea_enc_q_tmp = decoder_gaussian_mixed_chunk(
                    mean_list=[mean_sp_l.contiguous().view(-1), mean_hp_l.contiguous().view(-1), mean_ch_l.contiguous().view(-1)],
                    scale_list=[scale_sp_l.contiguous().view(-1), scale_hp_l.contiguous().view(-1), scale_ch_l.contiguous().view(-1)],
                    prob_list=[prob_sp_l.contiguous().view(-1), prob_hp_l.contiguous().view(-1), prob_ch_l.contiguous().view(-1)],
                    Q=self.Q,
                    file_name=os.path.join(root_path, f'g_fea_enc_q_sp{l_sp}_ch{l_ch}.b'),
                    chunk_size=c_size_fea,
                ).contiguous().view(k_ed-k_st, c_ed-c_st)  # [k_ed-k_st, 64]
                g_fea_enc_q[k_st:k_ed, c_st:c_ed] = g_fea_enc_q_tmp

        g_fea_out = self.Decoder_fea(g_fea_enc_q)  # [N_fea, 256]
        fe_dec = torch.cat([self.head_f_dc(g_fea_out), self.head_f_rst(g_fea_out)], dim=-1)  # [N_fea, 48]

        print('Start decompressing feq...')
        fe_q_hyp_q = decoder_factorized_chunk(
            lower_func=self.EF_feq._logits_cumulative,
            Q=self.Q,
            N_len=mask_feq.sum().item(),  # N_fea
            dim=24,
            file_name=os.path.join(root_path, 'fe_q_hyp_q.b')
        )  # [N_feq, 24]
        mean_hp, scale_hp, prob_hp = torch.split(self.Decoder_feq_hyp(fe_q_hyp_q), split_size_or_sections=[48, 48, 48], dim=-1)  # [N_feq, 48] for each
        fe_q = torch.zeros(size=[mask_feq.sum(), 48], dtype=torch.float32, device='cuda')  # [N_feq, 48]
        fe_final = torch.zeros(size=[N_g, 48], dtype=torch.float32, device='cuda')  # [N_g, 48]
        fe_final[mask_fea] = fe_dec
        for l_sp in range(4):
            t_st = tn[l_sp]
            t_ed = tn[l_sp+1]
            s_st = sn[l_sp]
            s_ed = sn[l_sp+1]
            if l_sp == 0:
                ctx_sn = torch.zeros(size=[t1-t0, self.cafeq_indim-self.freq_enc.output_dim], dtype=torch.float32, device='cuda')  # [t_st-t0, 48]=[t1-t0, 48]
            else:
                # norm_xyz_clamp: [N_g, 3], mask_feq: [N_g], fe_final: [N_g, 48], norm_xyz_clamp[s1:s2][mask_feq[s1:s2]]: [t2-t1, 3]
                ctx_sn = self.feq_spatial_ctx.forward(norm_xyz_clamp[s0:s_st], norm_xyz_clamp[s_st:s_ed][mask_feq[s_st:s_ed]], fe_final[s0:s_st], determ=True)  # [t_ed-t_st, dim]
            ctx_sn = torch.cat([ctx_sn, freq_enc_xyz[s_st:s_ed][mask_feq[s_st:s_ed]]], dim=-1)  # [t_ed-t_st, dim]
            mean_sp, scale_sp, prob_sp = torch.split(self.context_analyzer_feq(ctx_sn), split_size_or_sections=[48, 48, 48], dim=-1)
            for l_ch in range(3):
                mean_sp_l, scale_sp_l, prob_sp_l = mean_sp[:, l_ch::3], scale_sp[:, l_ch::3], prob_sp[:, l_ch::3]  # [t_ed-t_st, 16] for each
                mean_hp_l, scale_hp_l, prob_hp_l = mean_hp[t_st:t_ed, l_ch::3], scale_hp[t_st:t_ed, l_ch::3], prob_hp[t_st:t_ed, l_ch::3]  # [t_ed-t_st, 16] for each
                mean_ch_l, scale_ch_l, prob_ch_l = self.feq_channel_ctx.forward(fe_q[t_st:t_ed], to_dec=l_ch)  # [t_ed-t_st, 16] for each
                probs = torch.stack([prob_sp_l, prob_hp_l, prob_ch_l], dim=-1)  # [t_ed-t_st, 16, 3]
                probs = torch.softmax(probs, dim=-1)  # [t_ed-t_st, 16, 3]
                prob_sp_l, prob_hp_l, prob_ch_l = probs[..., 0], probs[..., 1], probs[..., 2]  # [t_ed-t_st, 16] for each
                fe_q_tmp = decoder_gaussian_mixed_chunk(
                    mean_list=[mean_sp_l.contiguous().view(-1), mean_hp_l.contiguous().view(-1), mean_ch_l.contiguous().view(-1)],
                    scale_list=[scale_sp_l.contiguous().view(-1), scale_hp_l.contiguous().view(-1), scale_ch_l.contiguous().view(-1)],
                    prob_list=[prob_sp_l.contiguous().view(-1), prob_hp_l.contiguous().view(-1), prob_ch_l.contiguous().view(-1)],
                    Q=Q_fe[t_st:t_ed, l_ch::3].contiguous().view(-1),  # from: [N_feq, 48]
                    file_name=os.path.join(root_path, f'fe_q_sp{l_sp}_ch{l_ch}.b'),
                    chunk_size=c_size_feq,
                ).contiguous().view(t_ed-t_st, 16)  # [t_ed-t_st, 16]
                fe_q[t_st:t_ed, l_ch::3] = fe_q_tmp
                fe_final[s_st:s_ed][mask_feq[s_st:s_ed], l_ch::3] = fe_q_tmp

        print('Start decompressing geo...')
        geo_q_hyp_q = decoder_factorized_chunk(
            lower_func=self.EF_geo._logits_cumulative,
            Q=self.Q,
            N_len=N_g,  # N_g
            dim=16,
            file_name=os.path.join(root_path, 'geo_q_hyp_q.b')
        )  # [N_g, 8]
        mean_hp, scale_hp, prob_hp = torch.split(self.Decoder_geo_hyp(geo_q_hyp_q), split_size_or_sections=[8, 8, 8], dim=-1)  # [N_g, 8] for each
        geo_q = torch.zeros(size=[N_g, 8], dtype=torch.float32, device='cuda')  # [N_g, 8]
        for l_sp in range(4):
            s_st = sn[l_sp]
            s_ed = sn[l_sp+1]
            if l_sp == 0:
                ctx_sn = torch.zeros(size=[s1-s0, self.cageo_indim-self.freq_enc.output_dim], dtype=torch.float32, device='cuda')  # [s_st-s0, 8]=[s1-s0, 8]
            else:
                # norm_xyz_clamp: [N_g, 3], geo_q: [N_g, 8], norm_xyz_clamp[s1:s2]: [s2-s1, 3]
                ctx_sn = self.feq_spatial_ctx.forward(norm_xyz_clamp[s0:s_st], norm_xyz_clamp[s_st:s_ed], geo_q[s0:s_st], determ=True)  # [s_ed-s_st, dim]
            ctx_sn = torch.cat([ctx_sn, freq_enc_xyz[s_st:s_ed]], dim=-1)  # [s_ed-s_st, dim]
            mean_sp, scale_sp, prob_sp = torch.split(self.context_analyzer_geo(ctx_sn), split_size_or_sections=[8, 8, 8], dim=-1)

            mean_sp_l, scale_sp_l, prob_sp_l = mean_sp, scale_sp, prob_sp  # [s_ed-s_st, 8] for each
            mean_hp_l, scale_hp_l, prob_hp_l = mean_hp[s_st:s_ed], scale_hp[s_st:s_ed], prob_hp[s_st:s_ed]  # [s_ed-s_st, 8] for each
            probs = torch.stack([prob_sp_l, prob_hp_l], dim=-1)  # [N_len, 8, 3]
            probs = torch.softmax(probs, dim=-1)  # [N_len, 8, 3]
            prob_sp_l, prob_hp_l = probs[..., 0], probs[..., 1]  # [N_len, 8] for each
            geo_q_tmp = decoder_gaussian_mixed_chunk(
                mean_list=[mean_sp_l.contiguous().view(-1), mean_hp_l.contiguous().view(-1)],
                scale_list=[scale_sp_l.contiguous().view(-1), scale_hp_l.contiguous().view(-1)],
                prob_list=[prob_sp_l.contiguous().view(-1), prob_hp_l.contiguous().view(-1)],
                Q=Q_geo[s_st:s_ed].contiguous().view(-1),  # from: [N_g, 8]
                file_name=os.path.join(root_path, f'geo_q_sp{l_sp}.b'),
                chunk_size=c_size_geo,
            ).contiguous().view(s_ed-s_st, 8)  # [s_ed-s_st, 16]
            geo_q[s_st:s_ed] = geo_q_tmp

        g_fea_fused_dec = torch.cat([fe_final, geo_q], dim=-1)
        return g_xyz, g_fea_fused_dec
