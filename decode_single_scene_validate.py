import time

import torch
import torch.nn as nn
from model.FCGS_model import FCGS
import os
from gaussian_renderer import render
from gaussian_renderer import GaussianModel
import numpy as np
from utils.image_utils import psnr
from tqdm import tqdm
from argparse import ArgumentParser
import sys
from scene import Scene
from typing import NamedTuple
import lpips
from utils.loss_utils import l1_loss, ssim
lpips_fn = lpips.LPIPS(net='vgg').to('cuda')

class D1(NamedTuple):
    data_device: str
    eval: bool
    images: str
    lod: int
    model_path: str
    resolution: int
    sh_degree: int
    source_path: str
    white_background: bool

class D2(NamedTuple):
    convert_SHs_python: bool
    compute_cov3D_python: bool
    debug: bool

def train(args):
    dataset = D1(
        data_device='cuda',
        eval=True,
        images='images',
        lod=0,
        model_path="",
        resolution=-1,
        sh_degree=3,
        source_path=args.source_path,
        white_background=False,
    )
    pipeline = D2(
        convert_SHs_python=False,
        compute_cov3D_python=False,
        debug=False
    )

    with torch.no_grad():
        gaussians = GaussianModel(3)  # dataset.sh_degree = 3
        scene = Scene(dataset, shuffle=False)
        views = scene.getTestCameras()

    step_num = len(os.listdir(os.path.join(args.bit_path_from, str(args.lmd))))
    lmd = args.lmd
    chunk_size_list = [200_0000, 100_0000, 100_0000]

    CM = FCGS(
        Q=1,
        resolutions_list=[300, 400, 500],
        resolutions_list_3D=[70, 80, 90],
        norm_radius=args.nr,
    ).cuda()
    CM.load_state_dict(torch.load(f'./checkpoints/checkpoint_{lmd}.pkl'), strict=True)

    g_xyz_list = []
    g_fea_list = []
    CM.eval()
    with torch.no_grad():
        for s in range(step_num):
            bit_save_path = os.path.join(args.bit_path_from, f"{lmd}/{s}")
            g_xyz_out, g_fea_out = CM.decomprss(root_path=bit_save_path, chunk_size_list=chunk_size_list)
            g_xyz_list.append(g_xyz_out)
            g_fea_list.append(g_fea_out)
            
    g_xyz = torch.cat(g_xyz_list, dim=0)
    g_fea = torch.cat(g_fea_list, dim=0)

    f_dc, f_rst, op, sc, ro = torch.split(g_fea, split_size_or_sections=[3, 45, 1, 3, 4], dim=-1)
    gaussians._xyz = nn.Parameter(g_xyz)
    gaussians._features_dc = nn.Parameter(f_dc.view(-1, 1, 3))
    gaussians._features_rest = nn.Parameter(f_rst.view(-1, 15, 3))
    gaussians._opacity = nn.Parameter(op.view(-1, 1))
    gaussians._scaling = nn.Parameter(sc.view(-1, 3))
    gaussians._rotation = nn.Parameter(ro.view(-1, 4))

    gaussians.save_ply(args.ply_path_to)
    print(f"Decompressed ply file saved to {args.ply_path_to}!")

    with torch.no_grad():
        ssim_test_sum = 0
        L1_test_sum = 0
        lpips_test_sum = 0
        psnr_test_sum = 0
        curr_rendering_list = []
        for _, view in enumerate(tqdm(views, desc="Rendering progress")):
            rendering = render(view, gaussians, pipe=pipeline, bg_color=torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda"))[
                "render"]  # [3, H, W]
            gt = view.original_image[0:3, :, :].to("cuda")
            rendering = torch.round(rendering.mul(255).clamp_(0, 255)) / 255.0
            ssim_test_sum += (ssim(rendering, gt)).mean().double().item()
            L1_test_sum += l1_loss(rendering, gt).mean().double().item()
            lpips_test_sum += lpips_fn(rendering, gt).mean().double().item()
            psnr_test_sum += psnr(rendering, gt).mean().double().item()
            curr_rendering_list.append(rendering)
        ssim_avg = ssim_test_sum / len(views)
        Ll1_avg = L1_test_sum / len(views)
        lpips_avg = lpips_test_sum / len(views)
        psnr_avg = psnr_test_sum / len(views)

        print(f"Evaluation results: psnr: {psnr_avg:.4f}, ssim: {ssim_avg:.4f}, lpips: {lpips_avg:.4f}, Ll1: {Ll1_avg:.4f}")


if __name__ == "__main__":
    parser = ArgumentParser(description="dataset_param")
    parser.add_argument("--lmd", default=1e-4, choices=[1e-4, 2e-4, 4e-4, 8e-4, 16e-4], type=float)
    parser.add_argument("--nr", default=3, type=float)
    parser.add_argument("--bit_path_from", default="./bitstreams/tmp/", type=str)
    parser.add_argument("--ply_path_to", default="./bitstreams/tmp/point_cloud.ply", type=str)
    parser.add_argument("--source_path", default="./path/to/scene/", type=str)
    args = parser.parse_args(sys.argv[1:])
    train(args)

