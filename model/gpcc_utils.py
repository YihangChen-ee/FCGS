import os
import time
from tempfile import TemporaryDirectory
from typing import Union

import numpy as np
from plyfile import PlyData
import torch

VOXELIZE_SCALE_FACTOR = 16
CHUNK_SIZE = 32768

def gpcc_encode(encoder_path: str, ply_path: str, bin_path: str) -> None:
    """
    Compress geometry point cloud by GPCC codec.
    """
    enc_cmd = (f'{encoder_path} '
               f'--mode=0 --trisoupNodeSizeLog2=0 --mergeDuplicatedPoints=0 --neighbourAvailBoundaryLog2=8 '
               f'--intra_pred_max_node_size_log2=3 --positionQuantizationScale=1 --inferredDirectCodingMode=3 '
               f'--maxNumQtBtBeforeOt=2 --minQtbtSizeLog2=0 --planarEnabled=0 --planarModeIdcmUse=0 --cabac_bypass_stream_enabled_flag=1 '
               f'--uncompressedDataPath={ply_path} --compressedStreamPath={bin_path} ')
    enc_cmd += '> nul 2>&1' if os.name == 'nt' else '> /dev/null 2>&1'
    exit_code = os.system(enc_cmd)
    assert exit_code == 0, f'GPCC encoder failed with exit code {exit_code}.'


def gpcc_decode(decoder_path: str, bin_path: str, recon_path: str) -> None:
    """
    Decompress geometry point cloud by GPCC codec.
    """
    dec_cmd = (f'{decoder_path} '
               f'--mode=1 --outputBinaryPly=1 '
               f'--compressedStreamPath={bin_path} --reconstructedDataPath={recon_path} ')
    dec_cmd += '> nul 2>&1' if os.name == 'nt' else '> /dev/null 2>&1'
    exit_code = os.system(dec_cmd)
    assert exit_code == 0, f'GPCC decoder failed with exit code {exit_code}.'


def write_ply_geo_ascii(geo_data: np.ndarray, ply_path: str) -> None:
    """
    Write geometry point cloud to a .ply file in ASCII format.
    """
    assert ply_path.endswith('.ply'), 'Destination path must be a .ply file.'
    assert geo_data.ndim == 2 and geo_data.shape[1] == 3, 'Input data must be a 3D point cloud.'
    geo_data = geo_data.astype(int)
    with open(ply_path, 'w') as f:
        # write header
        f.writelines(['ply\n', 'format ascii 1.0\n', f'element vertex {geo_data.shape[0]}\n',
                      'property float x\n', 'property float y\n', 'property float z\n', 'end_header\n'])
        # write data
        for point in geo_data:
            f.write(f'{point[0]} {point[1]} {point[2]}\n')


def read_ply_geo_bin(ply_path: str) -> np.ndarray:
    """
    Read geometry point cloud from a .ply file in binary format.
    """
    assert ply_path.endswith('.ply'), 'Source path must be a .ply file.'

    ply_data = PlyData.read(ply_path).elements[0]
    means = np.stack([ply_data.data[name] for name in ['x', 'y', 'z']], axis=1)  # shape (N, 3)
    return means


def voxelize(means: np.ndarray) -> tuple:
    """
    Voxelization of Gaussians.
    """
    # voxelize means
    means_min, means_max = means.min(axis=0), means.max(axis=0)
    voxelized_means = (means - means_min) / (means_max - means_min)  # normalize to [0, 1]
    voxelized_means = np.round(voxelized_means * (2 ** VOXELIZE_SCALE_FACTOR - 1))

    return voxelized_means, means_min, means_max


def devoxelize(voxelized_means: np.ndarray, means_min: np.ndarray, means_max: np.ndarray) -> np.ndarray:
    voxelized_means = voxelized_means.astype(np.float32)
    means_min = means_min.astype(np.float32)
    means_max = means_max.astype(np.float32)
    means = voxelized_means / (2 ** VOXELIZE_SCALE_FACTOR - 1) * (means_max - means_min) + means_min
    return means

def dec_enc_voxelize(means, means_min=None, means_max=None):
    # means should be a torch tensor
    if means_min is None:
        if isinstance(means, torch.Tensor):
            means_min, means_max = torch.min(means, dim=0, keepdim=True)[0], torch.max(means, dim=0, keepdim=True)[0]
        else:
            means_min, means_max = means.min(axis=0), means.max(axis=0)
    voxelized_means = (means - means_min) / (means_max - means_min)  # normalize to [0, 1]
    if isinstance(means, torch.Tensor):
        voxelized_means = torch.round(voxelized_means * (2 ** VOXELIZE_SCALE_FACTOR - 1))
    else:
        voxelized_means = np.round(voxelized_means * (2 ** VOXELIZE_SCALE_FACTOR - 1))
    means = voxelized_means / (2 ** VOXELIZE_SCALE_FACTOR - 1) * (means_max - means_min) + means_min
    return means


def remove_duplicated_voxels(voxelized_means: np.ndarray, other_params: list) -> tuple:
    """
    Remove duplicated voxels.
    """
    # calculate indices of unique voxels
    _, indices_unique = np.unique(voxelized_means, axis=0, return_index=True)
    # retain unique voxels
    voxelized_means = voxelized_means[indices_unique]
    other_params = [param[indices_unique] for param in other_params]
    return voxelized_means, other_params


def sorted_voxels(voxelized_means: np.ndarray, other_params = None) -> Union[np.ndarray, tuple]:
    """
    Sort voxels by their Morton code.
    """
    indices_sorted = np.argsort(voxelized_means @ np.power(voxelized_means.max() + 1, np.arange(voxelized_means.shape[1])), axis=0)
    voxelized_means = voxelized_means[indices_sorted]
    if other_params is None:
        return voxelized_means
    other_params = other_params[indices_sorted]
    return voxelized_means, other_params


def sorted_orig_voxels(means, other_params=None):
    means = means.detach().cpu().numpy().astype(np.float32)
    voxelized_means, means_min, means_max = voxelize(means=means)
    voxelized_means, other_params = sorted_voxels(voxelized_means=voxelized_means, other_params=other_params)
    means = devoxelize(voxelized_means=voxelized_means, means_min=means_min, means_max=means_max)
    means = torch.from_numpy(means).cuda().to(torch.float32)
    return means, other_params


def write_binary_data(dst_file_handle, src_bin_path: str) -> None:
    """
    Write binary data to a binary file handle.
    """
    with open(src_bin_path, 'rb') as f:
        data = f.read()
        dst_file_handle.write(np.array([len(data), ], dtype=np.uint32).tobytes())  # 4 bytes for length
        dst_file_handle.write(data)


def read_binary_data(dst_bin_path: str, src_file_handle) -> None:
    """
    Read binary data from file handle and write it to a binary file.
    """
    length = int(np.frombuffer(src_file_handle.read(4), dtype=np.uint32)[0])
    with open(dst_bin_path, 'wb') as f:
        f.write(src_file_handle.read(length))


def compress_gaussian_params(
        gaussian_params,
        bin_path,
        gpcc_codec_path='/home/ps/YihangChen/gaussian-splatting_generation_L40S/mpeg-pcc-tmc13-master/build/tmc3/tmc3'
):
    """
    Compress Gaussian model parameters.
    - Means are compressed by GPCC codec
    - Other parameters except opacity are first quantized, and opacity, indices and codebooks are losslessly compressed by numpy.
    """
    if isinstance(gaussian_params, torch.Tensor):
        gaussian_params = gaussian_params.detach().cpu().numpy()
    means = gaussian_params
    # voxelization
    voxelized_means, means_min, means_max = voxelize(means=means)
    # sort voxels
    voxelized_means = sorted_voxels(voxelized_means=voxelized_means, other_params=None)
    means_enc = None

    # compress and write to binary file
    with TemporaryDirectory() as temp_dir:
        # write voxelized means to .ply file and then compress it by GPCC codec
        ply_path = os.path.join(temp_dir, 'voxelized_means.ply')
        write_ply_geo_ascii(geo_data=voxelized_means, ply_path=ply_path)

        means_bin_path = os.path.join(temp_dir, 'compressed.bin')
        gpcc_encode(encoder_path=gpcc_codec_path, ply_path=ply_path, bin_path=means_bin_path)

        # write head info and merge all compressed data into binary file
        with open(bin_path, 'wb') as f:
            # write head info
            head_info = np.array([means_min, means_max], dtype=np.float32)
            f.write(head_info.tobytes())  # 2 * 3 * 4 = 24 bytes

            # write voxelized means
            write_binary_data(dst_file_handle=f, src_bin_path=means_bin_path)

            # collect file size of compressed data
            file_size = {
                'means': os.path.getsize(means_bin_path) / 1024 / 1024,  # MB
                'total': os.path.getsize(bin_path) / 1024 / 1024  # MB
            }

    compress_results = {
        'num_gaussians': voxelized_means.shape[0], 'file_size': file_size
    }
    return means_enc, voxelized_means, means_min, means_max, compress_results


def decompress_gaussian_params(
        bin_path,
        gpcc_codec_path='/home/ps/YihangChen/gaussian-splatting_generation_L40S/mpeg-pcc-tmc13-master/build/tmc3/tmc3'
):
    """
    Decompress Gaussian model parameters.
    """
    assert os.path.exists(bin_path), f'Bitstreams {bin_path} not found.'

    with TemporaryDirectory() as temp_dir:
        # read head info
        with open(bin_path, 'rb') as f:
            head_info = np.frombuffer(f.read(24), dtype=np.float32)
            means_min, means_max = head_info[:3], head_info[3:]

            # read voxelized means
            means_bin_path = os.path.join(temp_dir, 'compressed.bin')
            read_binary_data(dst_bin_path=means_bin_path, src_file_handle=f)

        # decompress voxelized means by GPCC codec
        ply_path = os.path.join(temp_dir, 'voxelized_means.ply')
        gpcc_decode(decoder_path=gpcc_codec_path, bin_path=means_bin_path, recon_path=ply_path)
        voxelized_means = read_ply_geo_bin(ply_path=ply_path).astype(np.float32)
        voxelized_means = sorted_voxels(voxelized_means)  # decoded voxelized means are unsorted, thus need sorting

        # devoxelize means
        means_dec = devoxelize(voxelized_means=voxelized_means, means_min=means_min, means_max=means_max)
        means_dec = torch.from_numpy(means_dec).cuda()

    return means_dec, voxelized_means, means_min, means_max
