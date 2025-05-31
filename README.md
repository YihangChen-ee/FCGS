# [ICLR'25] FCGS
Official Pytorch implementation of **Fast Feedforward 3D Gaussian Splatting Compression**.
## Compress existing 3DGS rapidly in seconds without optimization!

[Yihang Chen](https://yihangchen-ee.github.io), 
[Qianyi Wu](https://qianyiwu.github.io), 
[Mengyao Li](https://scholar.google.com/citations?user=fAIEYrEAAAAJ&hl=zh-CN&oi=ao), 
[Weiyao Lin](https://weiyaolin.github.io),
[Mehrtash Harandi](https://sites.google.com/site/mehrtashharandi/),
[Jianfei Cai](http://jianfei-cai.github.io)

[[`Paper`](https://openreview.net/pdf?id=DCandSZ2F1)] [[`Arxiv`](https://arxiv.org/pdf/2410.08017)] [[`Project`](https://yihangchen-ee.github.io/project_fcgs/)] [[`Github`](https://github.com/YihangChen-ee/FCGS)]

## Links
You are welcomed to check a series of works from our group on 3D radiance field representation compression as listed below:
- 🎉 [CNC](https://github.com/yihangchen-ee/cnc/) [CVPR'24]: efficient NeRF compression! [[`Paper`](https://openaccess.thecvf.com/content/CVPR2024/papers/Chen_How_Far_Can_We_Compress_Instant-NGP-Based_NeRF_CVPR_2024_paper.pdf)] [[`Arxiv`](https://arxiv.org/pdf/2406.04101)] [[`Project`](https://yihangchen-ee.github.io/project_cnc/)]
- 🏠 [HAC](https://github.com/yihangchen-ee/hac/) [ECCV'24]: efficient 3DGS compression! [[`Paper`](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/01178.pdf)] [[`Arxiv`](https://arxiv.org/pdf/2403.14530)] [[`Project`](https://yihangchen-ee.github.io/project_hac/)]
- 💪 [HAC++](https://github.com/yihangchen-ee/hac-plus/) [ARXIV'25]: an enhanced compression method over HAC! [[`Arxiv`](https://arxiv.org/pdf/2501.12255)] [[`Project`](https://yihangchen-ee.github.io/project_hac++/)]
- 🚀 [FCGS](https://github.com/yihangchen-ee/fcgs/) [ICLR'25]: fast optimization-free 3DGS compression! [[`Paper`](https://openreview.net/pdf?id=DCandSZ2F1)] [[`Arxiv`](https://arxiv.org/pdf/2410.08017)] [[`Project`](https://yihangchen-ee.github.io/project_fcgs/)]
- 🪜 [PCGS](https://github.com/yihangchen-ee/pcgs/) [ARXIV'25]: progressive 3DGS compression! [[`Arxiv`](https://arxiv.org/pdf/2503.08511)] [[`Project`](https://yihangchen-ee.github.io/project_pcgs/)]

## Overview
<p align="left">
<img src="assets/teaser.png" width=80% height=80% 
class="center">
</p>

Although various compression techniques have been proposed, previous art suffers from a common limitation: 
*for any existing 3DGS, per-scene optimization is needed to achieve compression, making the compression sluggish and slow.* 
To address this issue, we introduce Fast Compression of 3D Gaussian Splatting (FCGS), 
an optimization-free model that can compress 3DGS representations rapidly in a single feed-forward pass,
which significantly reduces compression time from minutes to seconds.

## Performance
<p align="left">
<img src="assets/main_curve.png" width=80% height=80% 
class="center">
</p>

While all the other approaches are optimization-based compression which have natural advantages for a better RD performance, we still outperform most of them in an optimization-free manner for fast compression.
Our compression time is only ```1/10``` compared to others!

## Installation

We tested our code on a server with Ubuntu 20.04.1, cuda 11.8, gcc 9.4.0. We use NVIDIA L40s GPU (48G).

1. Clone our code
```
git clone git@github.com:YihangChen-ee/FCGS.git --recursive
```

2. Install environment
```
conda env create --file environment.yml
conda activate FCGS_env
```

3. Install ```tmc3``` (for GPCC)

- Please refer to [tmc3 github](https://github.com/MPEGGroup/mpeg-pcc-tmc13) for installation.
- Don't forget to add ```tmc3``` to your environment variable, otherwise you must manually specify its location [in our code](https://github.com/YihangChen-ee/FCGS/blob/main/model/gpcc_utils.py) by searching ```change tmc3 path``` (2 places in total).
- Tips: ```tmc3``` is commonly located at ```PATH/TO/mpeg-pcc-tmc13/build/tmc3```.

## Dataset
DL3DV-GS-960P dataset contains 6939 samples of `undistorted images`, `camera poses`, and `pre-trained 3DGS`, under 960P resolution.
This dataset is originated from DL3DV, and post processed by us. More information can be found in our paper's Appendix.
- The dataset is now open-source under the DL3DV license. Please find the downloading instructions in the [DL3DV-GS-960P](https://huggingface.co/datasets/DL3DV/DL3DV-GS-960P) page.

## Run
FCGS can *directly* compress any existing 3DGS representations to bitstreams. The input should be a *.ply* file following the 3DGS format.

### To compress a *.ply* file to bitstreams, run:

```
python encode_single_scene.py --lmd A_lambda --ply_path_from PATH/TO/LOAD/point_cloud.ply --bit_path_to PATH/TO/SAVE/BITSTREAMS --determ 1
```
 - ```lmd```: the trade-off parameter for size and fidelity. Chosen in [```1e-4```, ```2e-4```, ```4e-4```, ```8e-4```, ```16e-4```].
 - ```ply_path_from```: A *.ply* file. Path to load the source *.ply* file.
 - ```bit_path_to```: A directory. Path to save the compressed bitstreams.
 - ```determ```: see [atomic statement](https://github.com/YihangChen-ee/FCGS/blob/main/docs/atomic_statement.md)

### To decompress a *.ply* file from bitstreams, run:

```
python decode_single_scene.py --lmd A_lambda --bit_path_from PATH/TO/LOAD/BITSTREAMS --ply_path_to PATH/TO/SAVE/point_cloud.ply
```
 - ```lmd```: the trade-off parameter for size and fidelity. Chosen in [```1e-4```, ```2e-4```, ```4e-4```, ```8e-4```, ```16e-4```].
 - ```bit_path_from```: A directory. Path to load the compressed bitstreams.
 - ```ply_path_to```: A *.ply* file. Path to save the decompressed *.ply* file.

### To decompress a *.ply* file from bitstreams and validate fidelity of the decompressed 3DGS, run:

```
python decode_single_scene_validate.py --lmd A_lambda --bit_path_from PATH/TO/LOAD/BITSTREAMS --ply_path_to PATH/TO/SAVE/point_cloud.ply --source_path PATH/TO/SOURCE/SCENES
```
 - ```source_path```: A directory. Path to load the source scene images for validation.

### Tips
FCGS is compatible with pruning-based techniques such as [Mini-Splatting](https://github.com/fatPeter/mini-splatting) and [Trimming the fat](https://github.com/salmanali96/trimming-the-fat). You can *directly* apply FCGS to the *.ply* file output by these two approaches to further boost the compression performance.

## CUDA accelerated arithmetic codec
We alongside publish a CUDA-based arithmetic codec implementation (based on [torchac](https://github.com/fab-jul/torchac)), you can find it in [arithmetic](https://github.com/YihangChen-ee/FCGS/blob/main/submodules/arithmetic) and its usage [here](https://github.com/YihangChen-ee/FCGS/blob/main/model/encodings_cuda.py).

## Contact

- Yihang Chen: yhchen.ee@sjtu.edu.cn

## Citation

If you find our work helpful, please consider citing:

```bibtex
@inproceedings{fcgs2025,
  title={Fast Feedforward 3D Gaussian Splatting Compression},
  author={Chen, Yihang and Wu, Qianyi and Li, Mengyao and Lin, Weiyao and Harandi, Mehrtash and Cai, Jianfei},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025}
}
```


## Acknowledgement

 - We thank all authors from [3D-GS](https://github.com/graphdeco-inria/gaussian-splatting) for presenting such an excellent work.
 - We thank all authors from [DL3DV](https://dl3dv-10k.github.io/DL3DV-10K/) for collecting the fantastic DL3DV dataset.
 - We thank [Xiangrui](https://liuxiangrui.github.io)'s help on GPCC codec.
