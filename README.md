# GAtFUN

This repository contains the PyTorch implementation for diffusion-based GAtFUN.

![GAtFuN](https://github.com/user-attachments/assets/b7797947-9ace-4944-9970-ec3dc26bf1fb)

## Comparison with SOTA methods on In-the-Wild videos

Check out our Demo video: 

https://github.com/user-attachments/assets/563bcd70-4530-40bd-851d-328ca5596dd9


## Dependencies
Make sure you have the following dependencies installed (python):

* pytorch >= 0.4.0
* matplotlib=3.1.0
* einops
* timm
* tensorboard
* CLIP
* Detectron2

```bash
pip install git+https://github.com/openai/CLIP.git
```
```
python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html
```

You should download [MATLAB](https://www.mathworks.com/products/matlab-online.html) if you want to evaluate our model on MPI-INF-3DHP dataset.

## Datasets

Our method is quantitatively evaluated on [Human3.6M](http://vision.imar.ro/human3.6m), [MPI-INF-3DHP](https://vcai.mpi-inf.mpg.de/3dhp-dataset/) and [HumanEva](http://humaneva.is.tue.mpg.de/) datasets. 

### Human3.6M
We set up the Human3.6M dataset in the same way as [VideoPose3D](https://github.com/facebookresearch/VideoPose3D/blob/master/DATASETS.md).  You can download the processed data from here: 

[`data_2d_h36m_cpn_ft_h36m_dbb.npz`](https://drive.google.com/file/d/1ina9zTS1ZnT2sjdFYr9GTnljWYdTM82S/view?usp=sharing) is the 2D keypoints detected by [CPN](https://github.com/GengDavid/pytorch-cpn).  

[`data_2d_h36m_gt.npz`](https://drive.google.com/file/d/1ZWQSCGaMjqpnjsPoOwvQS9dO_1XWc4jx/view?usp=sharing) is the ground truth of 2D keypoints. 

[`data_3d_h36m.npz`](https://drive.google.com/file/d/1GBPBBBnL19MbMHqx7Cl-xaLkgAy8vHqR/view?usp=sharing) is the ground truth of 3D human joints. 

Put them in the `./data` directory.

### MPI-INF-3DHP
We set up the MPI-INF-3DHP dataset following [P-STMO](https://github.com/paTRICK-swk/P-STMO) and [D3DP](https://github.com/paTRICK-swk/D3DP/tree/main). You can download the processed data from here:

[`data_ori_3dhp.rar`](https://drive.google.com/file/d/18ZC4bD0-esmx-JQoz4Gcu5ytwJDE1U8c/view?usp=sharing) includes both ground truth 2D and 3D poses (officially called "annot3").

Put them in the `./data` directory. 

### HumanEva-I
We set up the HumanEva-I dataset similar to [VideoPose3D](https://github.com/facebookresearch/VideoPose3D/blob/master/DATASETS.md). You can download the processed data from here:

[`data_2d_humaneva15_gt.npz`](https://drive.google.com/file/d/1kNoTuypL-jGRcdGqyBYRIB3iWfIeiQmh/view?usp=sharing) is the ground truth of 2D keypoints. 

[`data_3d_humaneva15.npz`](https://drive.google.com/file/d/1BtuijI1aYeXFZIgFI7je0PD8G7xP2-nt/view?usp=sharing) is the ground truth of 3D human joints. 

Put them in the `./data` directory.

## Training and Evaluation

We trained our models on 1*NVIDIA RTX 4090.

### Human3.6M

#### 2D CPN inputs
To train our model using the 2D keypoints obtained by CPN as inputs, please run:
```bash
python main_h36m.py -k cpn_ft_h36m_dbb -c checkpoint/h36m -gpu 0 --nolog
```

To evaluate our GAtFUN using the 2D keypoints obtained by CPN as inputs, please run:
```bash
python main_h36m.py -k cpn_ft_h36m_dbb -c checkpoint/h36m -gpu 0 --nolog --evaluate <checkpoint_file> -num_proposals 20 -sampling_timesteps 10 --p2
```

#### 2D ground truth inputs
To train our GAtFUN model using the 2D ground truth keypoints as inputs, please run:
```bash
python main_h36m.py -k gt -c checkpoint/h36m_gt -gpu 0 --nolog --save_lmin 21 --save_lmax 23
```

To evaluate our GAtFUN using the 2D ground truth keypoints as inputs, please run:
```bash
python main_h36m.py -k gt -c checkpoint/h36m_gt -gpu 0 --nolog --evaluate <checkpoint_file> -num_proposals 20 -sampling_timesteps 10 --p2
```

### MPI-INF-3DHP
To train our model using the ground truth 2D poses as inputs, please run:
```bash
python main_3dhp.py -c checkpoint/3dhp -gpu 0 --nolog
```

To evaluate our GAtFUN using the ground truth 2D poses as inputs, please run:
```bash
python main_3dhp.py -c checkpoint/3dhp -gpu 0 --nolog --evaluate <checkpoint_file> -num_proposals 20 -sampling_timesteps 10
```
After that, the predicted 3D poses under P-Best, P-Agg, J-Best, J-Agg settings are saved as four files (`.mat`) in `./checkpoint`. To get the MPJPE, AUC, PCK metrics, you can evaluate the predictions by running a Matlab script `./3dhp_test/test_util/mpii_test_predictions_ori_py.m` (you can change 'aggregation_mode' in line 29 to get results under different settings). Then, the evaluation results are saved in `./3dhp_test/test_util/mpii_3dhp_evaluation_sequencewise_ori_{setting name}_t{iteration index}.csv`. You can manually average the three metrics in these files over six sequences to get the final results.

### HumanEva-I
To train our model using the ground truth 2D poses as inputs, please run:
```bash
python main_humaneva.py -k gt -c 'checkpoint/humaneva_gt' -a 'Walk,Jog' -gpu 0 --nolog
```

To evaluate our GAtFUN using the ground truth 2D poses as inputs, please run:
```bash
python main_humaneva.py -k gt -c 'checkpoint/humaneva_gt' -a 'Walk,Jog' -gpu 0 --nolog --evaluate <checkpoint_file> --by-subject -num_proposals 20 -sampling_timesteps 10 --p2
```

### In-the-wild Inference



### Pretrained Models
[Google Drive](https://drive.google.com/drive/folders/1iEc6o7KlUfYpOYCN5Eo_rN0phLHnJrP1?usp=sharing)

## Acknowledgement
Our code refers to the following repositories.
* [MotionDiffuse](https://github.com/mingyuan-zhang/MotionDiffuse)
* [D3DP](https://github.com/paTRICK-swk/D3DP)
* [MixSTE](https://github.com/JinluZhang1126/MixSTE)
* [FinePOSE](https://github.com/PKU-ICST-MIPL/FinePOSE_CVPR2024)
* [KTPFormer](https://github.com/JihuaPeng/KTPFormer)
* [MotionAGFormer](https://github.com/TaatiTeam/MotionAGFormer)
* [MotionBERT](https://github.com/Walter0807/MotionBERT)

We thank the authors for releasing their codes.


