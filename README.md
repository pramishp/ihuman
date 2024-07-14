# iHuman

## Prerequisites

* Cuda 11.8
* Conda
* A C++14 capable compiler
  * __Linux:__ GCC/G++ 8 or higher

## Setup
First make sure all the Prerequisites are installed in your operating system. Then, invoke

```bash
conda env create -f environment.yml
conda activate ihuman
cd submodules
bash ./install.sh
```

## Running the code

### Step 1: Download Dataset
a. download dataset from this link https://drive.google.com/drive/folders/11h_uRhuMALHjmQKu-FIrpGi0kv5_ZC6u?usp=sharing
b. place it in {root}/data/people_snapshot/

This file is obtained 
### Step 2: Download Models
a. Download model from this link: (pending because of license issue)

[//]: # (https://drive.google.com/file/d/17OdyNkfdFKFqBnmFMZtmT9B-6AXKAZeG/view?usp=share_link)
<br>
b. unzip and place it in {root}/data/smpl/small/*

*Note: The License of this model belongs to the SMPL ...*

### Step 3:
4. python train.py

You can modify the training parameters in the `conf/mmpeoplesnapshot_fine.yaml` file.

## Acknowledgement

Our code is based on several interesting and helpful projects:

- InstantAvatar: <https://github.com/tijiang13/InstantAvatar>
- GaussianSplatting: <https://github.com/graphdeco-inria/gaussian-splatting>
- Diff-gaussian-rasterization: <https://github.com/ashawkey/diff-gaussian-rasterization>
- SuGaR: <https://github.com/Anttwo/SuGaR>
- Animatable 3D Gaussian:<https://github.com/jimmyYliu/Animatable-3D-Gaussian>
- GART: https://github.com/JiahuiLei/GART
- Anim-NeRF: https://github.com/JanaldoChen/Anim-NeRF

We are grateful to the developers and contributors of these repositories for their hard work and dedication to the open-source community. Without their contributions, our project would not have been possible.