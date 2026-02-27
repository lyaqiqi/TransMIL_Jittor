# TransMIL: Transformer based Correlated Multiple Instance Learning for Whole Slide Image Classification [NeurIPS 2021]

TransMIL: Transformer based Correlated Multiple Instance Learning for Whole Slide Image Classification.__[NeurIPS2021]__

> **Jittor Implementation** — This repository is a reimplementation of TransMIL using the [Jittor](https://github.com/Jittor/jittor) deep learning framework.

```
@article{shao2021transmil,
  title={Transmil: Transformer based correlated multiple instance learning for whole slide image classification},
  author={Shao, Zhuchen and Bian, Hao and Chen, Yang and Wang, Yifeng and Zhang, Jian and Ji, Xiangyang and others},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  pages={2136--2147},
  year={2021}
}
```

## Abstract

With the development of computational pathology, deep learning methods for Gleason grading through whole slide images (WSIs) have excellent prospects. Since the size of WSIs is extremely large, the image label usually contains only slide-level label or limited pixel-level labels. The current mainstream approach adopts multi-instance learning to predict Gleason grades. However, some methods only considering the slide-level label ignore the limited pixel-level labels containing rich local information. Furthermore, the method of additionally considering the pixel-level labels ignores the inaccuracy of pixel-level labels. To address these problems, we propose a mixed supervision Transformer based on the multiple instance learning framework. The model utilizes both slide-level label and instance-level labels to achieve more accurate Gleason grading at the slide level. The impact of inaccurate instance-level labels is further reduced by introducing an efficient random masking strategy in the mixed supervision training process. We achieve the state-of-the-art performance on the SICAPv2 dataset, and the visual analysis shows the accurate prediction results of instance level.

## Data Preprocess

We follow the CLAM's WSI processing solution (__https://github.com/mahmoodlab/CLAM__)

```bash
# WSI Segmentation and Patching
python create_patches_fp.py --source DATA_DIRECTORY --save_dir RESULTS_DIRECTORY --patch_size 256 --preset bwh_biopsy.csv --seg --patch --stitch

# Feature Extraction
CUDA_VISIBLE_DEVICES=0,1 python extract_features_fp.py --data_h5_dir DIR_TO_COORDS --data_slide_dir DATA_DIRECTORY --csv_path CSV_FILE_NAME --feat_dir FEATURES_DIRECTORY --batch_size 512 --slide_ext .svs
```

## Installation

* Linux (Tested on Ubuntu 18.04)
* NVIDIA GPU (Tested on a single Nvidia GeForce RTX 3090)
* Python (3.7+), Jittor (1.3+), h5py, opencv-python, numpy

Please refer to the following instructions.

```bash
# create and activate the conda environment
conda create -n transmil_jittor python=3.7 -y
conda activate transmil_jittor

# install Jittor
pip install jittor

# install related packages
pip install -r requirements.txt
```

> For Jittor installation details, please refer to the [Jittor official documentation](https://github.com/Jittor/jittor).

## Dataset

The dataset CSV files are located in `dataset_csv/camelyon16/`. Please organize your dataset according to the CSV format provided, and update the data path in the corresponding config file under the `Camelyon/` directory.

## Train

```bash
python train_jt.py --stage='train' --config='Camelyon/TransMIL.yaml' --gpus=0 --fold=0
```

## Test

```bash
python train_jt.py --stage='test' --config='Camelyon/TransMIL.yaml' --gpus=0 --fold=0
```

## Results

Training logs and checkpoints are saved to `logs/Camelyon/TransMIL.yaml/fold0/` by default.


© This code is made available under the GPLv3 License and is available for non-commercial academic purposes.
