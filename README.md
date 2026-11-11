<p align="center">
  <img src="resources/Logo.png" width="220" alt="GIFT Logo">
</p>

<h2 align="center">
  🎁 GIFT: Representation Geometry Matters for Single-Domain Generalized Object Detection
</h2>

<p align="center">
  <a href="https://zhangyin1996.github.io/">Yin Zhang</a>,
  <a href="https://yaoyz96.github.io/">Yaoyue Zheng</a>,
  <a href="https://scholar.google.com/citations?user=mgpE1noAAAAJ&hl">Yongqiang Zhang</a>,
  <a href="https://scholar.google.com/citations?user=zdIW07oAAAAJ">Bogdan Raducanu</a>,
  <a href="https://openreview.net/profile?id=~Dan_Liu14">Dan Liu</a>
</p>

<p align="center">
  <strong>🎉 Accepted at NeurIPS 2026</strong>
</p>

## :bell: Updates
+ 2026-11-09: Model checkpoints are  available. [Google Drive](https://drive.google.com/drive/folders/1wRsuyERJVGrfR_DgNy3IRTpp7lQjOoFF)
+ 2026-10-27: Paper is available. [arXiv](https://arxiv.org/pdf/2610.35374)
+ 2026-10-26: Code is available.

<p align="center">
  <img src="resources/Figure.png" width="700" alt="Figure">
</p>


## 1. Environment Setup
```bash
# python 3.9, cuda 11.8, pytorch 2.2.0
conda create -n gift -y python=3.9
conda activate gift
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r ./requirements.txt
pip install albumentations==1.4.4 timm einops
pip install -U openmim
mim install mmengine
mim install mmcv==2.2.0
pip install xformers==0.0.24 --index-url https://download.pytorch.org/whl/cu118
pip install -v -e .
pip install numpy==1.26.0
```

## 2. Pretraining Weights

* You can directly download the preprocessed weights from this [link](https://drive.google.com/drive/folders/1SGlq6eXOg2PWBB_aUdLFk5VuEQTxlCvs) and place them in the `checkpoints` folder.
* You can also follow the steps below to process them yourself.
  - Download pre-trained weights [dinov2_vitl14_pretrain.pth](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth) from facebookresearch
  - Convert
  ```bash
  python tools/convert_models/convert_dinov2.py checkpoints/dinov2_vitl14_pretrain.pth checkpoints/dinov2_converted_1024.pth --height 1024 --width 1024
  python tools/convert_models/convert_dinov2.py checkpoints/dinov2_vitl14_pretrain.pth checkpoints/dinov2_converted_640.pth --height 640 --width 640
  ```

## 3. Dataset Preparation
+ Download the [DWD](https://github.com/AmingWu/Single-DGOD), `/data/datasets/Single-DGOD`.
+ Download the [original Cityscapes](https://www.cityscapes-dataset.com/downloads/), `/data/datasets/cityscapes`, and prepare the  COCO-format detection annotations.
  - Run the following command:
  ```
  python tools/dataset_converters/cityscapes.py /data/datasets/cityscapes –nproc 8
  ```
  - Place the resulting `instancesonly_filtered_gtFine_train.json` and `instancesonly_filtered_gtFine_val.json` in the `/data/datasets/cityscapes/annotations/`.
+ The final dataset directory structure under `/data/datasets` is as follows:

```
|-- Single-DGOD/
|   |-- Daytime_Sunny/
|   |   |-- daytime_clear/
|   |       |-- VOC2007
|   |           |-- Annotations
|   |               |-- 0a0a0b1a-7c39d841.xml
|   |               |-- ...xml
|   |           |-- ImageSets
|   |               |-- Main
|   |                   |-- train.txt
|   |                   |-- ...txt
|   |           |-- JPEGImages
|   |               |-- 0a0a0b1a-7c39d841.jpg
|   |-- DaytimeFoggy/
|   |   |-- daytime_foggy/
|   |       |-- VOC2007
|   |           |-- ...
|   |-- Dusk-rainy/
|   |   |-- dusk_rainy/
|   |       |-- VOC2007
|   |           |-- ...
|   |-- Night_rainy/
|   |   |-- night_rainy/
|   |       |-- VOC2007
|   |           |-- ...
|   |-- Night-Sunny/
|   |   |-- Night-Sunny/
|   |       |-- VOC2007  ### [Important] The official 'Night-Sunny' dataset does not include a VOC2007 folder, you need to create one yourself.
|   |           |-- ...

|-- cityscapes/
|   |-- leftImg8bit/
|   |   |-- train/
|   |       |-- aachen/
|   |       |-- ...
|   |   |-- val/
|   |       |-- franfurt/
|   |       |-- ...
|   |-- annotations/
|   |   |-- instancesonly_filtered_gtFine_train.json
|   |   |-- instancesonly_filtered_gtFine_val.json
```
:rotating_light: During robustness evaluation, Cityscapes-C corruptions are generated on the fly, no pre-generated Cityscapes-C dataset is required.

## 4. Train and Test
+ DWD
```shell
# Train with Faster R-CNN (single GPU)
CUDA_VISIBLE_DEVICES=0 bash tools/dist_train.sh configs/gift/gift-faster-rcnn_r50_fpn_dwd.py 1 --amp --work-dir ./work_dir/gift_faster_dwd --find_unused_parameters
# Train with Faster R-CNN (mutiple GPUs)
CUDA_VISIBLE_DEVICES=0,1,2,3 bash tools/dist_train.sh configs/gift/gift-faster-rcnn_r50_fpn_dwd.py 4 --amp --work-dir ./work_dir/gift_faster_dwd --find_unused_parameters
# Test
bash tools/dist_test.sh configs/gift/gift-faster-rcnn_r50_fpn_dwd.py ./work_dir/gift_faster_dwd/epoch_12.pth 1 --work-dir ./work_dir/test_gift_faster_dwd

# Train with DETR-based detector Dino (single GPUs)
CUDA_VISIBLE_DEVICES=0 bash tools/dist_train.sh configs/gift/gift_dinov2_dinohead_bs1x4_dwd.py 1 --amp --work-dir ./work_dir/gift_detr_dwd --find_unused_parameters
# Train with DETR-based detector Dino (mutiple GPUs)
CUDA_VISIBLE_DEVICES=0,1,2,3 bash tools/dist_train.sh configs/gift/gift_dinov2_dinohead_bs1x4_dwd.py 4 --amp --work-dir ./work_dir/gift_detr_dwd --find_unused_parameters
# Test
bash tools/dist_test.sh configs/gift/gift-faster-rcnn_r50_fpn_dwd.py ./work_dir/gift_detr_dwd/epoch_12.pth 1 --work-dir ./work_dir/test_gift_detr_dwd
```

+ Cityscapes-C

```shell
# Taking a single GPU as an example, the same applies to multiple GPUs.
# Train on Source Domain (Cityscapes)
bash tools/dist_train.sh configs/gift/gift-faster-rcnn_r50_fpn_cityscapes.py 1 --amp --work-dir ./work_dir/gift_faster_cityscapes --find_unused_parameters
# Test on Target Domain
bash tools/dist_test_robustness.sh configs/gift/gift-faster-rcnn_r50_fpn_cityscapes.py ./work_dir/gift_faster_cityscapes/epoch_12.pth 1 --out path/to/your/xxx.pkl --work-dir ./work_dir --corruptions benchmark
```
## 5. Results 
💻 We re-ran the experiments on October 23 using a single NVIDIA RTX 3090 GPU. (Slightly higher than the results reported in the paper)
+ DWD

| Model  | Day Clear | Day Foggy | Dusk Rainy | Night Rainy | Night Clear | Avg. |mPC 
|:------:|:---------:|:---------:|:----------:|:-----------:|:-----------:|:----:|:----:|
| Faster RCNN | 75.8      | 56.5      | 64.6      | 48.1        | 62.6        | 61.6| **58.0**|
| DETR-based | 76.2      | 58.8      | 66.3      | 50.5        | 64.1        | 63.2| **59.9**|

+ Cityscapes-C

| Corruption | Level-1 | Level-2 | Level-3 | Level-4 | Level-5 | Avg. |
|:----------:|:-------:|:-------:|:-------:|:-------:|:-------:|:----:|
| Norm.      | -    | -    | -    | -    | -    | 63.9 |
| Gaussian   | 49.8 | 39.7 | 22.9 | 6.5  | 0.0  | 23.8 |
| Shot       | 53.7 | 43.6 | 28.9 | 5.1  | 0.7  | 26.4 |
| Impulse    | 46.0 | 33.7 | 21.6 | 4.5  | 0.1  | 21.2 |
| Defocus    | 61.6 | 58.7 | 51.0 | 44.1 | 37.3 | 50.5 |
| Glass      | 58.0 | 54.2 | 39.5 | 35.9 | 29.8 | 43.5 |
| Motion     | 60.0 | 55.6 | 48.9 | 42.1 | 37.9 | 48.9 |
| Zoom       | 29.1 | 23.4 | 18.5 | 15.2 | 12.5 | 19.7 |
| Snow       | 51.4 | 38.3 | 41.6 | 34.1 | 34.9 | 40.1 |
| Frost      | 57.9 | 47.2 | 43.6 | 39.6 | 34.2 | 44.5 |
| Fog        | 61.2 | 61.5 | 60.3 | 59.1 | 55.5 | 59.5 |
| Brightness | 63.7 | 63.8 | 63.7 | 62.5 | 61.5 | 63.0 |
| Contrast   | 62.9 | 62.9 | 61.3 | 57.0 | 50.7 | 59.0 |
| Elastic    | 62.8 | 62.0 | 61.4 | 61.1 | 60.6 | 61.6 |
| Pixelate   | 61.2 | 59.7 | 56.5 | 52.6 | 43.7 | 54.7 |
| JPEG       | 55.5 | 52.5 | 51.1 | 43.9 | 37.8 | 48.2 |
| **mPC**    | -    | -    | -    | -    | -    | **44.3** |


+ You can follow the steps below to reproduce the results: 
  - Download the weight from here. [Google Drive](https://drive.google.com/drive/folders/1wRsuyERJVGrfR_DgNy3IRTpp7lQjOoFF)
  - Place it under `weights/` (create the folder if it doesn’t exist), e.g., `weights/gift_faster_dwd.pth`
  - Run evaluation:
  ```bash
  # DWD Faster R-CNN
  bash tools/dist_test.sh configs/gift/gift-faster-rcnn_r50_fpn_dwd.py ./weights/gift_faster_dwd.pth 1 --work-dir ./work_dir/test_gift_faster_dwd
  # DWD DETR-based detector Dino
  bash tools/dist_test.sh configs/gift/gift_dinov2_dinohead_bs1x4_dwd.py ./weights/gift_detr_dwd.pth 1 --work-dir ./work_dir/test_gift_detr_dwd
  # Cityscapes-C
  bash tools/dist_test_robustness.sh configs/gift/gift-faster-rcnn_r50_fpn_cityscapes.py ./weights/gift_faster_cityscapes.pth 1 --out path/to/your/xxx.pkl --work-dir ./work_dir/test_gift_faster_cityscapes --corruptions benchmark
  ```
## 6. Citation
If you find GIFT useful in your research, please consider citing and give us a star ⭐ on GitHub:
```
@article{zhang2026gift,
  title={GIFT: Representation Geometry Matters for Single-Domain Generalized Object Detection},
  author={Zhang, Yin and Zheng, Yaoyue and Zhang, Yongqiang and Raducanu, Bogdan and Liu, Dan},
  journal={arXiv preprint arXiv:2610.35374},
  year={2026}
}
```

## 7. Acknowledgment
+ The code is based on [Cauvis](https://github.com/lichen1015/Cauvis) (NeurIPS 2025) 
+ In addition, some ideas inspired by [OFT](https://oft.wyliu.com/) (NeurIPS 2023) and [PAID](https://github.com/wangkunyu241/PAID) (NeurIPS 2025).

**Many thanks for these great works!**
