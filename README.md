## GIFT: Representation Geometry Matters for Single-Domain Generalized Object Detection

## Installation
### Pretrained DINOv2 
* **Download:** Download pre-trained weights **dinov2_vitl14_pretrain.pth** from facebookresearch. 
* **Convert:** 
  ```bash
  python tools/convert_models/convert_dinov2.py checkpoints/dinov2_vitl14_pretrain.pth checkpoints/dinov2_converted_1024.pth --height 1024 --width 1024
  python tools/convert_models/convert_dinov2.py checkpoints/dinov2_vitl14_pretrain.pth checkpoints/dinov2_converted_640.pth --height 640 --width 640
  ```
### environment
```bash
conda create -n gift -y python=3.9
conda activate gift
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r ./requirements.txt
pip install albumentations==1.4.4 timm einops
pip install -U openmim
mim install mmengine
mim install mmcv==2.2.0
##pip install xformers==0.0.24
pip install xformers==0.0.24 --index-url https://download.pytorch.org/whl/cu118
pip install -v -e .
pip install numpy==1.26.0
```

## Data Preparation
Download the **DWD** and **Cityscapes-C** datasets, and organize it in `/data/datasets` folder as follows:
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
|   |-- Dusk-rainy/
|   |-- Night_rainy/
|   |-- Night-Sunny/
```

## Train
### DWD
```shell
# Train with Faster R-CNN (single GPU)
CUDA_VISIBLE_DEVICES=0 bash tools/dist_train.sh configs/gift/gift-faster-rcnn_r50_fpn_dwd.py 1 --amp --work-dir ./work_dir/gift_faster_dwd --find_unused_parameters
# Train with Faster R-CNN (mutiple GPUs)
CUDA_VISIBLE_DEVICES=0,1,2,3 bash tools/dist_train.sh configs/gift/gift-faster-rcnn_r50_fpn_dwd.py 4 --amp --work-dir ./work_dir/gift_faster_dwd --find_unused_parameters
# Train with DETR-based detector Dino (mutiple GPUs)
CUDA_VISIBLE_DEVICES=0,1,2,3 bash tools/dist_train.sh configs/gift/gift_dinov2_dinohead_bs1x4_dwd.py 4 --amp --work-dir ./work_dir/gift_detr_dwd --find_unused_parameters
# test
bash tools/dist_test.sh configs/gift/gift-faster-rcnn_r50_fpn_dwd.py path/to/your.pth 1 --work-dir ./work_dir/test_gift_faster_dwd
```

### Cityscapes-C

```shell
# Train on Source Domain (Cityscapes)
bash tools/dist_train.sh configs/gift/gift-faster-rcnn_r50_fpn_cityscapes.py 1 --amp --work-dir ./work_dir/gift_faster_cityscapes --find_unused_parameters

# test on Target Domain
bash tools/dist_test_robustness.sh configs/gift/gift-faster-rcnn_r50_fpn_cityscapes.py path/to/your/epoch_12.pth 1 --out path/to/your/xxx.pkl --work-dir ./work_dir --corruptions benchmark
```
