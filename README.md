# HST-UNet

> The codes for the work "Hybrid Shunted Transformer Embedding UNet
> for Remote Sensing Image Semantic Segmentation"

## Training

```shell
!python train_Potsdam.py --batch_size 32 \
    --model 'hst_unet' \
    --epochs 150 \
    --img_size 256 \
    --pretrained '' \
    --weight_decay 1e-4 \
    --lr 0.01 \
    --seed 512 \
    --root '/mnt' \
    --num_classes 6 \
    --device 'cuda' \
    --workers 2 \
    --log_path '/saveModels/logging/Potsdam/' \
    --resume
```

```shell
!python train_Vaihingen.py --batch_size 32 \
    --model 'hst_unet' \
    --epochs 150 \
    --img_size 256 \
    --pretrained '' \
    --weight_decay 1e-4 \
    --lr 0.001 \
    --seed 512 \
    --root '/mnt' \
    --num_classes 6 \
    --device 'cuda' \
    --workers 2 \
    --log_path '/saveModels/logging/Vaihingen/' \
    --resume
```

## Method

|       Method       | Parameters(MB) | Impervious Surface(IoU) | Building(IoU) | Low Vegetation(IoU) | Tree(IoU)  |  Car(IoU)  |    MIoU    |    m-F1    |
|:------------------:|:--------------:|:-----------------------:|:-------------:|:-------------------:|:----------:|:----------:|:----------:|:----------:|
|        FCN         |     22.70      |         73.32%          |    78.97%     |       54.80%        |   70.38%   |   39.92%   |   63.46%   |   76.65%   |
|        UNet        |     25.13      |         72.91%          |    81.68%     |       57.23%        |   71.63%   |   48.29%   |   66.35%   |   79.13%   |
|    Deeplab V3+     |     38.48      |         74.85%          |    83.01%     |       56.09%        |   71.54%   |   50.30%   |   67.16%   |   79.71%   |
|      UperNet       |     102.13     |         73.45%          |    81.50%     |       55.65%        |   71.31%   |   47.26%   |   65.84%   |   78.69%   |
|       DANet        |     45.36      |         73.54%          |    81.40%     |       56.88%        |   71.21%   |   42.68%   |   65.14%   |   78.00%   |
|     TransUNet      |     100.44     |         73.27%          |    81.01%     |       55.07%        |   71.08%   |   55.13%   |   67.11%   |   79.86%   |
|      ST-UNet       |     160.97     |         76.36%          |    82.98%     |       57.79%        |   72.53%   |   61.48%   |   70.23%   |   82.15%   |
|     MsanlfNet      |     123.13     |         76.75%          |    82.83%     |       60.25%        |   71.10%   |   55.51%   |   69.29%   |   81.45%   |
| **HST-Unet(Ours)** |     112.11     |       **76.78%**        |  **85.35%**   |     **60.26%**      | **72.78%** | **62.01%** | **71.44%** | **83.00%** |

## Model

The checkpoints can be found
at [Google Drive](https://drive.google.com/drive/folders/1cs84JBY7JLlUVanKiMxIEMBqL9Yed6jm?usp=sharing),
[Baidu Pan](https://pan.baidu.com/s/1HXR9CZforKhz2aPuSHulsg)(code:dti5)
