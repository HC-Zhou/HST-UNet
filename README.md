# HST-UNet

> The codes for the work "Hybrid Shunted Transformer Embedding UNet
> for Remote Sensing Image Semantic Segmentation"

## Requirement
```text
matplotlib==3.3.4
numpy==1.19.2
Pillow==9.2.0
tifffile==2020.10.1
timm==0.4.12
torch==1.10.2
torchsummary==1.5.1
torchvision==0.11.3
tqdm==4.63.0
```

## Model
![图片](HST-UNet.png)

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
    --log_path '/saveModels/logging/Potsdam/'
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
    --log_path '/saveModels/logging/Vaihingen/'
```
## Citation
```
Zhou, H., Xiao, X., Li, H. et al. Hybrid Shunted Transformer embedding UNet for remote sensing image semantic segmentation. Neural Comput & Applic (2024). https://doi.org/10.1007/s00521-024-09888-4
```

