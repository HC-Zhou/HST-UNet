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

## Model
The checkpoints can be found at [Google Drive](https://drive.google.com/drive/folders/1cs84JBY7JLlUVanKiMxIEMBqL9Yed6jm?usp=sharing), 
[Baidu Pan](https://pan.baidu.com/s/1HXR9CZforKhz2aPuSHulsg)(code:dti5)
