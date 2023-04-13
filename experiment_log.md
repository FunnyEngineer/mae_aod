### 2023.03.20

utilized AOD vars:
1. Optical_Depth_055: the actual AOD value
2. AOD_QA: contains the cloud cover status


### 2023.03.21

Cover Ratio Report for CA 2018 ~ 2023:
count    9430.000000
mean        0.253064
std         0.231661
min         0.000000
25%         0.034147
50%         0.197697
75%         0.438874
max         0.944073


### 2023.03.28

1. The mean and std for CA dataset:
    mean=[99.763], std=[97.445]
2. The pytorch transform function and experimental size:
    Since the original repo for training image is 224, which is mismatch with the size of AOD image (1200), we modified the function RandomResizedCrop to RandomCrop. 
    We could modified more sub image for the AOD dataset.

### 2023.03.30

1. stuck on the aod dataset since the nan value
2. we found the random masking is not masking the raw image, instead, it is masking the latent which transfer by the patch embed layer
3. 

### 2023.04.12

1. List the detail forward process of MAE:
    1. forward encoder:
        1. patch embed
        2. positional embed
        3, random masking
        4. cls token
        5. apply transformer block
    2. forward decoder:
        1, decoder embed
        2. append mask tokens to sequence
        3. add positional embed
        4. apply transformer block
        5. predictor projection -> a Linear Layer
        6. remove cls token
    3. forward loss:
        1. patchify the raw images
        2. MSE loss
        3. mean loss per patch
        4. mean loss on removed patches

2.  related to the arch, let's think about the modify that 
    1. We have nan values in original images, which made us have to move the masking in the first stemp in forward encoder.
    2. patch embed is using conv2d to change sub-image to singal. However, since we have nan values in nearby pixels. Does that will increase the cover ratio? 

Reply to 2-2. After resizing the image, the cover ratio will decrease! What we should do is -> instead of creating the positional embeding layers for patched images, creating that for raw images -> which means to do the pixel-wise encoding. If we do this, another 

#### 2023.04.13

1. Let's move the masking step toward the patch
    1, examine the cover ratio if it reach 0.25
        if not, trim that batch
    2. 