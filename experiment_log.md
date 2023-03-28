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
3. 