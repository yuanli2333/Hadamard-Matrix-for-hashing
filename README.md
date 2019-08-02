# Codes for paper: Central Similarity Hashing via Hadamard matrix, [arxiv](https://arxiv.org/abs/1908.00347)

We first release all codes and configurations for image hashing. Codes and configutations for video hashing will be released in the future.

## Prerequisties

Ubuntu 16.04

NVIDIA GPU + CUDA and corresponidng Pytorch framework (v0.4.1)

Python 3.6


## Datasets
1. Download database for the retrieval list of imagenet in the anonymous link [here](https://drive.google.com/open?id=1xDfg2liQzjzXxp51DEgSVMEI1trKJ_RA), and put database.txt in 'data/imagenet/'

2. Download MS COCO, ImageNet2012, NUS_WIDE in their official website: [COCO](http://cocodataset.org/#download), [ImageNet](http://image-net.org/download-images), [NUS_WIDE](https://lms.comp.nus.edu.sg/research/NUS-WIDE.htm). Unzip all data and put in 'data/dataset_name/'.


## Hash center (target)
Here, we put hash centers for imagenet we used in 'data/imagenet/hash_centers'. The methods to generate hash centers are given in the tutorial: [Tutorial_ hash_center_generation.ipynb](https://github.com/yuanli2333/Hadamard-Matrix-for-hashing/blob/master/Tutorial_%20hash_center_generation.ipynb)


## Test

Pretrained models are in the anonymous link, [here](https://drive.google.com/drive/folders/1HFLDfPvSrVITCFwolcQ3arym4PTODMHQ?usp=sharing)


It will take a long time to generate hash codes for database, because of the large-scale data size for database


Test for imagenet:

Download pre-trained model 'imagenet_64bit_0.8734_resnet50.pkl' for imagenet, put it in 'data/imagenet/', then run:

```
python test.py --data_name imagenet --gpus 0,1  --R 1000  --model_name 'imagenet_64bit_0.8734_resnet50.pkl' 
```


Test for coco:

Download pre-trained model 'coco_64bit_0.8612_resnet50.pkl' for coco, put it in 'data/coco/', then run:

```
python test.py --data_name coco --gpus 0,1  --R 5000  --model_name 'coco_64bit_0.8612_resnet50.pkl' 
```



Test for nus_wide:

Download pre-trained model 'nus_wide_64bit_0.8391_resnet50.pkl' for nus_wide, put it in 'data/nus_wide/', then run:

```
python test.py --data_name nus_wide --gpus 0,1  --R 5000  --model_name 'nus_wide_64bit_0.8391_resnet50.pkl' 
```


The MAP of retrieval on the three datasets are shown in the following:


| Dataset  | MAP(64bit) | MAP(32bit) | MAP(16bit)|
| :---     |    :---:   |    :---:   |   ---:    |
| ImageNet |    0.873   |    0.865   |   0.851   |
| MS COCO  |    0.861   |    0.838   |   0.796   |
| NUS WIDE |    0.839   |    0.825   |   0.810   |





## Train

Train on imagenet, hash bit: 64bit 

Trained model will be saved in 'data/imagenet/models/'

```
python train.py --data_name imagenet --hash_bit 64 --gpus 0,1 --model_type resnet50 --lambda1 0  --lambda2 0.05  --R 1000
```





Train on coco, hash bits: 64bit 

Trained model will be saved in 'data/coco/models/'

```
python train.py --data_name coco --hash_bit 64 --gpus 0,1 --model_type resnet50 --lambda1 0  --lambda2 0.05 --multi_lr 0.05  --R 5000
```





Train on nus_wide, hash bit: 64bit 

Trained model will be saved in 'data/nus_wide/models/'

```
python train.py --data_name nus_wide --hash_bit 64 --gpus 0,1 --model_type resnet50 --lambda1 0  --lambda2 0.05  --multi_lr 0.05 --R 5000
```



