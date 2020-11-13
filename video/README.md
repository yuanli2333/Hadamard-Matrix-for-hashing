### The codes of video hashing are based on the [MFNet](https://github.com/cypw/PyTorch-MFNet)

### Prepare dataset and 3D pretrained models: 
Download the UCF101 and HMDB51 datasets, put the videos in dataset/UCF101/raw/data/ and dataset/HMDB51/raw/data/, please refer details on [MFNet](https://github.com/cypw/PyTorch-MFNet). 

Download the pretrained models on classification in [here](https://drive.google.com/file/d/1mz7Zh0ufQICStxsXzen6FWiZod-cH850/view?usp=sharing), unzip it and put in 'network/pretrained'.

### Train

On UCF101:
```
python train_ucf101.py --batch-size 32 --gpus 0,1
```

On HMDB51:
```
python train_hmdb51.py --batch-size 32 --gpus 0,1
```

### Test
Download the trained models in [here](https://drive.google.com/drive/folders/1LW7Tdc_-2h3BhZNg5KVirsXC8iSgLXa4?usp=sharing), then put in 'exp/models/'


Calculate mAP on UCF101:

```
CUDA_VISIBLE_DEVICES=0 python hash_test.py --dataset UCF101 --pretrained_3d exp/models/64bit_87.38_PyTorch-MFNet-master.pth
```


Calculate mAP on HMDB51:
```
CUDA_VISIBLE_DEVICES=0 python hash_test.py --dataset HMDB51 --pretrained_3d exp/models/64bit_57.67_PyTorch-MFNet-master_ep-0099_HMDB51_ndp.pth
```


### Reference
If you find this repo useful, please consider citing:
```
@inproceedings{yuan2020central,
  title={Central Similarity Quantization for Efficient Image and Video Retrieval},
  author={Yuan, Li and Wang, Tao and Zhang, Xiaopeng and Tay, Francis EH and Jie, Zequn and Liu, Wei and Feng, Jiashi},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={3083--3092},
  year={2020}
}

@inproceedings{chen2018multifiber,
  title={Multi-Fiber networks for Video Recognition},
  author={Chen, Yunpeng and Kalantidis, Yannis and Li, Jianshu and Yan, Shuicheng and Feng, Jiashi},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2018}
}

```
