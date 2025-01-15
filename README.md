# CMNet
## Abstract
## Model weights and Datasets
Datasets can be downloaded [here](https://drive.google.com/drive/folders/1OjFbQ7ykV5x96MrAMySHwMpaJKzFgGSe?usp=sharing)  
Model weights can be download [here](https://drive.google.com/drive/folders/1kPUbCKoDKesxbpubPTxedD3DWsGGJ1qj?usp=sharing)  
## Directory structure
make a directory named "models" and put the pretrained resnet18 model in it.
make a directory named "data" and put the dataset in it.
make a directory named "experiment" for model weight files. 
The structute is like following:  

    .  
    ├── ...  
    ├── models  
    │   └── resnet18_msceleb.pth  
    ├── data  
    │   ├── rafdb  
    │   │   ├── train  
    │   │   └── test  
    │   ├── affectnet-7  
    │   │   ├── train  
    │   │   └── test  
    │   ├── ...  
    ├── experiment  
    │   ├── visual
    │   ├── rafdb
    │   │   └── rafdb.pth
    │   ├── affectnet-7  
    │   |   └── affectnet-7.pth    
    │   ├── affectnet-8  
    │   |   └── affectnet-8.pth  
    │   ├── sfew  
    │   |   └── sfew.pth  
    │   ├── caer-s  
    │   |   └── caer-s.pth  
    └── ...  

## Commands
### Train
`python train_[dataset].py`
### Test
1. **Single dataset**   
`python evaluation.py`  
2. **Cross dataset**  
`python eval_cross.py`  
