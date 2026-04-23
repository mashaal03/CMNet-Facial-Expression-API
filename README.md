# A cross-modal network for facial expression recognition
## Abstract
Deep neural networks enriched with structural information have been widely employed for facial expression recognition tasks. However, these methods often depend on hierarchical information rather than face property to finish expression recognition. In this paper, we propose a cross-modal network with strong biological and structural information for facial expression recognition (CMNet). CMNet can respectively learn expression information via face symmetry on a whole face, left and right half faces to extract complementary facial features. To prevent native effect of biological and structural information fusion, a salient facial information refinement module can obtain salient facial expression information to improve stability of an obtained facial expression classifier. To reduce reliance on unilateral facial features, a half-face alignment optimization mechanism is designed to align obtained expression information of learned left and right half faces. Our experimental results demonstrate that CMNet outperforms several novel methods, i.e., SCN and LAENet-SA for facial expression recognition.
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
