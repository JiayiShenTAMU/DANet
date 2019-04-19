[TOC]

## Image Segmentation

### *Note: the operating system should be <u>ubuntu</u> when implementing*

### 1. Task Introduction

This project will be focusing on image segmentation using deep neural network techniques. We use a Dual Attention Network (DANet) to adaptively integrate local features with their global dependencies based on the self-attention mechanism.

### 2.Dataset

The data will come from the "Skin Lesion Analysis Towards Melanoma Detection" challenge, which can be downloaded from <https://challenge.kitware.com/#phase/5abcb19a56357d0139260e53>. The dataset is very large, more than 12GB, so here we only show 5 samples for demonstration(We here only use the training phase data).

### 3. Performance

#### 3.1 Evaluation Method

**pixel accuracy** : When considering the per-class pixel accuracy we're essentially evaluating a binary mask; a true positive represents a pixel that is correctly predicted to belong to the given class (according to the target mask) whereas a true negative represents a pixel that is correctly identified as not belonging to the given class.

​		$\operatorname{accuracy}=\frac{T P+T N}{T P+T N+F P+F N}$ 

**MIoU(Mean Intersection over Union)** : Assume that there are k+1 classes (from $L_0$ to $L_k$, including an empty class or background), and $p_{ij}$ represents the number of pixels that belong to class $i$ but are predicted to be class $j$. That is, $p_{ii}$ represents the true quantity, while $p_{ij} $,$p_{ji}$ is interpreted as false positive and false negative respectively, although both are the sum of false positive and false negative. Then

​		$M I o U=\frac{1}{k+1} \sum_{i=0}^{k} \frac{p_{i i}}{\sum_{j=0}^{k} p_{i j}+\sum_{j=0}^{k} p_{j i}-p_{i i}}$

#### 3.2 Result

**testing** (remaining 30% of the data)

average pixel accuracy = **0.908**

average MIoU = **0.791** 

### 4. How to run code 

#### 4.1 Install pytorch 

You can do as follows:

- The code is tested on python3.6 and official [Pytorch@commitfd25a2a](https://github.com/pytorch/pytorch/tree/fd25a2a86c6afa93c7062781d013ad5f41e0504b#from-source), please install PyTorch from source.
- The code is modified from [PyTorch-Encoding](https://github.com/zhanghang1989/PyTorch-Encoding). 

or, you can just downlowd the environment through https://drive.google.com/file/d/1N6hsZkTAdAaAb_7d69xf0Rw4fWrqqyHA/view?usp=sharing. and decompress it into the following path in your anaconda (if you have virtual environment, decompress it to the virtual environment path): ` lib/pythonX.X/site-pacages/`

#### 4.2 Clone the repository:

```shell
git clone https://github.com/JiayiShenTAMU/DANet.git 
cd DANet 
python setup.py install
```

Here is the model that has been well-trained. 

<https://drive.google.com/file/d/1_vQIhZXw0dI-mfnwFg2GEOpQVbu664fV/view?usp=sharing>

- Download trained model DANet above (no need to decompress) and put it in folder `./danet/cityscapes/model/` (You need to create an empty folder `/cityscapes/model/` by yourself)

#### 4.3 Testing/Evaluation

- `cd danet`
- For single scale testing, please run:

```shell
   CUDA_VISIBLE_DEVICES=0,1,2,3 python test.py --dataset ISIC --model danet --resume-dir cityscapes/model --base-size 2048 --crop-size 768 --workers 1 --backbone resnet101 --multi-grid --multi-dilation 4 8 16 --eval
   
```

- For multi-scale testing, please run:

```shell
   CUDA_VISIBLE_DEVICES=0,1,2,3 python test.py --dataset ISIC --model danet --resume-dir cityscapes/model --base-size 2048 --crop-size 1024 --workers 1 --backbone resnet101 --multi-grid --multi-dilation 4 8 16 --eval --multi-scales
```

- If you don't want to visualize the result ,only for the numeric evaluation, you can run:

```shell
   CUDA_VISIBLE_DEVICES=0,1,2,3 python test.py --dataset ISIC --model danet --resume-dir cityscapes/model --base-size 2048 --crop-size 768 --workers 1 --backbone resnet101 --multi-grid --multi-dilation 4 8 16
```

#### 4.4 Evaluation Result:

The scores will be shown as follows:

(single scale testing denotes as 'ss' and multiple scale testing denotes as 'ms')

DANet on images val set (mIoU/pAcc): **79.93/95.97** (ss) and **81.49/96.41** (ms)

#### 4.5 Training:

- `cd danet`

```shell
 CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --dataset ISIC --model  danet --backbone resnet101 --checkname danet101  --base-size 1024 --crop-size 768 --epochs 240 --batch-size 8 --lr 0.003 --workers 2 --multi-grid --multi-dilation 4 8 16
```

 Note that: We adopt multiple losses in end of the network for better training. 

#### 4.6 GUI 

a demo is shown here [https://youtu.be/ogsPt-ebnl4](https://youtu.be/ogsPt-ebnl4).

The sample data used for testing is put in `datasets/ISIC`

`cd DANet/gui`

`python gui.py`

After you run the gui, the predicted picture that has finished the segmentation will be automatically generated in `datasets/ISIC/danet_vis`

Note: in order to run the gui, step 5.1 and 5.2 must be successfully run.



