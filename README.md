#  SkimthenFocus:Integrating Contextual and Fine-grained Views for Repetitive Action Counting

Here is the official implementation for IJCV 2025 paper "SkimthenFocus: Integrating Contextual and Fine-grained Views for Repetitive Action Counting" 

## Introduction

 The key to action counting is accurately locating each video’s repetitive actions. Instead of estimating the probability of
 each frame belonging to an action directly, we propose a dual-branch network, i.e., SkimFocusNet, working in a two-step
 manner. The model draws inspiration from empirical observations indicating that humans initially engage in coarse skimming
 of entire sequences to quickly locate potential target action frames and grasp general motion patterns. This is followed by
 finer, frame-by-frame focusing to precisely determine whether the located frames align with the target actions. Specifically,
 SkimFocusNet incorporates a skim branch and a focus branch.The skim branch scans the global contextual information
 throughout the sequence to identify potential target action for guidance. Subsequently, the focus branch utilizes the guidance
 to diligently identify repetitive actions using a long-short adaptive guidance (LSAG) block. Additionally, we have observed
 that videos in existing datasets often feature only one type of repetitive action, which inadequately represents the real-world
 scenarios. To more accurately describe real-life situations, we establish the Multi-RepCount dataset, which includes videos
 containing multiple repetitive motions. On Multi-RepCount, our SkimFoucsNet can perform specified action counting, that
 is, to enable counting a particular action type by referencing an exemplary video. This capability substantially exhibits
 our method’s robustness, particularly in accurately performing action counting despite the presence of interfering actions.
 Extensive experiments demonstrate that SkimFocusNet achieves state-of-the-art performance with significant improvements. We also conduct a thorough ablation study to evaluate the network components. 

## Usage
This implementation is based on [TransRAC](https://github.com/SvipRepetitionCounting/TransRAC).
### Dataset Preparation
Please refer to the Homepage of [RepCount Dataset](https://svip-lab.github.io/dataset/RepCount_dataset.html). 

### Environment
Please refer to [requirement.txt](https://github.com/isotopezzq/SkimFocusNet/blob/main/requirement.txt) for installation.

### Checkpoint Preparation
Firstly, you should load the pretrained backbone model [videoswintiny](https://pan.baidu.com/s/1L5nIYyTIccDdk1troYzdSQ) with code g68w into the folder [pretrained](https://github.com/isotopezzq/SkimFocusNet/blob/main/pretrained).

Secondly, for testing, we also prepared our [checkpoint](https://pan.baidu.com/s/1Y-vcwsuT05byPD1wWBJfOg) with code vu65. you should load the checkpoint into the folder [checkpoint/ours](https://github.com/isotopezzq/SkimFocusNet/blob/main/checkpoint/ours).
### NPZ File Generation
` python ./tools/newnpzselect.py `

### TRAIN & TEST
` python train.py `

` python test.py `

## Citation 
If you find this project or dataset useful, please consider citing the paper.
```
Zhao, Z., Huang, X., Zhou, H. et al. Skim then Focus: Integrating Contextual and Fine-grained Views for Repetitive Action Counting. Int J Comput Vis (2025). https://doi.org/10.1007/s11263-025-02471-x
```
