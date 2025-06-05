#  SkimthenFocus:Integrating Contextual and Fine-grained Views for Repetitive Action Counting

Here is the official implementation for IJCV 2025 paper "SkimthenFocus:Integrating Contextual and Fine-grained Views for Repetitive Action Counting" 

## Introduction

 The key to action counting is accurately locating each video’s repetitive actions. Instead of estimating the probability of
 each frame belonging to an action directly, we propose a dual-branch network, i.e., SkimFocusNet, working in a two-step
 manner. The modeldrawsinspiration from empirical observations indicating that humans initially engage in coarse skimming
 of entire sequences to quickly locate potential target action frames and grasp general motion patterns. This is followed by
 finer, frame-by-frame focusing to precisely determine whether the located frames align with the target actions. Specifically,
 SkimFocusNet incorporates a skim branch and a focus branch.Theskim branch scans the global contextual information
 throughout the sequence to identify potential target action for guidance. Subsequently, the focus branch utilizes the guidance
 to diligently identify repetitive actions using a long-short adaptive guidance (LSAG) block. Additionally, we have observed
 that videos in existing datasets often feature only one type of repetitive action, which inadequately represents real-world
 scenarios. To more accurately describe real-life situations, we establish the Multi-RepCount dataset, which includes videos
 containing multiple repetitive motions. On Multi-RepCount, our SkimFoucsNet can perform specified action counting, that
 is, to enable counting a particular action type by referencing an exemplary video. This capability substantially exhibits
 our method’s robustness, particularly in accurately performing action counting despite the presence of interfering actions.
 Extensive experimentsdemonstratethatSkimFocusNetachievesstate-of-the-artperformanceswithsignificantimprovements. We also conduct a thorough ablation study to evaluate the network components. 

## Usage
### Dataset Preparation
Please refer to the Homepage of [RepCount Dataset](https://svip-lab.github.io/dataset/RepCount_dataset.html). 

### Enviroment
Please refer to [requirement.txt](https://github.com/isotopezzq/SkimFocusNet/blob/main/requirement.txt) for installation

### NPZ File Generation
` python ./tools/newnpz64.py `

### TRAIN & TEST
` python train.py `

` python test.py `
