# Uncertainty Estimation by Density Aware Evidential Deep Learning
This repo contains an official PyTorch implementation for the paper [*Uncertainty Estimation by Density Aware Evidential Deep Learning*](https://openreview.net/pdf?id=JtkruFHcRK) in ICML 2024. 

## Abstract 
Evidential deep learning (EDL) has shown remarkable success in uncertainty estimation. However, there is still room for improvement, particularly in out-of-distribution (OOD) detection and classification tasks. The limited OOD detection performance of EDL arises from its inability to reflect the distance between the testing example and training data when quantifying uncertainty, while its limited classification performance stems from its parameterization of the concentration parameters. To address these limitations, we propose a novel method called *Density Aware Evidential Deep Learning (DAEDL)*. DAEDL integrates the feature space density of the testing example with the output of EDL during the prediction stage, while using a novel parameterization that resolves the issues in the conventional parameterization. We prove that DAEDL enjoys a number of favorable theoretical properties. DAEDL demonstrates state-of-the-art performance across diverse downstream tasks related to uncertainty estimation and classification. 

## Method
![DAEDL](https://github.com/TaeseongYoon/DAEDL/assets/65948713/961b63ce-2920-4b02-865f-86d0a9156972)



## Citation
If the code or the paper has been useful in your research, pleace consider citing our paper :
'''
@inproceedings{
yoon2024uncertainty,
title={Uncertainty Estimation by Density Aware Evidential Deep Learning},
author={Taeseong Yoon and Heeyoung Kim},
booktitle={Forty-first International Conference on Machine Learning},
year={2024},
url={https://openreview.net/forum?id=JtkruFHcRK}
}
'''
