# Uncertainty Estimation by Density Aware Evidential Deep Learning

This repository contains the official PyTorch implementation of the paper [*Uncertainty Estimation by Density Aware Evidential Deep Learning*](https://openreview.net/pdf?id=JtkruFHcRK), accepted for publication at **ICML 2024**.

## Abstract

Evidential deep learning (EDL) has shown remarkable success in uncertainty estimation. However, there is still room for improvement, particularly in out-of-distribution (OOD) detection and classification tasks. The limited OOD detection performance of EDL arises from its inability to reflect the distance between a test example and the training data when quantifying uncertainty, while its limited classification performance stems from its parameterization of the concentration parameters. To address these limitations, we propose a novel method called *Density Aware Evidential Deep Learning (DAEDL)*. DAEDL integrates the feature-space density of a test example with the output of EDL during the prediction stage, while using a novel parameterization that resolves the issues in the conventional parameterization. We prove that DAEDL enjoys a number of favorable theoretical properties. DAEDL demonstrates state-of-the-art performance across diverse downstream tasks related to uncertainty estimation and classification.

## Method

![thumb](https://github.com/user-attachments/assets/f55a8db7-9f0e-4d0c-b4f9-f1f110362253)

## Citation

If this code or paper has been useful in your research, please consider citing our paper:

```latex
@inproceedings{yoon2024uncertainty,
  title={Uncertainty Estimation by Density Aware Evidential Deep Learning},
  author={Yoon, Taeseong and Kim, Heeyoung},
  booktitle={International Conference on Machine Learning},
  pages={57217--57243},
  year={2024},
  organization={PMLR}
}
```
