# Uncertainty Estimation by Density Aware Evidential Deep Learning
This repo contains an official PyTorch implementation for the paper "Uncertainty Estimation by Density Aware Evidential Deep Learning" in ICML 2024.

## Abstract 
Evidential deep learning (EDL) has shown remarkable success in uncertainty estimation. However, there is still room for improvement, particularly in out-of-distribution (OOD) detection and classification tasks. The limited OOD detection performance of EDL arises from its inability to reflect the distance between the testing example and training data when quantifying uncertainty, while its limited classification performance stems from its parameterization of the concentration parameters. To address these limitations, we propose a novel method called *Density Aware Evidential Deep Learning (DAEDL)*. DAEDL integrates the feature space density of the testing example with the output of EDL during the prediction stage, while using a novel parameterization that resolves the issues in the conventional parameterization. We prove that DAEDL enjoys a number of favorable theoretical properties. DAEDL demonstrates state-of-the-art performance across diverse downstream tasks related to uncertainty estimation and classification. 


