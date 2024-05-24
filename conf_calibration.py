"""
Code for Confidence Calibration
"""

import math
import numpy as np
import numpy.random as npr
import pandas as pd
import scipy
import matplotlib.pyplot as plt
from tqdm import tqdm  

import sklearn 

import torch

from utils.ensemble_utils import ensemble_forward_pass
from metrics.classification_metrics import get_logits_labels, get_logits_labels2
from metrics.uncertainty_confidence import entropy, logsumexp, confidence
import warnings
warnings.filterwarnings('ignore')

import density_estimation
from density_estimation import *

def conf_calibration_daedl(model, gda, p_z_train, testloader, num_classes, device):
    brier = 0
    cnt = 0

    Y = []
    PI = []
    ALPHA = []
    
    min, max = p_z_train.min().item(), p_z_train.max().item()

    with torch.no_grad():
        for i,(x,y) in enumerate(tqdm(testloader)):
            x,y = x.to(device), y.to(device)
            
            z = model(x)

            p_z = torch.Tensor(torch.logsumexp(gmm_forward(model, gda, x), dim = -1))

            p_z[p_z < min] = min
            p_z_norm = (p_z - min) / (max-min)
                                            
            alpha = torch.exp(z * p_z_norm.reshape(-1,1))
            pi = alpha / alpha.sum(1).reshape(-1,1)

            # i) Brier Score 
            cnt += x.shape[0]
            brier += torch.sum((F.one_hot(y, num_classes) - pi) ** 2)          

            Y.append(y)      
            ALPHA.append(torch.Tensor(alpha))     
            PI.append(torch.Tensor(pi))
       
        brier_score = (brier / cnt).item()

    labels = torch.cat(Y)
    prob = torch.cat(PI)
    alpha = torch.cat(ALPHA)
    
    correct = torch.Tensor(prob.argmax(1) == labels).cpu().numpy()
    scores_alea = prob.max(1).values.cpu().numpy()   
    scores_epis = alpha.max(1).values.cpu().numpy()
    
    aupr_alea =  sklearn.metrics.average_precision_score(correct, scores_alea)
    aupr_epis =  sklearn.metrics.average_precision_score(correct, scores_epis)
    
    auroc_alea = sklearn.metrics.roc_auc_score(correct, scores_alea)
    auroc_epis = sklearn.metrics.roc_auc_score(correct, scores_alea)
    
    conf_aupr = [aupr_alea, aupr_epis]
    conf_auroc = [auroc_alea, auroc_epis]

    return brier, conf_aupr, conf_auroc

