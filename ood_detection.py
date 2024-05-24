"""
Code for OOD Detection
"""

import math
import numpy as np
import numpy.random as npr
import pandas as pd
import scipy
import matplotlib.pyplot as plt
from tqdm import tqdm  

import sklearn 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

from scipy.stats import beta
from scipy.stats import dirichlet
from scipy.special import gammaln
from scipy.special import digamma
from scipy.stats import multivariate_normal as mvn

from torch.distributions.dirichlet import Dirichlet
from torch.distributions.kl import kl_divergence as kl_div

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.kl import kl_divergence as kl_div
from torch.nn.utils import spectral_norm
import torchvision
import torchvision.transforms as transforms

from utils.ensemble_utils import ensemble_forward_pass
from metrics.classification_metrics import get_logits_labels, get_logits_labels2
from metrics.uncertainty_confidence import entropy, logsumexp, confidence

import warnings
warnings.filterwarnings('ignore')

import utility, density_estimation
from utility import *
from density_estimation import *


def logsumexp(logits):
    return torch.logsumexp(logits, dim=1, keepdim=False)

def identity(x):
    return x

def check(x):
    nan = torch.sum(torch.isnan(torch.Tensor(x)))
    inf = torch.sum(torch.isinf(torch.Tensor(x)))
    
    if (inf + nan) !=0 : 
        #print("error")
        x = torch.nan_to_num(x)
        #print(x)
        
    return x

def entropy_density(logits, densities):
    
    weighted_logit = logits * densities.reshape(-1,1)
    p = F.softmax(weighted_logit, dim=1)
    logp = F.log_softmax(weighted_logit, dim=1)
    total_unc = -torch.sum(p * logp, dim=1)   
    return check(total_unc)

def alea_unc_density(logits, densities):

    weighted_logit = logits * densities.reshape(-1,1)
    p_star = F.softmax(weighted_logit, dim = 1)
    
    alpha_star = torch.exp(weighted_logit)
    alpha0_star = torch.sum(alpha_star, dim = 1)
    
    a = torch.digamma(alpha_star + 1) - torch.digamma(alpha0_star + 1).reshape(-1,1)
    alea_unc = -torch.sum(p_star * a, dim =1)   
    return check(alea_unc)

def maxP_density(logits, densities):
    weighted_logit = logits * densities.reshape(-1,1)        
    p = F.softmax(weighted_logit , dim = 1)
    max_p = p.max(1).values

    return check(max_p)

def max_alpha_density(logits, densities): 

    weighted_logit = logits * densities.reshape(-1,1)             
    alpha_star =  1e-6 + torch.exp(weighted_logit)                 
    max_alpha = alpha_star.max(1).values

    return check(max_alpha)

def alpha0_density(logits, densities):
    
    weighted_logit = logits * densities.reshape(-1,1)
    alpha_star = 1e-6 + torch.exp(weighted_logit)
    alpha0 = torch.sum(alpha_star, dim = 1) 
    
    return check(alpha0)


def dist_unc_density(logits, densities):
    return check(entropy_density(logits, densities) - alea_unc_density(logits, densities))

def max_logits_density(logits, densities):
    
    weighted_logits = logits * densities.reshape(-1,1)

    return check(weighted_logits.max(1).values)



def auroc_aupr(net, test_loader, ood_test_loader, train_density, id_density, ood_density, uncertainty, device):
    
    logits, _ = get_logits_labels(net, test_loader, device)
    ood_logits, _ = get_logits_labels(net, ood_test_loader, device)
    
    min = train_density.min()
    max = train_density.max()

    ood_density[ood_density < min] = min
    ood_density[ood_density > max] = max

    id_density[id_density < min] = min
    id_density[id_density > max] = max

    id_norm_density = (id_density - min) / (max - min)
    ood_norm_density = (ood_density - min) / (max - min) 
    train_density_norm = (train_density - min) / (max - min)

    uncertainties = uncertainty(logits, id_norm_density)
    ood_uncertainties = uncertainty(ood_logits, ood_norm_density)

    bin_labels = torch.ones(uncertainties.shape[0]).to(device)
    bin_labels = torch.cat((bin_labels, torch.zeros(ood_uncertainties.shape[0]).to(device)))
                      
    in_scores = uncertainties
    ood_scores = ood_uncertainties
        
    print(uncertainty)
    print("ID Scores (Density):",in_scores)
    print("OOD Scores (Density):",ood_scores)
  
    scores = torch.cat((in_scores, ood_scores))
    auroc = sklearn.metrics.roc_auc_score(bin_labels.cpu().numpy(), scores.cpu().numpy())
    aupr = sklearn.metrics.average_precision_score(bin_labels.cpu().numpy(), scores.cpu().numpy())

    return auroc, aupr


def ood_detection_daedl(model, gda, p_z_train, testloader, ood_loader1, ood_loader2, num_classes, device):     
    logits_edl, _ = gmm_evaluate(model, gda, testloader, device, num_classes, device)
    ood_logits_edl1, _ = gmm_evaluate(model, gda, ood_loader1, device, num_classes,device)
    ood_logits_edl2, _ = gmm_evaluate(model, gda, ood_loader2, device, num_classes,device)
    
    p_z_test = check(torch.logsumexp(logits_edl, dim = -1))
    p_z_ood1 = check(torch.logsumexp(ood_logits_edl1, dim = -1))  
    p_z_ood2 = check(torch.logsumexp(ood_logits_edl2, dim = -1))
          
    auroc_alea1, aupr_alea1 = auroc_aupr(model, testloader, ood_loader1, p_z_train, p_z_test, p_z_ood1, maxP_density, device)   
    auroc_alea2, aupr_alea2 = auroc_aupr(model, testloader, ood_loader2, p_z_train, p_z_test, p_z_ood2, maxP_density, device)  
    auroc_epis1, aupr_epis1 = auroc_aupr(model, testloader, ood_loader1, p_z_train, p_z_test, p_z_ood1, alpha0_density, device)
    auroc_epis2, aupr_epis2 = auroc_aupr(model, testloader, ood_loader2, p_z_train, p_z_test, p_z_ood2, alpha0_density, device)

    ## AUROC
    OOD1 = {}
    OOD1["MaxP"] = auroc_alea1
    OOD1["Alpha0"] = auroc_epis1
     
    OOD2 = {}
    OOD2["MaxP"] = auroc_alea2
    OOD2["Alpha0"] = auroc_epis2
    
    ## AUPR
    OOD1_ = {}
    OOD1_["MaxP"] = aupr_alea1
    OOD1_["Alpha0"] = aupr_epis1

    OOD2_ = {}
    OOD2_["MaxP"] = aupr_alea2
    OOD2_["Alpha0"] = aupr_epis2
       
    AUROC = [OOD1, OOD2]
    AUPR = [OOD1_, OOD2_]   

    return AUROC, AUPR

