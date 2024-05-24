"""
Code for Density Estimation
"""

import math
import numpy as np
import numpy.random as npr
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm  

from sklearn.datasets import make_moons
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from scipy.stats import beta
from scipy.stats import dirichlet
from scipy.special import gammaln
from scipy.special import digamma
from scipy.stats import multivariate_normal as mvn

import torch
from torch.distributions.dirichlet import Dirichlet
from torch.distributions.kl import kl_divergence as kl_div

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.kl import kl_divergence as kl_div
from torch.nn.utils import spectral_norm

import torchvision
import torchvision.transforms as transforms

import  warnings
warnings.filterwarnings('ignore')

def check(x):
    nan = torch.sum(torch.isnan(torch.Tensor(x)))
    inf = torch.sum(torch.isinf(torch.Tensor(x)))
    
    if (inf + nan) !=0 : 
        print("error")
        x = torch.nan_to_num(x)
        print(x)
        
    return x


DOUBLE_INFO = torch.finfo(torch.double)
JITTERS = [0, DOUBLE_INFO.tiny] + [10 ** exp for exp in range(-308, 0, 1)]


def centered_cov_torch(x):
    n = x.shape[0]
    res = 1 / (n - 1) * x.t().mm(x)
    return res


def get_embeddings(net, loader: torch.utils.data.DataLoader, num_dim: int, dtype, device, storage_device):

    num_samples = len(loader.dataset)
    embeddings = torch.empty((num_samples, num_dim), dtype=dtype, device=storage_device)
    
    labels = torch.empty(num_samples, dtype=torch.int, device=storage_device)

    with torch.no_grad():
        start = 0
        for data, label in tqdm(loader):
            data = data.to(device)
            label = label.to(device)

            if isinstance(net, nn.DataParallel):
                out = net.module(data)
                out = net.module.feature
            else:
                out = net(data)
                out = net.feature
            
            end = start + len(data)
            embeddings[start:end].copy_(out, non_blocking=True)
            labels[start:end].copy_(label, non_blocking=True)
            start = end

    return embeddings, labels


def gmm_forward(net, gaussians_model, data_B_X):

    if isinstance(net, nn.DataParallel):
        features_B_Z = net.module(data_B_X)
        features_B_Z = net.module.feature
    else:
        features_B_Z = net(data_B_X)
        features_B_Z = net.feature

    log_probs_B_Y = gaussians_model.log_prob(features_B_Z[:, None, :])

    return log_probs_B_Y


def gmm_evaluate(net, gaussians_model, loader, device, num_classes, storage_device):
    num_samples = len(loader.dataset)
    logits_N_C = torch.empty((num_samples, num_classes), dtype=torch.float, device=storage_device)
    labels_N = torch.empty(num_samples, dtype=torch.int, device=storage_device)

    with torch.no_grad():
        start = 0
        for data, label in tqdm(loader):
            data = data.to(device)
            label = label.to(device)

            logit_B_C = gmm_forward(net, gaussians_model, data)

            end = start + len(data)
            logits_N_C[start:end].copy_(logit_B_C, non_blocking=True)
            labels_N[start:end].copy_(label, non_blocking=True)
            start = end

    return logits_N_C, labels_N


def gmm_evaluate2(net, gaussians_model, loader, device, num_classes, storage_device):

    num_samples = len(loader.dataset)
    logits_N_C = torch.empty((num_samples, num_classes), dtype=torch.float, device=storage_device)
    labels_N = torch.empty(num_samples, dtype=torch.int, device=storage_device)

    with torch.no_grad():
        start = 0
        for a in tqdm(loader):
            data = torch.Tensor(a["images"].to(torch.float) /255.0).reshape(-1,1,28,28).to(device)
            label = torch.Tensor(a["labels"]).reshape(-1,).to(device)

            logit_B_C = gmm_forward(net, gaussians_model, data)

            end = start + len(data)
            logits_N_C[start:end].copy_(logit_B_C, non_blocking=True)
            labels_N[start:end].copy_(label, non_blocking=True)
            start = end

    return logits_N_C, labels_N


def gmm_get_logits(gmm, embeddings):

    log_probs_B_Y = gmm.log_prob(embeddings[:, None, :])
    return log_probs_B_Y


def gmm_fit(embeddings, labels, num_classes):

    JITTERS = [0, DOUBLE_INFO.tiny] + [10 ** exp for exp in range(-308, 0, 1)]

    with torch.no_grad():
        classwise_mean_features = torch.stack([torch.mean(embeddings[labels == c], dim=0) for c in range(num_classes)])
        classwise_cov_features = torch.stack(
            [centered_cov_torch(embeddings[labels == c] - classwise_mean_features[c]) for c in range(num_classes)]
        )
    
    with torch.no_grad():
        for jitter_eps in JITTERS:
            try:
                jitter = jitter_eps * torch.eye(classwise_cov_features.shape[1], device=classwise_cov_features.device,).unsqueeze(0)
                gmm = torch.distributions.MultivariateNormal(loc=classwise_mean_features, covariance_matrix=(classwise_cov_features + jitter),)
                
            except RuntimeError as e:
                if "cholesky" in str(e):
                    continue

            except ValueError as e:
                if "Expected parameter covariance_matrix" in str(e):
                    continue
            
            break

    return gmm, jitter_eps


def fit_gda(model, trainloader, num_classes, embedding_dim, device):
    embeddings, labels = get_embeddings(model, trainloader, embedding_dim, torch.double, device, device) 
    gda, jitter_eps = gmm_fit(embeddings, labels, num_classes)  
    train_log_probs, _ = gmm_evaluate(model, gda, trainloader, device, num_classes, device)
    p_z_train = torch.logsumexp(train_log_probs, dim=-1)

    return gda, p_z_train


"""


def get_density_id(id_loader, num_classes, feature_extractor, device):
    embeddings, labels = get_embeddings(feature_extractor, id_loader, 128, torch.double, device, device)
    
    gda, jitter_eps = gmm_fit(embeddings, labels, num_classes)

    train_log_probs_B_Y, train_labels = gmm_evaluate(feature_extractor, gda, id_loader, device, num_classes, device)
    train_densities = torch.logsumexp(train_log_probs_B_Y, dim=-1)
    
    return gda, train_densities


def get_density_ood(ood_loader, num_classes, feature_extractor, gda, device):
    train_log_probs_B_Y_ood, train_labels_ood1 = gmm_evaluate(feature_extractor, gda, ood_loader, device, 32, device)
    train_densities_ood = torch.logsumexp(train_log_probs_B_Y_ood, dim=-1)

    return train_densities_ood


def get_feature_space_density(feature_extractor, id_loader, ood_loader1, ood_loader2, nid_loader, device, num_classes, plot_density, norm):
    gda, train_densities = get_density_id(id_loader, num_classes, feature_extractor, device)
    train_densities_ood1 = get_density_ood(ood_loader1, num_classes, feature_extractor, gda, device)
    train_densities_ood2 = get_density_ood(ood_loader2, num_classes, feature_extractor, gda, device)
    
    if plot_density == True and norm == False:
        plt.figure()
        plt.title("Feature Space Density")
        plt.hist(train_densities.cpu().detach().numpy(), label ="ID : MNIST")
        plt.hist(train_densities_ood1.cpu().detach().numpy(), label = "OOD1 : FMNIST")
        plt.hist(train_densities_ood2.cpu().detach().numpy(), label = "OOD2 : KMNIST")
    
        plt.legend()

    if plot_density == True and norm == True:
        plt.figure()
        plt.title("Normalized Feature Space Density")
        plt.hist(np.log(-train_densities.cpu().detach().numpy()), label ="ID : MNIST")
        plt.hist(np.log(-train_densities_ood1.cpu().detach().numpy()), label = "OOD1 : FMNIST")
        plt.hist(np.log(-train_densities_ood2.cpu().detach().numpy()), label = "OOD2 : KMNIST")
    
        plt.legend()

    return gda


def get_density_edl_id(id_loader, num_classes, embedding_dim, feature_extractor_edl, device):

    embeddings_edl, labels_edl = get_embeddings(feature_extractor_edl, id_loader, embedding_dim, torch.double, device, device) 
    print("Embedding Dim:",embeddings_edl.shape)
    gda_edl, jitter_eps_edl = gmm_fit(embeddings_edl, labels_edl, num_classes)

    train_log_probs_B_Y_edl, train_labels_edl = gmm_evaluate(feature_extractor_edl, gda_edl, id_loader, device, num_classes, device)
    train_densities_edl = torch.logsumexp(train_log_probs_B_Y_edl, dim=-1)

    return gda_edl, train_densities_edl


def get_density_edl_ood(ood_loader, num_classes, feature_extractor_edl, gda_edl, device):
    train_log_probs_B_Y_ood_edl, train_labels_ood_edl = gmm_evaluate(feature_extractor_edl, gda_edl, ood_loader, device, num_classes, device)
    train_densities_ood_edl = torch.logsumexp(train_log_probs_B_Y_ood_edl, dim=-1)

    return train_densities_ood_edl
"""