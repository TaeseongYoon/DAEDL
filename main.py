import argparse
import pickle
import math
from tqdm import tqdm  
import  warnings
warnings.filterwarnings('ignore')

import numpy as np
import numpy.random as npr
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import beta
from scipy.stats import dirichlet
from scipy.special import gammaln
from scipy.special import digamma
from scipy.stats import multivariate_normal as mvn
from sklearn.model_selection import train_test_split

from torch.distributions.dirichlet import Dirichlet
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.kl import kl_divergence as kl_div
from torch.nn.utils import spectral_norm
from torch.utils.data import DataLoader

import utility, train, density_estimation, ood_detection, conf_calibration 
from utility import *
from train import *
from density_estimation import *
from ood_detection import * 
from conf_calibration import *

import argparse
import json
import os

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--ID_dataset", default = "CIFAR-10", choices=["MNIST", "CIFAR-10", "CIFAR-100"], help="Pick a dataset")
    parser.add_argument("--batch_size", type=int, default = 64, help="Batch size to use for training")
    parser.add_argument("--val_size", type=float, default = 0.05, help="Size of the validation set")
    parser.add_argument("--val_seed", type=int, default = 99, help="Validation seed for training")
    parser.add_argument("--num_classes", type=int, default = 10, help="Number of classes in the data")
    parser.add_argument("--embedding_dim", type=int, default = 512, help="Dimension of the feature space")
    
    parser.add_argument("--learning_rate", type=float, default = 1e-3, help="Learning rate")
    parser.add_argument("--dropout_rate", type=float, default = 0.5, help="Dropout rate")
    parser.add_argument("--reg_param", type=float, default = 5e-2, help="Regularization parameter")
    parser.add_argument("--num_epochs", type=int, default = 100, help="Num Epochs")
    
    parser.add_argument("--index", type=int, default=0, help = "Index of the Pretrained Model")
    parser.add_argument("--device", type=str, default = "cuda:0", help="GPU")
    parser.add_argument("--output_dir", type=str, default = "saved_results", help="Directory to save the results")
    parser.add_argument("--pretrained", action='store_true', help="Using pretrained model")
    
    args = parser.parse_args()
    return args


def main(args):
    print("=================Load Dataset=================")
    trainloader, validloader, testloader, ood_loader1, ood_loader2 = load_datasets(args.ID_dataset, args.batch_size, args.val_size)
    print("=================Load Model=================")
    model = load_model(args.ID_dataset, args.pretrained, args.index, args.dropout_rate, args.device)   
    print("=================Train & Evaluate=================")
    train_daedl(model, args.learning_rate, args.reg_param, args.num_epochs, trainloader, validloader, args.num_classes, args.device)   
    test_acc = eval_daedl(model, testloader, args.device)  
    print("=================Density Estimation=================")
    gda, p_z_train = fit_gda(model, trainloader, args.num_classes, args.embedding_dim, args.device)
    print("=================OOD Detection=================")
    ood_auroc, ood_aupr = ood_detection_daedl(model, gda, p_z_train, testloader, ood_loader1, ood_loader2, args.num_classes,
                                              args.device)
    
    result = {"Test Acc": test_acc,"OOD AUROC": ood_auroc,"OOD AUPR": ood_aupr}
    
    if args.ID_dataset == "CIFAR-10":
        brier, conf_aupr, conf_auroc = conf_calibration_daedl(model, gda, p_z_train, testloader, args.num_classes, args.device)        
        result["Conf AUROC"] = conf_auroc
        result["Conf AUPR"] = conf_aupr
        
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    result_filepath = os.path.join(output_dir, 'results.json')
    
    with open(result_filepath, 'w') as result_file:
        json.dump(result, result_file, indent=4)
    

if __name__ == "__main__":
    args = parse_args()
    main(args)
