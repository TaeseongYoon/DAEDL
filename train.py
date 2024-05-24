"""
Code for training & evaluation of the model
"""

import os
import math
import numpy as np
import numpy.random as npr
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm  
import  warnings
warnings.filterwarnings('ignore')
from itertools import cycle

from scipy.stats import beta
from scipy.stats import dirichlet
from scipy.special import gammaln
from scipy.special import digamma
from scipy.stats import multivariate_normal as mvn

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.dirichlet import Dirichlet
from torch.distributions.kl import kl_divergence as kl_div
from torch.nn.utils import spectral_norm
from torch.utils.data import DataLoader, Dataset


def train_daedl(model, learning_rate, reg_param, num_epochs, trainloader, validloader, num_classes, device):
    
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda epoch: 0.95 ** epoch)  
    
    VAL_ACC = []
    VAL_LOSS = []
    cnt = 0

    model.to(device)        
    model.train()
    for epoch in tqdm(range(num_epochs)):
        running_loss = 0.0
          
        for i, (x,y) in enumerate(trainloader):
            optimizer.zero_grad()
            x,y = x.to(device), y.to(device)
          
            alpha = 1e-6 + torch.exp(model(x))                                                         
            alpha0 = alpha.sum(1).reshape(-1,1)
            y_oh = F.one_hot(y, num_classes).to(device)
            alpha_tilde = alpha * (1-y_oh) + y_oh
                
            expected_mse = torch.sum((y_oh - alpha / alpha0) ** 2 ) + torch.sum(((alpha * (alpha0 - alpha))) / ((alpha0 ** 2) * (alpha0 + 1)))                                                                           
            kl_regularizer = kl_div(Dirichlet(1e-6 + alpha_tilde), Dirichlet(torch.ones_like(alpha_tilde))).sum() 
            loss = expected_mse + reg_param * kl_regularizer
                
            loss.backward()      
            optimizer.step()
            running_loss += loss.item()    
            
        scheduler.step()
        
        if epoch % 10 == 0 and epoch > 0:
        
            total=0
            correct=0
            val_loss = 0

            with torch.no_grad():
                for i, (x_v,y_v) in enumerate(validloader):
                    x_v, y_v = x_v.to(device), y_v.to(device)

                    alpha_v= 1e-6 + torch.exp(model(x_v))
                    alpha0_v = alpha_v.sum(1).reshape(-1,1)
                    y_oh_v = F.one_hot(y_v, num_classes).to(device)  
                    alpha_v_tilde = alpha_v * (1-y_oh_v) + y_oh_v
                    
                    expected_mse_v = torch.sum((y_oh_v - alpha_v/ alpha0_v) ** 2 ) + torch.sum(((alpha_v * (alpha0_v- alpha_v))) / ((alpha0_v ** 2) * (alpha0_v + 1)))
                    kl_regularizer_v = kl_div(Dirichlet(alpha_v_tilde), Dirichlet(torch.ones_like(alpha_v_tilde))).sum()
                    
                    val_loss += expected_mse_v + reg_param * kl_regularizer_v
                    
                    y_pred_v = alpha_v.argmax(1)
                    
                    total += y_v.size(0)
                    correct += (y_pred_v == y_v.to(device)).sum().item()

            val_acc = 100*correct/total
            VAL_LOSS.append(val_loss)
            VAL_ACC.append(val_acc)
            
            if len(VAL_ACC) > 2 : 
                
                r_acc = (VAL_ACC[-1] - VAL_ACC[-2]) / VAL_ACC[-2]
                r_loss = (VAL_LOSS[-1] - VAL_LOSS[2]) / VAL_LOSS[-2]

                if r_loss > -0.0001 :
                    cnt = cnt + 1
                else : 
                    cnt = 0
                    
            if cnt > 3 :
                break
                
            print('Epoch {}, loss = {:.3f}'.format(epoch, val_loss)) 
            print('Validation Accuracy = {:.3f}'.format(val_acc))
                 
def eval_daedl(model, testloader, device):    
    model.eval()
    total=0
    correct=0
    
    with torch.no_grad():
        for i, (x,y) in enumerate(testloader):
            x,y = x.to(device), y.to(device)
            alpha_pred = torch.exp(model(x))
            y_pred = alpha_pred.argmax(1)
            
            total += y.size(0)
            correct += (y_pred == y).sum().item()
            
        test_acc = 100*correct/total
        print("Test Accuracy:",test_acc)
    
    return test_acc