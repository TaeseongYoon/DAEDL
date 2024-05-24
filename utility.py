"""
Code for loading the datasets/architectures
"""

import math
import numpy as np
import torch.utils.data as data
from torch.utils.data import DataLoader
from torch.utils.data import Subset

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

import torchvision
import torchvision.transforms as transforms
from torchvision import datasets

from net.spectral_normalization.spectral_norm_conv_inplace import spectral_norm_conv
from net.spectral_normalization.spectral_norm_fc import spectral_norm_fc


def load_model(ID_dataset, pretrained, index, dropout_rate, device):
    if ID_dataset == "MNIST":
        if pretrained : 
            model = conv_net()     
            model.load_state_dict(torch.load(f"saved_models/mnist_conv_daedl_{index+1}.pt" ))
        else : 
            model = conv_net()             
            
    if ID_dataset == "CIFAR-10":
        if pretrained : 
            model = vgg16(dropout_rate)
            model.load_state_dict(torch.load(f"/saved_models/cifar10_vgg_daedl_{index+1}"))
            
        else :
            model = vgg16(dropout_rate)   
            
    if ID_dataset == "CIFAR-100":
        if pretrained : 
            pass
        else : 
            model = resnet()         
    model.to(device)
            
    return model

# Load Datasets
def load_datasets(ID_dataset, batch_size, val_size):
    if ID_dataset == "MNIST":
        trainloader, validloader, testloader, ood_loader1, ood_loader2 = dataloaders_mnist(batch_size, val_size)       
    if ID_dataset == "CIFAR-10":
        trainloader, validloader, testloader, ood_loader1, ood_loader2 = dataloaders_cifar10(batch_size, val_size)       
    if ID_dataset == "CIFAR-100":
        trainloader, validloader, testloader, ood_loader1, ood_loader2 = dataloaders_cifar100(batch_size, val_size)
            
    return trainloader, validloader, testloader, ood_loader1, ood_loader2


def dataloaders_mnist(batch_size, val_size):
    root = "/data"
    transform = transforms.ToTensor()
    
    mnist_trainval_dataset = datasets.MNIST(root = root, train = True, download=True, transform=transform)
    mnist_test_dataset = datasets.MNIST(root = root, train = False, download=True, transform=transform)
    mnist_train_dataset, mnist_val_dataset = torch.utils.data.random_split(mnist_trainval_dataset, [48000, 12000])
    fmnist_dataset = datasets.FashionMNIST(root = root, download=True, train = False, transform = transform)
    kmnist_dataset = datasets.KMNIST(root = root, download=True, train = False, transform = transform)
    
    mnist_trainloader = DataLoader(mnist_train_dataset, shuffle = True, batch_size = batch_size)
    mnist_validloader = DataLoader(mnist_val_dataset, shuffle = True, batch_size = batch_size)
    mnist_testloader = DataLoader(mnist_test_dataset, shuffle = True, batch_size = batch_size)
    fmnist_loader = DataLoader(fmnist_dataset, batch_size = batch_size, shuffle = True)
    kmnist_loader = DataLoader(kmnist_dataset, batch_size = batch_size, shuffle = True)

    return mnist_trainloader, mnist_validloader, mnist_testloader, fmnist_loader, kmnist_loader


def dataloaders_cifar10(batch_size, val_size):
    num_workers = 4
    root = "./data"
    
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010],)
    
    train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),transforms.RandomCrop(32, 4), transforms.RandomRotation(degrees=15),transforms.ToTensor(), normalize,])
    valid_transform = transforms.Compose([transforms.ToTensor(), normalize,])
    test_transform = transforms.Compose([transforms.ToTensor(), normalize,])
      
    cifar10_train_dataset = datasets.CIFAR10(root = root, train = True, transform = train_transform)   
    cifar10_valid_dataset = datasets.CIFAR10(root = root, train = True, transform = valid_transform)
    
    num_train = len(cifar10_train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(val_size * num_train))
    np.random.shuffle(indices)

    cifar10_train_dataset = Subset(cifar10_train_dataset, indices[split:])
    cifar10_valid_dataset = Subset(cifar10_valid_dataset, indices[:split])
    cifar10_test_dataset = datasets.CIFAR10(root = root, train = False, download=True, transform = test_transform)   
    svhn_dataset = datasets.SVHN(root = root, split="test", download=True, transform = test_transform)
    cifar100_dataset = datasets.CIFAR100(root = root, train = False, download=True, transform = test_transform)

    cifar10_trainloader = DataLoader(cifar10_train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    cifar10_validloader = DataLoader(cifar10_valid_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False) 
    cifar10_testloader = DataLoader(cifar10_test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle = False)
    svhn_loader = DataLoader(svhn_dataset, batch_size=batch_size, num_workers = num_workers, shuffle=False)
    cifar100_loader = DataLoader(cifar100_dataset, batch_size=batch_size, num_workers = num_workers, shuffle = False)

    return cifar10_trainloader, cifar10_validloader, cifar10_testloader, svhn_loader, cifar100_loader


def dataloaders_cifar100(batch_size, val_size):
    num_workers = 4
    root = "./data"    
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010],)
    resize = transforms.Resize((32, 32))
    tensorize = transforms.ToTensor()
  
    train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),transforms.RandomCrop(32, 4),
                                          transforms.RandomRotation(degrees=15), tensorize, normalize])
    valid_transform = transforms.Compose([tensorize, normalize])
    test_transform = transforms.Compose([tensorize, normalize])
    mnist2cifar = transforms.Compose([resize, transforms.Grayscale(3)])
    tin2cifar = transforms.Compose([resize, tensorize])
      
    cifar100_train_dataset = datasets.CIFAR100(root = root, train = True, transform = train_transform)   
    cifar100_valid_dataset = datasets.CIFAR100(root = root, train = True, transform = valid_transform)
    
    num_train = len(cifar100_train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(val_size * num_train))
    np.random.shuffle(indices)
    
    cifar100_train_dataset = Subset(cifar100_train_dataset, indices[split:])
    cifar100_valid_dataset = Subset(cifar100_valid_dataset, indices[:split])
    cifar100_test_dataset = datasets.CIFAR10(root = root, train = False, download=True, transform = test_transform)   
    
    svhn_dataset = datasets.SVHN(root = root, split="test", download=True)
    fmnist_test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform = mnist2cifar)
    tin_test_dataset = datasets.ImageFolder(root="data/tiny-imagenet-200/test", transform = tin2cifar)

    cifar100_trainloader = DataLoader(cifar100_train_dataset, batch_size = batch_size, num_workers = num_workers, shuffle=True)
    cifar100_validloader = DataLoader(cifar100_valid_dataset, batch_size = batch_size, num_workers = num_workers, shuffle=False) 
    cifar100_testloader =  DataLoader(cifar100_test_dataset, batch_size=batch_size, num_workers = num_workers, shuffle = False)
    
    svhn_loader = DataLoader(svhn_dataset, batch_size = batch_size, num_workers = num_workers)
    fmnist_loader = DataLoader(fmnist_test_dataset, batch_size = batch_size, num_workers = num_workers)
    tin_loader = DataLoader(tin_test_dataset, batch_size = batch_size, num_workers = num_workers)

    return cifar100_trainloader, cifar100_validloader, cifar100_testloader, svhn_loader, tin_loader



class AvgPoolShortCut(nn.Module):
    def __init__(self, stride, out_c, in_c):
        super(AvgPoolShortCut, self).__init__()
        self.stride = stride
        self.out_c = out_c
        self.in_c = in_c

    def forward(self, x):
        if x.shape[2] % 2 != 0:
            x = F.avg_pool2d(x, 1, self.stride)
        else:
            x = F.avg_pool2d(x, self.stride, self.stride)
        pad = torch.zeros(x.shape[0], self.out_c - self.in_c, x.shape[2], x.shape[3], device=x.device,)
        x = torch.cat((x, pad), dim=1)
        return x



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, input_size, wrapped_conv, in_planes, planes, stride=1, mod=True):
        super(BasicBlock, self).__init__()
        self.conv1 = wrapped_conv(input_size, in_planes, planes, kernel_size=3, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = wrapped_conv(math.ceil(input_size / stride), planes, planes, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.mod = mod
        self.activation = F.leaky_relu if self.mod else F.relu

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            if mod:
                self.shortcut = nn.Sequential(AvgPoolShortCut(stride, self.expansion * planes, in_planes))
            else:
                self.shortcut = nn.Sequential(
                    wrapped_conv(input_size, in_planes, self.expansion * planes, kernel_size=1, stride=stride,),
                    nn.BatchNorm2d(planes),
                )

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.activation(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, input_size, wrapped_conv, in_planes, planes, stride=1, mod=True):
        super(Bottleneck, self).__init__()
        self.conv1 = wrapped_conv(input_size, in_planes, planes, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = wrapped_conv(input_size, planes, planes, kernel_size=3, stride=stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = wrapped_conv(math.ceil(input_size / stride), planes, self.expansion * planes, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.mod = mod
        self.activation = F.leaky_relu if self.mod else F.relu

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            if mod:
                self.shortcut = nn.Sequential(AvgPoolShortCut(stride, self.expansion * planes, in_planes))
            else:
                self.shortcut = nn.Sequential(
                    wrapped_conv(input_size, in_planes, self.expansion * planes, kernel_size=1, stride=stride,),
                    nn.BatchNorm2d(self.expansion * planes),
                )

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.activation(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.activation(out)
        return out



class ResNet(nn.Module):
    def __init__(
        self,
        block,
        num_blocks,
        num_classes,
        temp=1.0,
        spectral_normalization=True,
        mod=True,
        coeff=3,
        n_power_iterations=1,
        mnist=False,
    ):
        
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.mod = mod

        def wrapped_conv(input_size, in_c, out_c, kernel_size, stride):
            padding = 1 if kernel_size == 3 else 0

            conv = nn.Conv2d(in_c, out_c, kernel_size, stride, padding, bias=False)

            if not spectral_normalization:
                return conv

            # NOTE: Google uses the spectral_norm_fc in all cases
            if kernel_size == 1:
                # use spectral norm fc, because bound are tight for 1x1 convolutions
                wrapped_conv = spectral_norm_fc(conv, coeff, n_power_iterations)
            else:
                # Otherwise use spectral norm conv, with loose bound
                shapes = (in_c, input_size, input_size)
                wrapped_conv = spectral_norm_conv(conv, coeff, shapes, n_power_iterations)

            return wrapped_conv

        self.wrapped_conv = wrapped_conv

        self.bn1 = nn.BatchNorm2d(64)

        if mnist:
            self.conv1 = wrapped_conv(28, 1, 64, kernel_size=3, stride=1)
            self.layer1 = self._make_layer(block, 28, 64, num_blocks[0], stride=1)
            self.layer2 = self._make_layer(block, 28, 128, num_blocks[1], stride=2)
            self.layer3 = self._make_layer(block, 14, 256, num_blocks[2], stride=2)
            self.layer4 = self._make_layer(block, 7, 512, num_blocks[3], stride=2)
        else:
            self.conv1 = wrapped_conv(32, 3, 64, kernel_size=3, stride=1)
            self.layer1 = self._make_layer(block, 32, 64, num_blocks[0], stride=1)
            self.layer2 = self._make_layer(block, 32, 128, num_blocks[1], stride=2)
            self.layer3 = self._make_layer(block, 16, 256, num_blocks[2], stride=2)
            self.layer4 = self._make_layer(block, 8, 512, num_blocks[3], stride=2)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.activation = F.leaky_relu if self.mod else F.relu
        self.feature = None
        self.temp = temp

    def _make_layer(self, block, input_size, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(input_size, self.wrapped_conv, self.in_planes, planes, stride, self.mod,))
            self.in_planes = planes * block.expansion
            input_size = math.ceil(input_size / stride)
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        self.feature = out.clone().detach()
        out = self.fc(out) / self.temp
        return out


def resnet18(num_classes, spectral_normalization=True, mod=True, temp=1.0, mnist=False, **kwargs):
    model = ResNet(num_classes, 
        BasicBlock,
        [2, 2, 2, 2],
        spectral_normalization=spectral_normalization,
        mod=mod,
        temp=temp,
        mnist=mnist,
        **kwargs
    )
    return model



class SpectralLinear(nn.Module):
    def __init__(self, input_dim, output_dim, k_lipschitz=1.0):
        super().__init__()
        self.k_lipschitz = k_lipschitz
        self.spectral_linear = spectral_norm(nn.Linear(input_dim, output_dim))

    def forward(self, x):
        y = self.k_lipschitz * self.spectral_linear(x)
        return y


class SpectralConv(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_dim, padding, k_lipschitz=1.0):
        super().__init__()
        self.k_lipschitz = k_lipschitz
        self.spectral_conv = spectral_norm(nn.Conv2d(input_dim, output_dim, kernel_dim, padding=padding))

    def forward(self, x):
        y = self.k_lipschitz * self.spectral_conv(x)
        return y


def linear_sequential(input_dims, hidden_dims, output_dim, k_lipschitz=None, p_drop=None):
    dims = [np.prod(input_dims)] + hidden_dims + [output_dim]
    num_layers = len(dims) - 1
    layers = []
    for i in range(num_layers):
        if k_lipschitz is not None:
            l = SpectralLinear(dims[i], dims[i + 1], k_lipschitz ** (1./num_layers))
            layers.append(l)
        else:
            layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < num_layers - 1:
            layers.append(nn.ReLU())
            if p_drop is not None:
                layers.append(nn.Dropout(p=p_drop))
    return nn.Sequential(*layers)

def convolution_sequential(input_dims, hidden_dims, output_dim, kernel_dim, k_lipschitz=None, p_drop=None):
    channel_dim = input_dims[2]
    dims = [channel_dim] + hidden_dims
    num_layers = len(dims) - 1
    layers = []
    for i in range(num_layers):
        if k_lipschitz is not None:
            l = SpectralConv(dims[i], dims[i + 1], kernel_dim, (kernel_dim - 1) // 2, k_lipschitz ** (1./num_layers))
            layers.append(l)
        else:
            layers.append(nn.Conv2d(dims[i], dims[i + 1], kernel_dim, padding=(kernel_dim - 1) // 2))
        layers.append(nn.ReLU())
        if p_drop is not None:
            layers.append(nn.Dropout(p=p_drop))
        layers.append(nn.MaxPool2d(2, padding=0))
    return nn.Sequential(*layers)


class ConvLinSeq(nn.Module):
    def __init__(self, input_dims, linear_hidden_dims, conv_hidden_dims, output_dim, kernel_dim, batch_size, k_lipschitz, p_drop):
        super().__init__()
        if k_lipschitz is not None:
            k_lipschitz = k_lipschitz ** (1./2.)
        self.convolutions = convolution_sequential(input_dims=input_dims,
                                                   hidden_dims=conv_hidden_dims,
                                                   output_dim=output_dim,
                                                   kernel_dim=kernel_dim,
                                                   k_lipschitz=k_lipschitz,
                                                   p_drop=p_drop)
        
        # We assume that conv_hidden_dims is a list of same hidden_dim values
        self.linear = linear_sequential(input_dims=[conv_hidden_dims[-1] * (input_dims[0] // 2 ** len(conv_hidden_dims)) * (input_dims[1] // 2 ** len(conv_hidden_dims))],
                                        hidden_dims=linear_hidden_dims,
                                        output_dim=output_dim,
                                        k_lipschitz=k_lipschitz,
                                        p_drop=p_drop)

    def forward(self, input):
        batch_size = input.size(0)
        input = self.convolutions(input)
        self.feature = input.clone().detach().reshape(batch_size,-1)

        input = self.linear(input.reshape(batch_size, -1))
        return input


def convolution_linear_sequential(input_dims, linear_hidden_dims, conv_hidden_dims, output_dim, kernel_dim, batch_size, k_lipschitz, p_drop=None):
    return ConvLinSeq(input_dims=input_dims,
                      linear_hidden_dims=linear_hidden_dims,
                      conv_hidden_dims=conv_hidden_dims,
                      output_dim=output_dim,
                      kernel_dim=kernel_dim, batch_size = batch_size,
                      k_lipschitz=k_lipschitz,
                      p_drop=p_drop)



class VGG(nn.Module):
    '''
    VGG model
    '''
    def __init__(self, features, output_dim, k_lipschitz=None, p_drop=None):
        super(VGG, self).__init__()
        self.features = features
        if k_lipschitz is not None:
            l_1, l_2, l_3 = SpectralLinear(512, 512, k_lipschitz), SpectralLinear(512, 512, k_lipschitz), SpectralLinear(512, output_dim, k_lipschitz)
            
            self.classifier = nn.Sequential(
                nn.Dropout(p=p_drop),
                l_1,
                nn.ReLU(True),
                nn.Dropout(p=p_drop),
                l_2,
                nn.ReLU(True),
                l_3,
            )
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(p=p_drop),
                nn.Linear(512, 512),
                nn.ReLU(True),
                nn.Dropout(p=p_drop),
                nn.Linear(512, 512),
                nn.ReLU(True),
                nn.Linear(512, output_dim),
            )
         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

        

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        self.feature = x.clone().detach().reshape(x.shape[0],-1)
        x = self.classifier(x)
        return x


def make_layers(cfg, batch_norm=False, k_lipschitz=None):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            if k_lipschitz is not None:
                conv2d = SpectralConv(in_channels, v, kernel_dim=3, padding=1, k_lipschitz=k_lipschitz)
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
          512, 512, 512, 512, 'M'],
}




def vgg16_bn(output_dim, k_lipschitz=None, p_drop=.5):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    if k_lipschitz is not None:
        k_lipschitz = k_lipschitz ** (1. / 16.)
    return VGG(make_layers(cfg['D'], batch_norm=True, k_lipschitz=k_lipschitz),
               output_dim=output_dim,
               k_lipschitz=k_lipschitz,
               p_drop=p_drop)
    
    
# Main Architectures : ConvNet / VGG16 / ResNet18
def conv_net():        
    input_dims = [28, 28, 1]
    linear_hidden_dims =[64, 64, 64]
    conv_hidden_dims = [64, 64, 64]
    output_dim = 10
    kernel_dim = 5
    k_lipschitz = 1
    batch_size = 64

    return convolution_linear_sequential(input_dims, linear_hidden_dims, conv_hidden_dims, output_dim, kernel_dim, batch_size, k_lipschitz, p_drop=None)

def vgg16(p_drop):               
    output_dim = 10
    k_lipschitz = 1

    return vgg16_bn(output_dim = 10, k_lipschitz = 1, p_drop = p_drop)


def resnet(num_classes):
    return resnet18(num_classes, spectral_normalization = True, mod=True, temp=1.0, mnist=False)


"""
def vgg16(output_dim, k_lipschitz=None, p_drop=.5):
    VGG 16-layer model (configuration "D")
    if k_lipschitz is not None:
        k_lipschitz = k_lipschitz ** (1. / 16.)
    return VGG(make_layers(cfg['D'], k_lipschitz=k_lipschitz),
               output_dim=output_dim,
               k_lipschitz=k_lipschitz,
               p_drop=p_drop)
               
"""