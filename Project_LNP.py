import os
import glob
import cv2
import numpy as np

import matplotlib.pyplot as plt
# import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics

import warnings
warnings.filterwarnings('ignore')
import math
#import utils

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from collections import OrderedDict

import argparse
from tqdm import tqdm
import scipy.io as sio
from skimage import img_as_ubyte
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

print('Successfully import all requirements!')

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".jpg"])


def load_img(filepath):
    img = cv2.imread(filepath)
    img = img.astype(np.float32)
    img = img/255.
    return img


class DataLoaderVal(Dataset):
    def __init__(self, rgb_dir, target_transform=None):
        super(DataLoaderVal, self).__init__()

        self.target_transform = target_transform

        real_files = sorted(os.listdir(os.path.join(rgb_dir, '0_real')))
        fake_files = sorted(os.listdir(os.path.join(rgb_dir, '1_fake')))
        
        self.real_filenames = [os.path.join(rgb_dir, '0_real', x) for x in real_files if is_image_file(x)]
        self.fake_filenames = [os.path.join(rgb_dir, '1_fake', x) for x in fake_files if is_image_file(x)]
        

        self.tar_size = len(self.real_filenames)  

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index = index % self.tar_size
        

        real = torch.from_numpy(np.float32(load_img(self.real_filenames[tar_index])))
        fake = torch.from_numpy(np.float32(load_img(self.fake_filenames[tar_index])))
        
        real_filename = os.path.split(self.real_filenames[tar_index])[-1]
        fake_filename = os.path.split(self.fake_filenames[tar_index])[-1]

        real = real.permute(2,0,1)
        fake = fake.permute(2,0,1)

        return real, fake, real_filename, fake_filename

    
#########################################################################################################

def get_validation_data(rgb_dir):
    return DataLoaderVal(rgb_dir, None)


def conv(in_channels, out_channels, kernel_size, bias=True, padding = 1, stride = 1):
    
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)


## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y
    

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=False, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

    
class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

    
class spatial_attn_layer(nn.Module):
    def __init__(self, kernel_size=3):
        super(spatial_attn_layer, self).__init__()
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        # import pdb;pdb.set_trace()
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out) # broadcasting
        return x * scale


## Dual Attention Block (DAB)
class DAB(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size, reduction,
        bias=True, bn=False, act=nn.ReLU(True)):

        super(DAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        
        self.SA = spatial_attn_layer()            ## Spatial Attention
        self.CA = CALayer(n_feat, reduction)     ## Channel Attention
        self.body = nn.Sequential(*modules_body)
        self.conv1x1 = nn.Conv2d(n_feat*2, n_feat, kernel_size=1)

    def forward(self, x):
        res = self.body(x)
        sa_branch = self.SA(res)
        ca_branch = self.CA(res)
        res = torch.cat([sa_branch, ca_branch], dim=1)
        res = self.conv1x1(res)
        res += x 
        return res


## Recursive Residual Group (RRG)
class RRG(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act,  num_dab):
        super(RRG, self).__init__()
        modules_body = []
        modules_body = [
            DAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=act) \
            for _ in range(num_dab)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


#########################################################################################################

class DenoiseNet(nn.Module):
    def __init__(self, conv=conv):
        super(DenoiseNet, self).__init__()
        num_rrg = 4
        num_dab = 8
        n_feats = 64
        kernel_size = 3
        reduction = 16 
        inp_chans = 3 
        act =nn.PReLU(n_feats)
        
        modules_head = [conv(inp_chans, n_feats, kernel_size = kernel_size, stride = 1)]

        modules_body = [
            RRG(
                conv, n_feats, kernel_size, reduction, act=act, num_dab=num_dab) \
            for _ in range(num_rrg)]

        modules_body.append(conv(n_feats, n_feats, kernel_size))
        modules_body.append(act)

        modules_tail = [conv(n_feats, inp_chans, kernel_size)]


        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)


    def forward(self, noisy_img):
        x = self.head(noisy_img)     
        x = self.body(x)
        x = self.tail(x)
        return -x

def load_checkpoint(model, weights):
    checkpoint = torch.load(weights,map_location=torch.device('cpu'))
    try:
        model.load_state_dict(checkpoint["state_dict"])
    except:
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]   # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

def compute_LNP(where, batch_size=32):   # where = 'train' or 'val' or 'test'
    dataset = get_validation_data(path + '/'+where+'/')
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=False)
    
    with torch.no_grad():
        for ii, data_test in enumerate(tqdm(loader), 0):

            rgb_real = data_test[0].to(device)
            rgb_fake = data_test[1].to(device)
            name_real = data_test[2]
            name_fake = data_test[3]

            rgb_LNP_real = torch.clamp(model_restoration(rgb_real),0,1)
            rgb_LNP_fake = torch.clamp(model_restoration(rgb_fake),0,1)

            rgb_LNP_real = rgb_LNP_real.permute(0, 2, 3, 1).cpu().detach().numpy()
            rgb_LNP_fake = rgb_LNP_fake.permute(0, 2, 3, 1).cpu().detach().numpy()


            for batch in range(len(rgb_LNP_real)):

                LNP_img_real = img_as_ubyte(rgb_LNP_real[batch])
                LNP_img_fake = img_as_ubyte(rgb_LNP_fake[batch])

                # cv2.imwrite -> plt.imsave
                if not os.path.exists('./real-vs-fake/results_mask/LNP/' + where):
                    os.makedirs('./real-vs-fake/results_mask/LNP/' + where + '/0_real/')
                    os.makedirs('./real-vs-fake/results_mask/LNP/' + where + '/1_fake/')
                cv2.imwrite(path + '/LNP/'+ where + '/0_real/' + name_real[batch][:-4] +'.png', LNP_img_real*255)
                cv2.imwrite(path + '/LNP/' + where + '/1_fake/' + name_fake[batch][:-4] +'.png', LNP_img_fake*255)  


if __name__ == '__main__':
    path = './real-vs-fake/results_mask'
    if not os.path.exists('./real-vs-fake/results_mask/LNP/'):
        os.makedirs('./real-vs-fake/results_mask/LNP/')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_restoration = DenoiseNet()
    load_checkpoint(model_restoration, '/home/wanyue/sidd_rgb.pth')
    model_restoration.to(device)
    model_restoration = nn.DataParallel(model_restoration)
    model_restoration.eval()

    mask_list = ['train','val','test']
    for i in range(len(mask_list)):
        compute_LNP(mask_list[i])








