import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch
import numpy
from torchsummary import summary
import torch.fft as fft
import numpy as np
#### Import mobilenet v2
from networks.mobilenetv2 import MobileNetV2

__all__ = ['ResNet', 'resnet50']

model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}

################# add DAB Block
def conv(in_channels, out_channels, kernel_size, bias=True, padding = 1, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)

## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=1):
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
        bias=True, bn=False, act=nn.ReLU(True),if_spacial=False):  ## Use only channel attention or Dual Attention

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
        self.conv1x1_c = nn.Conv2d(n_feat,n_feat,kernel_size=1)  ## 此处有待讨论 要是channel attention 到底需不需要这层卷积
        self.if_spacial = if_spacial

    def forward(self, x):
        res = self.body(x)
        if self.if_spacial == True:
            sa_branch = self.SA(res)
            ca_branch = self.CA(res)
            res = torch.cat([sa_branch, ca_branch], dim=1)
            res = self.conv1x1(res)
            res += x
        else: 
            ca_branch = self.CA(res)
            res = ca_branch
            res += x
        return res


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, _mode=1,zero_init_residual=False,if_spacial=False):    ## 0: Att(A.S.+ P.S.) + LNP
        super(ResNet, self).__init__()                                                                         ## 1: Att(A.S + P.S. + LNP)
        self.inplanes = 64                                                                                     ## 2: (A.S + P.S. + LNP)
        self.conv1 = nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.if_spacial = if_spacial
        self.mode = _mode
        if self.mode == 0:
            self.DAB = DAB(conv,reduction=1,n_feat=2,kernel_size=3,bias=True,bn=False,act=nn.ReLU(True),if_spacial=self.if_spacial)
        if self.mode == 1 or self.mode == 2:
            self.DAB = DAB(conv,reduction=1,n_feat=5,kernel_size=3,bias=True,bn=False,act=nn.ReLU(True),if_spacial=self.if_spacial)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        input = x
        x_fft = fft2d(input)
        x_phase = torch.atan2(x_fft.imag + 1e-8, x_fft.real + 1e-8)
        phase = fftshift2d(x_phase)
        phase_1 = 1/3 * phase[:, 0, :, :] + 1/3 * phase[:, 1, :, :] + 1/3 * phase[:, 2, :, :]
        x_amplitude = torch.abs(x_fft)
        x_amplitude = torch.pow(x_amplitude + 1e-8, 0.8)
        amplitude = fftshift2d(x_amplitude)
        amplitude_1 = 1/3 * amplitude[:, 0, :, :] + 1/3 * amplitude[:, 1, :, :] + 1/3 * amplitude[:, 2, :, :]

        amplitude = torch.unsqueeze(amplitude_1, dim=1)
        phase = torch.unsqueeze(phase_1, dim=1)

        if self.mode == 0: ## 0: Att(A.S.+ P.S.) + LNP
            a_p = torch.cat((amplitude,phase),dim=1)
            a_p_DAB = self.DAB(a_p)
            x = torch.cat((x, a_p_DAB), dim=1)
        if self.mode == 1: ## 1: Att(A.S + P.S. + LNP)
            x = torch.cat((x,amplitude),dim=1)
            x = torch.cat((x,phase),dim=1)
            x = self.DAB(x)
        if self.mode == 2:  ## 2: (A.S + P.S. + LNP)
            x = torch.cat((x,amplitude),dim=1)
            x = torch.cat((x,phase),dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def fft2d(input):
    fft_out = fft.fftn(input, dim=(2, 3), norm='ortho')
    return fft_out


def fftshift2d(input):
    b, c, h, w = input.shape
    fs11 = input[:, :, -h // 2:h, -w // 2:w]
    fs12 = input[:, :, -h // 2:h, 0:w // 2]
    fs21 = input[:, :, 0:h // 2, -w // 2:w]
    fs22 = input[:, :, 0:h // 2, 0:w // 2]
    output = torch.cat([torch.cat([fs11, fs21], axis=2), torch.cat([fs12, fs22], axis=2)], axis=3)
    return output


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(block=Bottleneck,layers=[3, 4, 6, 3],if_spacial = False,**kwargs)
    if pretrained:
        print('enter pretrain')
        path_resnet = './pretrained/resnet50.pth'
        state_dict = torch.load(path_resnet)
        name_dict = dict([(name, param) for name, param in state_dict.items()])
        del name_dict['conv1.weight']
        model.load_state_dict(name_dict,strict=False)
    return model




