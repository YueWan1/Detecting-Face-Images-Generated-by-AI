"""
Creates a MobileNetV2 Model as defined in:
Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen. (2018). 
MobileNetV2: Inverted Residuals and Linear Bottlenecks
arXiv preprint arXiv:1801.04381.
import from https://github.com/tonylins/pytorch-mobilenet-v2
"""
import torch
import torch.nn as nn
import math
import torch.fft as fft

__all__ = ['mobilenetv2']


############### add DAB Block
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


## P.S. & A.S. Transform
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





def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1, width_mult=1.,_mode=1,if_spacial=False):
        super(MobileNetV2, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = [
            # t, c, n, s
            [1,  16, 1, 1],
            [6,  24, 2, 2],
            [6,  32, 3, 2],
            [6,  64, 4, 2],
            [6,  96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        input_channel = _make_divisible(32 * width_mult, 4 if width_mult == 0.1 else 8)
        layers = [conv_3x3_bn(5, input_channel, 2)]   
        # building inverted residual blocks
        block = InvertedResidual
        for t, c, n, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 4 if width_mult == 0.1 else 8)
            for i in range(n):
                layers.append(block(input_channel, output_channel, s if i == 0 else 1, t))
                input_channel = output_channel
        self.features = nn.Sequential(*layers)
        # building last several layers
        output_channel = _make_divisible(1280 * width_mult, 4 if width_mult == 0.1 else 8) if width_mult > 1.0 else 1280
        self.conv = conv_1x1_bn(input_channel, output_channel)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(output_channel, num_classes)
        self._initialize_weights()
        self.if_spacial = if_spacial
        self.mode = _mode 
        if self.mode == 0:
            self.DAB = DAB(conv,reduction=1,n_feat=2,kernel_size=3,bias=True,bn=False,act=nn.ReLU(True),if_spacial=self.if_spacial)
        if self.mode == 1 or self.mode == 2:
            self.DAB = DAB(conv,reduction=1,n_feat=5,kernel_size=3,bias=True,bn=False,act=nn.ReLU(True),if_spacial=self.if_spacial)


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

        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def mobilenetv2(pretrained=False,**kwargs):
    """Constructs a MobilenetV2 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = MobileNetV2(if_spacial=False,**kwargs)
    if pretrained:
        print('enter pretrain')
        path_mobilenetv2 = './pretrained/mobilenetv2-c5e733a8.pth'
        state_dict = torch.load(path_mobilenetv2)
        name_dict = dict([(name, param) for name, param in state_dict.items()])
        del name_dict['features.0.0.weight']
        del name_dict['classifier.weight']
        del name_dict['classifier.bias']
        model.load_state_dict(name_dict,strict=False)
    return model