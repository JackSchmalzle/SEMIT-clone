"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license
(https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import torch
import torch.nn.functional as F
from torch import nn

from functools import partial
import pdb

# normalization edge-case note
# quick local checks:
# # if False:
# #     import torch
# #     x = torch.randn(2, 64, 32, 32)
# #     print("diag:", x.mean().item(), x.std().item())
# for dev diagnostics only and is intentionally disabled.

import random
import wavelet

class WTConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, padding=0,
                 bias=False, wt_type='db1', alpha_in=0.25, alpha_out=0.25, type='normal'):
        super(WTConv2d, self).__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.type = type
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.wt_type = wt_type

        hf_ch_in = in_channels
        hf_ch_out = out_channels
        lf_ch_in = in_channels
        lf_ch_out = out_channels

        # register wavelet filters based on type
        if type == 'first':
            wt_filter, iwt_filter = wavelet.create_2d_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
            self.register_buffer('wt_filter', wt_filter)
            self.register_buffer('iwt_filter', iwt_filter)

            self.convh = nn.Conv2d(
                in_channels, hf_ch_out,
                kernel_size=kernel_size, stride=stride, padding=padding, bias=bias
            )
            self.convl = nn.Conv2d(
                in_channels, lf_ch_out,
                kernel_size=kernel_size, stride=stride, padding=padding, bias=bias
            )

        elif type == 'last':
            wt_filter, iwt_filter = wavelet.create_2d_wavelet_filter(wt_type, lf_ch_in, lf_ch_in, torch.float)
            self.register_buffer('wt_filter', wt_filter)
            self.register_buffer('iwt_filter', iwt_filter)

            self.convh = nn.Conv2d(hf_ch_in, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
            self.convl = nn.Conv2d(lf_ch_in, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)

        else:  # 'normal'
            # For dwt on hf (hf_ch_in channels)
            wt_filter_in, iwt_filter_in = wavelet.create_2d_wavelet_filter(wt_type, hf_ch_in, hf_ch_in, torch.float)
            self.register_buffer('wt_filter_in', wt_filter_in)
            self.register_buffer('iwt_filter_in', iwt_filter_in)

            # For idwt after L2H (hf_ch_out channels)
            wt_filter_out, iwt_filter_out = wavelet.create_2d_wavelet_filter(wt_type, hf_ch_out, hf_ch_out, torch.float)
            self.register_buffer('wt_filter_out', wt_filter_out)
            self.register_buffer('iwt_filter_out', iwt_filter_out)

            self.L2L = nn.Conv2d(
                lf_ch_in, lf_ch_out,
                kernel_size=kernel_size, stride=stride, padding=padding, bias=bias
            )
            self.L2H = nn.Conv2d(
                lf_ch_in, hf_ch_out,
                kernel_size=kernel_size, stride=stride, padding=padding, bias=bias
            )
            self.H2L = nn.Conv2d(
                hf_ch_in, lf_ch_out,
                kernel_size=kernel_size, stride=stride, padding=padding, bias=bias
            )
            self.H2H = nn.Conv2d(
                hf_ch_in, hf_ch_out,
                kernel_size=kernel_size, stride=stride, padding=padding, bias=bias
            )

    def dwt(self, x, wt_filter=None):
        """Perform 2D discrete wavelet transform and return LL subband"""
        if wt_filter is None:
            wt_filter = self.wt_filter

        b, c, h, w = x.shape
        # pad if necessary
        if (h % 2 > 0) or (w % 2 > 0):
            x = F.pad(x, (0, w % 2, 0, h % 2))

        x_wt = wavelet.wavelet_2d_transform(x, wt_filter)
        # return only LL subband (index 0)
        return x_wt[:, :, 0, :, :]

    def idwt(self, x_ll, iwt_filter=None):
        """Perform inverse 2D wavelet transform from LL subband (upsampling)"""
        if iwt_filter is None:
            iwt_filter = self.iwt_filter

        b, c, h, w = x_ll.shape
        # create zero high-frequency subbands
        x_h = torch.zeros(b, c, 3, h, w, device=x_ll.device, dtype=x_ll.dtype)
        x = torch.cat([x_ll.unsqueeze(2), x_h], dim=2)

        return wavelet.inverse_2d_wavelet_transform(x, iwt_filter)

    def mask(self, hf, lf, alpha_in=0.25, alpha_out=0.25, order=True):
        mask_hf = torch.zeros_like(hf)
        mask_lf = torch.zeros_like(lf)
        c = hf.shape[1]
        hf_ch_out = int(c * (1 - alpha_out))
        lf_ch_out = c - hf_ch_out
        if order:
            index_hf = [i for i in range(hf_ch_out)]
        else:
            index_hf = random.sample(list(range(c)), hf_ch_out)
        index_lf = [i for i in range(c) if i not in index_hf]
        assert len(index_hf) == hf_ch_out
        assert len(index_lf) == lf_ch_out

        mask_hf[:, index_hf, :, :] = 1.
        mask_lf[:, index_lf, :, :] = 1.
        hf = hf * mask_hf
        lf = lf * mask_lf
        return hf, lf

    def forward(self, x, alpha_in, alpha_out):
        if self.type == 'first':
            hf = self.convh(x)
            # use DWT for low frequency decomposition
            lf = self.dwt(x)
            lf = self.convl(lf)
            hf, lf = self.mask(hf, lf, alpha_in=alpha_in, alpha_out=alpha_out)
            return hf, lf

        elif self.type == 'last':
            hf, lf = x
            # use IDWT for wavelet reconstruction
            return self.convh(hf) + self.convl(self.idwt(lf))

        else:  # 'normal'
            hf, lf = x
            # use IDWT for upsampling (L->H) and DWT for downsampling (H->L)
            hf_new = self.H2H(hf) + self.idwt(self.L2H(lf), self.iwt_filter_out)
            lf_new = self.L2L(lf) + self.H2L(self.dwt(hf, self.wt_filter_in))
            hf, lf = self.mask(hf_new, lf_new, alpha_in=alpha_in, alpha_out=alpha_out)
            return hf, lf


class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        self.bias = None

    def forward(self, x):
        return torch.mul(self.weight, x)

def oct_conv7x7(in_planes, out_planes,alpha_in=0.25, alpha_out=0.25, kernel_size=7, stride=1, padding =3, type='normal'):
    """7x7 convolution with padding"""
    return WTConv2d(in_planes, out_planes, alpha_in=alpha_in, alpha_out=alpha_out, kernel_size=kernel_size, stride=stride, padding=padding,  type=type)
def norm_conv7x7(in_planes, out_planes,alpha_in=0.25, alpha_out=0.25, kernel_size=7, stride=1, padding =3, type=None):
    """7x7 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)

def oct_conv5x5(in_planes, out_planes,alpha_in=0.25, alpha_out=0.25, kernel_size=5, stride=1, padding =3, type='normal'):
    """5x5 convolution with padding"""
    return WTConv2d(in_planes, out_planes, alpha_in=alpha_in, alpha_out=alpha_out, kernel_size=kernel_size, stride=stride, padding=padding, type=type)
def norm_conv5x5(in_planes, out_planes,alpha_in=0.25, alpha_out=0.25, kernel_size=5, stride=1, padding =3,  type=None):
    """5x5 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,  padding=padding, bias=False)

def oct_conv4x4(in_planes, out_planes,alpha_in=0.25, alpha_out=0.25, kernel_size=4, stride=2, padding =3, type='normal'):
    """4x4 convolution with padding"""
    return WTConv2d(in_planes, out_planes, alpha_in=alpha_in, alpha_out=alpha_out, kernel_size=kernel_size, stride=stride, padding=padding,  type=type)
def norm_conv4x4(in_planes, out_planes,alpha_in=0.25, alpha_out=0.25, kernel_size=4, stride=2, padding=3, type=None):
    """4x4 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)

# conv toggle snippet
# # try_toggle = True
# # if try_toggle:
# #     conv3x3 = norm_conv3x3
# # else:
# #     conv3x3 = oct_conv3x3
# # allows local switching when debugging conv vs octconv effects.
# # commented so runtime remains unchanged.


def oct_conv3x3(in_planes, out_planes,alpha_in=0.25, alpha_out=0.25, kernel_size=3, stride=1, padding =3, type='normal'):
    """3x3 convolution with padding"""
    return WTConv2d(in_planes, out_planes, alpha_in=alpha_in, alpha_out=alpha_out, kernel_size=kernel_size, stride=stride, padding=padding, type=type)
def norm_conv3x3(in_planes, out_planes,alpha_in=0.25, alpha_out=0.25, kernel_size=3, stride=1, padding =3, type=None):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)

class Oct_conv_norm(nn.Module):
    def __init__(self, planes, alpha_in=0.25, alpha_out=0.25, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, norm='in'):
        super(Oct_conv_norm, self).__init__()
        #hf_ch = int(num_features * (1 - alpha_in))
        #lf_ch = num_features - hf_ch
        hf_ch = planes 
        lf_ch = planes 
        if norm=='in':
            self.bnh = nn.InstanceNorm2d(hf_ch)
            self.bnl = nn.InstanceNorm2d(lf_ch)
        elif norm=='adain':
            self.bnh = AdaptiveInstanceNorm2d(hf_ch)
            self.bnl = AdaptiveInstanceNorm2d(lf_ch)
        else:
            self.bnh = nn.BatchNorm2d(hf_ch)
            self.bnl = nn.BatchNorm2d(lf_ch)
      

    def forward(self, x, alpha_in=0.25, alpha_out=0.25):
        hf, lf = x
        return self.bnh(hf), self.bnl(lf)
class conv_norm(nn.Module):
    def __init__(self, planes, alpha_in=0.25, alpha_out=0.25, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True, norm='in'):
        super(conv_norm, self).__init__()
        #hf_ch = int(num_features * (1 - alpha_in))
        #lf_ch = num_features - hf_ch
        ch = planes 
        ch = planes 
        if norm=='bn':
            self.bn = nn.BatchNorm2d(ch)
        else:
            self.bn = nn.InstanceNorm2d(ch)

    def forward(self, x, alpha_in=0.25, alpha_out=0.25):
        return self.bn(x)

class Oct_conv_reLU(nn.ReLU):
    def forward(self, x):
        hf, lf = x
        hf = super(Oct_conv_reLU, self).forward(hf)
        lf = super(Oct_conv_reLU, self).forward(lf)
        return hf, lf
class Oct_conv_lreLU(nn.LeakyReLU):
    def forward(self, x):
        hf, lf = x
        hf = super(Oct_conv_lreLU, self).forward(hf)
        lf = super(Oct_conv_lreLU, self).forward(lf)
        return hf, lf
class Oct_conv_up(nn.Upsample):
    def forward(self, x):
        hf, lf = x
        hf = super(Oct_conv_up, self).forward(hf)
        lf = super(Oct_conv_up, self).forward(lf)
        return hf, lf

class ResBlocks(nn.Module):
    def __init__(self, num_blocks, dim, norm, activation, pad_type):
        super(ResBlocks, self).__init__()
        self.model = []
        for i in range(num_blocks):
            self.model += [ResBlock(dim,
                                    norm=norm,
                                    activation=activation,
                                    pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)

class ActFirst_no_normalization(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, conv_dim=64, oct_conv_on = True,  norm='in'):
        super(ActFirst_no_normalization, self).__init__()
        oct_conv_on = True
        conv3x3 = oct_conv3x3 if oct_conv_on else norm_conv3x3
        norm_func = Oct_conv_norm if oct_conv_on else conv_norm
        act_func = Oct_conv_lreLU if oct_conv_on else nn.LeakyReLU

        self.conv1 = conv3x3(in_planes=conv_dim, out_planes=conv_dim, padding=1, type="normal")
        self.re1 = act_func(negative_slope=0.2, inplace=False)

        self.conv2 = conv3x3(in_planes=conv_dim, out_planes=conv_dim, padding=1, type="normal")
        self.re2 = act_func(negative_slope=0.2, inplace=False)

    def forward(self, x, alpha_in=0.25, alpha_out=0.25):
        #basic block: 2 conv + input
        #conv1
        out=self.re1(x)
        out=self.conv1(out, alpha_in=alpha_in, alpha_out=alpha_out)
        #conv2
        out=self.re2(out)
        out=self.conv2(out, alpha_in=alpha_in, alpha_out=alpha_out)
        return x[0] + out[0], x[1] + out[1] 


class ResidualOctBlock_basic(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, conv_dim=64, alpha_in=0.25, alpha_out=0.25, oct_conv_on = True,  norm='in'):
        super(ResidualOctBlock_basic, self).__init__()
        oct_conv_on = True
        alpha_in, alpha_out = 0.5, 0.5
        conv3x3 = oct_conv3x3 if oct_conv_on else norm_conv3x3
        norm_func = Oct_conv_norm if oct_conv_on else conv_norm
        act_func = Oct_conv_reLU if oct_conv_on else nn.ReLU

        self.conv1 = conv3x3(in_planes=conv_dim, out_planes=conv_dim, alpha_in=alpha_in, alpha_out=alpha_out, padding=1, type="normal")
        self.bn1 = norm_func(planes = conv_dim, norm = norm)
        self.re1 = act_func(inplace=True)

        self.conv2 = conv3x3(in_planes=conv_dim, out_planes=conv_dim, alpha_in=alpha_in, alpha_out=alpha_out, padding=1, type="normal")
        self.bn2 = norm_func(planes = conv_dim, norm = norm)

    def forward(self, x, alpha_in=0.25, alpha_out=0.25):
        #basic block: 2 conv + input
        #conv1
        out=self.conv1(x, alpha_in=alpha_in, alpha_out=alpha_out)
        out=self.bn1(out)
        out=self.re1(out)
        #conv2
        out=self.conv2(out, alpha_in=alpha_in, alpha_out=alpha_out)
        out=self.bn2(out)
        return x[0] + out[0], x[1] + out[1] 



class ResBlock(nn.Module):
    def __init__(self, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlock, self).__init__()
        model = []
        model += [Conv2dBlock(dim, dim, 3, 1, 1,
                              norm=norm,
                              activation=activation,
                              pad_type=pad_type)]
        model += [Conv2dBlock(dim, dim, 3, 1, 1,
                              norm=norm,
                              activation='none',
                              pad_type=pad_type)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out


class ActFirstResBlock(nn.Module):
    def __init__(self, fin, fout, fhid=None,
                 activation='lrelu', norm='none'):
        super().__init__()
        self.learned_shortcut = (fin != fout)
        self.fin = fin
        self.fout = fout
        self.fhid = min(fin, fout) if fhid is None else fhid
        self.conv_0 = Conv2dBlock(self.fin, self.fhid, 3, 1,
                                  padding=1, pad_type='reflect', norm=norm,
                                  activation=activation, activation_first=True)
        self.conv_1 = Conv2dBlock(self.fhid, self.fout, 3, 1,
                                  padding=1, pad_type='reflect', norm=norm,
                                  activation=activation, activation_first=True)
        if self.learned_shortcut:
            self.conv_s = Conv2dBlock(self.fin, self.fout, 1, 1,
                                      activation='none', use_bias=False)

    def forward(self, x):
        x_s = self.conv_s(x) if self.learned_shortcut else x
        dx = self.conv_0(x)
        dx = self.conv_1(dx)
        out = x_s + dx
        return out


class LinearBlock(nn.Module):
    def __init__(self, in_dim, out_dim, norm='none', activation='relu'):
        super(LinearBlock, self).__init__()
        use_bias = True
        self.fc = nn.Linear(in_dim, out_dim, bias=use_bias)

        # initialize normalization
        norm_dim = out_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=False)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=False)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

    def forward(self, x):
        out = self.fc(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out


class Conv2dBlock(nn.Module):
    def __init__(self, in_dim, out_dim, ks, st, padding=0,
                 norm='none', activation='relu', pad_type='zero',
                 use_bias=True, activation_first=False):
        super(Conv2dBlock, self).__init__()
        self.use_bias = use_bias
        self.activation_first = activation_first
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = out_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'adain':
            self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=False)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=False)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        self.conv = nn.Conv2d(in_dim, out_dim, ks, st, bias=self.use_bias)

    def forward(self, x):
        if self.activation_first:
            if self.activation:
                x = self.activation(x)
            x = self.conv(self.pad(x))
            if self.norm:
                x = self.norm(x)
        else:
            x = self.conv(self.pad(x))
            if self.norm:
                x = self.norm(x)
            if self.activation:
                x = self.activation(x)
        return x


class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weight = None
        self.bias = None
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and \
               self.bias is not None, "Please assign AdaIN weight first"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])
        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)
        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'
