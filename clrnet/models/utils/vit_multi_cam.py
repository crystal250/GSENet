import math
from copy import copy
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d
from PIL import Image
from torch.cuda import amp

class simam_module(torch.nn.Module):
    def __init__(self, channels = None, e_lambda = 1e-4):
        super(simam_module, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):

        b, c, h, w = x.size()
        
        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2,3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2,3], keepdim=True) / n + self.e_lambda)) + 0.5
        # print(y)
        return x * self.activaton(y)

class DilatedContextAttentionModule(nn.Module):
    def __init__(self, mode='dot_product', heads=8,channels=512,patch_size=5,img_w=25,img_h=10):
        super(DilatedContextAttentionModule, self).__init__()
        self.in_channels = channels
        self.inter_channels = channels
        # self.img_size = img_size
        self.patch_size = patch_size
        # self.embed_dim = embed_dim
        self.num_patches = (img_w // patch_size) *(img_h//patch_size)
        self.num_heads = heads
        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        self.W = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                           kernel_size=1, stride=1, padding=0)

        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                               kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        if mode == 'dot_product':
            self.operation_function = self._dot_product
        elif mode == 'embedded_gaussian':
            self.operation_function = self._embedded_gaussian
        self.bn = nn.BatchNorm2d(self.in_channels)
    def forward(self, x1, x2):
        """
        :param x1: (b, c, t, h, w)
        :param x2: (b, c, t, h, w)
        :return:
        """
        output = self.operation_function(x1, x2)
        return output

    def _embedded_gaussian(self, xi, xj):  # xi is the original feature map , xj is the context one
        """
        :param xi: (b, c, t, h, w)
        :param xj: (b, c, t, h, w)
        :return:
        """
        batch_size, channels, height, width = xi.size()

        # Split the input tensor into patches and flatten them
        xj = xj.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        # print(x.shape)
        xj = xj.flatten(start_dim=4, end_dim=5).flatten(start_dim=2, end_dim=3)
        # print(xj.shape)
        g_x = self.g(xj).view(batch_size, self.num_patches, self.num_heads, -1)
        g_x = g_x.permute(0, 3, 1, 2)

        theta_x = self.theta(xj).view(batch_size, self.num_patches, self.num_heads, -1)
        # theta_x = self.theta(x).view(-1,self.num_patches, self.num_heads)
        theta_x = theta_x.permute(0,3,1,2)
        # print(theta_x.shape)
        phi_x = self.phi(xj).view(batch_size, self.num_patches, self.num_heads, -1)
        # phi_x = self.phi(x).view(-1,self.num_patches, self.num_heads)
        
        phi_x = phi_x.permute(0,3, 2,1)
        # print(phi_x.shape)
        
        f = torch.matmul(theta_x, phi_x)
        # print(f.shape)
        f_div_C = F.softmax(f, dim=-1)
        # print(f_div_C.shape)
        y = torch.matmul(f_div_C, g_x)
        # f = f.softmax(dim=-1)
        # print(y.shape)
        # y = torch.matmul(f_x, g_x)
        z = y.view(batch_size,channels,height,width)
        # print(z.shape)
        z =z +xi
        return z

    def _dot_product(self, xi, xj):  # xi is the original feature map , xj is the context one
        batch_size = xi.size(0)
        # g=>(b, c, t, h, w)->(b, 0.5c, t, h, w)->(b, thw, 0.5c)
        g_x = self.g(xj).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(xi).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(xj).view(batch_size, self.inter_channels, -1)

        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *xi.size()[2:])
        W_y = self.W(y)  # conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0)
        z = W_y + xi
        z = self.bn(z)
        return z

class Dilated_Context_Attention_Sum(nn.Module):

    def __init__(self, c1, c2, k, s, padding_dilation, input_channels, p=None, g=1, act=True):
        super(Dilated_Context_Attention_Sum, self).__init__()
        self.branch0 = nn.Conv2d(input_channels, input_channels, kernel_size=k, stride=s, padding=padding_dilation,
                                 dilation=padding_dilation, bias=True)
        self.silu =  nn.SiLU()
        self.cam = DilatedContextAttentionModule(mode='embedded_gaussian', channels=input_channels)
        self.reduction = nn.Conv2d(input_channels * 2, input_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.simam = simam_module(input_channels)

    def forward(self, input):
        x0 = self.branch0(input)
        x0 = self.silu(x0)
        out = self.cam(input, x0)
        layer_cat = torch.cat((out, input), 1)
        out = self.reduction(layer_cat)
        out = self.silu(out)  ### 最好加silu
        out = self.simam(out)
        return out

# if __name__ == '__main__':
#     img = torch.rand(16, 32, 80, 80)
#     model = Dilated_Context_Attention_Sum(32, 32, 3, 1, padding_dilation=5, input_channels=32)
#     out = model(img)
#     # print(out.shape)

