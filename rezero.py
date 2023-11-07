#
#Copyright (C) 2023 ISTI-CNR
#Licensed under the BSD 3-Clause Clear License (see license.txt)
#

import torch
import torch.nn as nn
import torch.nn.functional as F

#
#
#
class Conv2DRZ(nn.Module):

    def __init__(self, in_channels, kernel_size = 3, stride_rz = 1, padding_rz = 1, dilation_rz=1, groups_rz=1, bias_rz=True, padding_mode_rz='zeros'):
        super(Conv2DRZ, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size, stride_rz, padding_rz, dilation_rz, groups_rz, bias_rz, padding_mode_rz)

        self.alpha = nn.Parameter(torch.Tensor([0]))

    def forward(self, x):
        f_x = self.conv(x)
        return x + self.alpha * torch.relu(f_x)

#
#
#
class Conv2DRZx2(nn.Module):
    
    def __init__(self, in_channels, kernel_size = 3, stride_rzx2 = 1, padding_rzx2 = 1, dilation_rzx2=1, groups_rzx2=1, bias_rzx2=True, padding_mode_rzx2='zeros'):
        super(Conv2DRZx2, self).__init__()
    
        self.f0 = Conv2DRZ(in_channels, kernel_size, stride_rz = stride_rzx2, padding_rz = padding_rzx2, dilation_rz = dilation_rzx2, groups_rz = groups_rzx2, bias_rz = bias_rzx2, padding_mode_rz = padding_mode_rzx2)
        self.f1 = Conv2DRZ(in_channels, kernel_size, stride_rz = stride_rzx2, padding_rz = padding_rzx2, dilation_rz = dilation_rzx2, groups_rz = groups_rzx2, bias_rz = bias_rzx2, padding_mode_rz = padding_mode_rzx2)

    def forward(self, x):
        y0 = self.f0(x)
        y1 = self.f1(x)
        return torch.cat((y0, y1), 1)

#
#
#
class LayerRZ(nn.Module):

    def __init__(self, layer):
        super(LayerRZ, self).__init__()
        
        self.layer = layer
        self.alpha = nn.Parameter(torch.Tensor([0]))
    
    def forward(self, x):
        f_x = self.layer(x)
        return x + self.alpha * f_x
