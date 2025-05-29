#
#Copyright (C) 2023 ISTI-CNR
#Licensed under the BSD 3-Clause Clear License (see license.txt)
#

import torch
import torch.nn as nn
import torch.nn.functional as F
from rezero import *
from regressor import *

#
#
#
class BlockQ(nn.Module):

    def __init__(self, in_size, out_size, std = 1):
        super(BlockQ, self).__init__()
    
        self.conv = nn.Sequential(
                    nn.Conv2d(in_size, out_size, 3, stride = std, padding=1),
                    nn.ReLU())

    def forward(self, input):
        return self.conv(input)
            
#
#
#
class QNetRZF(nn.Module):

    def __init__(self, in_size=1, out_size=1, params_size = None, bSigmoid = True):
        super(QNetRZF, self).__init__()

        self.conv = nn.Sequential(
                    BlockQ(in_size, 32),
                    Conv2DRZ(32),
                    nn.MaxPool2d(2),
                                  
                    Conv2DRZx2(32),
                    Conv2DRZ(64),
                    nn.MaxPool2d(2),
                                  
                    Conv2DRZx2(64),
                    Conv2DRZ(128),
                    nn.MaxPool2d(2),
                                  
                    Conv2DRZx2(128),
                    Conv2DRZ(256),
                    nn.MaxPool2d(2),

                    Conv2DRZx2(256),
                    Conv2DRZ(512),
                    nn.MaxPool2d(2)

        )
        
        self.regressor = Regressor(512, out_size, params_size, bSigmoid)

    def forward(self, stim, lmax = None):
        features = self.conv(stim)
        q = self.regressor(features, lmax)
        return q

if __name__ == '__main__':

    model = QNetRZF()
    print(model)
