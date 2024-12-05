#
#Copyright (C) 2023 ISTI-CNR
#Licensed under the BSD 3-Clause Clear License (see license.txt)
#

import torch
import torch.nn as nn
import torch.nn.functional as F
from regressor import *

#
#
#
class BlockQN(nn.Module):
    
    def __init__(self, in_size, out_size, std = 1, layer_norm = 0):
        super(BlockQN, self).__init__()
        
        if layer_norm == 0:
            ln = nn.BatchNorm2d(out_size)
        elif layer_norm == 1:
            ln = nn.InstanceNorm2d(out_size)
        
        self.conv = nn.Sequential(
                        nn.Conv2d(in_size, out_size, 3, stride = std, padding=1),
                        ln,
                        nn.ReLU())

    def forward(self, input):
        return self.conv(input)

#
#
#                             
class QNetBN(nn.Module):

    #
    #
    #
    def __init__(self, in_size=1, out_size=1, params_size = None, layer_norm = 0, bSigmoid = True):
        super(QNetBN, self).__init__()
        
        self.conv = nn.Sequential(
                                  BlockQN(in_size, 32, 1, layer_norm),
                                  BlockQN(32, 32, 1, layer_norm),
                                  nn.MaxPool2d(2),

                                  BlockQN(32, 64, 1, layer_norm),
                                  BlockQN(64, 64, 1, layer_norm),
                                  nn.MaxPool2d(2),

                                  BlockQN(64, 128, 1, layer_norm),
                                  BlockQN(128, 128, 1, layer_norm),
                                  nn.MaxPool2d(2),

                                  BlockQN(128, 256, 1, layer_norm),
                                  BlockQN(256, 256, 1, layer_norm),
                                  nn.MaxPool2d(2),
                                  
                                  BlockQN(256, 512, 1, layer_norm),
                                  BlockQN(512, 512, 1, layer_norm),
                                  nn.MaxPool2d(2)

                                  )
            
        self.regressor = Regressor(512, out_size, params_size, bSigmoid)

    #
    #
    #
    def forward(self, stim, lmax = None):
        features = self.conv(stim)
        q = self.regressor(features, lmax)
        return q

#
#
#
if __name__ == '__main__':

    model = QNetBN()
    print(model)
