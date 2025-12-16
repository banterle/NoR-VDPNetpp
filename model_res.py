#
#Copyright (C) 2023 ISTI-CNR
#Licensed under the BSD 3-Clause Clear License (see license.txt)
#

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet18_Weights
from regressor import *

#
#
#
class QNetRes(nn.Module):

    #
    #
    #
    def __init__(self, in_size = 1, out_size=1, params_size = None, whichResnet = 18, bSigmoid = True):
        super(QNetRes, self).__init__()
        
        """Load the pretrained ResNet-50 and replace top fc layer."""
        if whichResnet == 18:
            resnet = models.resnet18(weights=None)
           
        if whichResnet == 50:
            resnet = models.resnet50(weights = None)

        for param in resnet.parameters():
            param.requires_grad = False
            
        resnet.eval()
            
        modules = list(resnet.children())[:-1] #remove the last fc layer.
        self.resnet = nn.Sequential(*modules)
        
        self.resnet.eval()

        #network            
        self.regressor = Regressor(resnet.fc.in_features, out_size, params_size, bSigmoid)
    
     
    #
    #
    #
    def forward(self, stim, lmax = None):
    
        # ResNet CNN
        sz = stim.shape

        if sz[1] == 1:
            tmp = torch.zeros((sz[0], 3, sz[2], sz[3]))
            tmp[:,0,:,:] = stim[:,0,:,:]
            tmp[:,1,:,:] = stim[:,0,:,:]
            tmp[:,2,:,:] = stim[:,0,:,:]
        
            if torch.cuda.is_available():
                stim = tmp.cuda()
            else:
                stim = tmp

        stim = (stim - 0.45) / 0.225
            
        with torch.no_grad():
            x = self.resnet(stim)      #apply ResNet
        x = x.view(x.size(0), -1)  #flatten output of conv

        q = self.regressor(x, lmax)
            
        return q


if __name__ == '__main__':

    model = QNetRes()
    print(model)
