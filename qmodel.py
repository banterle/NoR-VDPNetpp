#
#Copyright (C) 2023 ISTI-CNR
#Licensed under the BSD 3-Clause Clear License (see license.txt)
#

import os
import re
import glob2
import argparse

import torch

from model_classic import QNetC
from model_bn import QNetBN
from model_rz import QNetRZ
from model_res import QNetRes

from util import read_img_cv2

#
#
#
class QModel:

    #
    #
    #
    def __init__(self, run, btype = 2, maxClip = 1400, grayscale = True, colorspace = 'REC709', display_referred = 'yes', qbSigmoid = True):
    
        self.run = run
        ext = os.path.splitext(run)[1]
        
        if ext == '':
            ckpt_dir = os.path.join(run, 'ckpt')
            ckpts = glob2.glob(os.path.join(ckpt_dir, '*.pth'))
            assert ckpts, "No checkpoints to resume from!"

            def get_epoch(ckpt_url):
                s = re.findall("ckpt_e(\\d+).pth", ckpt_url)
                epoch = int(s[0]) if s else -1
                return epoch, ckpt_url

            start_epoch, ckpt = max(get_epoch(c) for c in ckpts)
            print('Checkpoint:', ckpt)
        else:
            ckpt = run

        bLoad = True

        if ckpt == 'none.pth':
            bLoad = False
            
        if grayscale:
            n_in =1
        else:
            n_in = 3
                        
        if bLoad:
            if torch.cuda.is_available():
                ckpt = torch.load(ckpt, weights_only=True)
            else:
                ckpt = torch.load(ckpt, weights_only=True, map_location=torch.device('cpu'))

        if 'grayscale' in ckpt:
            if ckpt['grayscale']:
                n_in =1
            else:
                n_in = 3

        if 'sigmoid' in ckpt:
            if ckpt['sigmoid'] == 1:
                qbSigmoid = True

            if ckpt['sigmoid'] == 0:
                qbSigmoid = False

            if ckpt['sigmoid'] == True:
                qbSigmoid = True

            if ckpt['sigmoid'] == False:
                qbSigmoid = False

        if 'type' in ckpt:
            btype = ckpt['type']

        if btype == 0:
            model = QNetC(n_in, 1, bSigmoid = qbSigmoid)
        elif btype == 1:
            model = QNetBN(n_in, 1, bSigmoid = qbSigmoid)
        elif btype == 2:
            model = QNetRZ(n_in, 1, bSigmoid = qbSigmoid)
        elif btype == 3:
            model = QNetRes(n_in, 1, bSigmoid = qbSigmoid)

        model.load_state_dict(ckpt['model'])

        if(torch.cuda.is_available()):
            model = model.cuda()

        model.eval()
        
        self.model = model

        self.colorspace = colorspace
        self.maxClip = maxClip
        self.grayscale = grayscale
        self.display_referred = (display_referred == 'yes')
    
    #
    #
    #
    def getModel():
        return self.model

    #
    #
    #
    def predict(self, fn):
               
        stim = read_img_cv2(fn, maxClip = self.maxClip, grayscale = self.grayscale, colorspace = self.colorspace, display_referred = self.display_referred)
        stim = stim.unsqueeze(0)

        if torch.cuda.is_available():
            stim = stim.cuda()

        with torch.no_grad():
            out = self.model(stim)

        out = out.data.cpu().numpy().squeeze()

        return out

    #
    #
    #
    def predict_t(self, stim):
        with torch.no_grad():
             out = self.model(stim)

        return out
