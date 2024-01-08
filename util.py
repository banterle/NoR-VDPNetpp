#
#Copyright (C) 2023 ISTI-CNR
#Licensed under the BSD 3-Clause Clear License (see license.txt)
#

import os
import sys
import matplotlib.pyplot as plt
import torch
from torchvision.transforms.functional import to_tensor
from PIL import Image
import numpy as np
import cv2

LOGe_1400 = 7.24422751560335

#
#
#
def fromPILtoNP(img, bNorm = False):
    #img_np = np.array(img.getdata()).reshape(img.size[0], img.size[1], 3)
    img_np = np.array(img);
    img_np = img_np.astype('float32')
    if bNorm:
        img_np /= 255.0
    return img_np
    
#
#
#
def fromTorchToPil(p):
    sz = p.shape
    if len(sz) == 2:
        out = np.zeros((sz[0], sz[1], 3))
        for i in range(0, 3):
            out[:,:,i] = p
    else:
        sp = 1
        if sz[0] == 1:
            sz_0 = 3
            sp = 0
        else:
            sz_0 = sz[0]
        
        out = np.zeros((sz[1], sz[2], sz_0))
                
        c = 0
        for i in range(0, sz_0):
            tmp = p[c, 0:sz[1], 0:sz[2]]
            out[:,:,i] = tmp
            c += sp
            
    return fromNPtoPIL(out)

#
#
#
def fromNPtoPIL(img):
    formatted = (img * 255.0).astype('uint8')
    img_pil = Image.fromarray(formatted)
    return img_pil

#
#
#
def read_img_cv2(filename, maxClip = 1e4, grayscale = True, colorspace = 'REC709', display_referred = True):

    ext = (os.path.splitext(filename)[1]).lower()
    
    log_range = False

    if ext == '.hdr' or ext == '.exr':
        log_range = True

    img = cv2.imread(filename, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

    if log_range:
        img[img < 0.0] = 0.0
    
    if not log_range: #SDR images
        img = img.astype('float32')
        img = img / 255.0
        img = np.power(img, 2.2) #lineariation        

    if grayscale: #REC 709
        if len(img.shape) == 3:
            if colorspace == 'REC709':
                y = 0.2126 * img[:,:,2] + 0.7152 * img[:,:,1] + 0.0722 * img[:,:,0]
            elif colorspace == 'REC2020':
                y = 0.263  * img[:,:,2] + 0.678  * img[:,:,1] + 0.059  * img[:,:,0]
        else:
            y = img
    else:
        sz = img.shape
        y = np.reshape(img, (sz[2], sz[1], sz[0]))

    if log_range:
        if display_referred:
            y = (y * maxClip) /np.max(y)
        y = np.log(y + 1) / np.log(maxClip)

    z = torch.FloatTensor(y)

    if grayscale:
        z = z.unsqueeze(0)

    return z

#
#
#
def plotGraph(array1, array2, array3, folder, name_f):    # plot
    fig = plt.figure(figsize=(10, 4))
    n = min([len(array1), len(array2), len(array3)])
    plt.plot(np.arange(1, n + 1), array1[0:n])  # train loss (on epoch end)
    plt.plot(np.arange(1, n + 1), array2[0:n])  # train loss (on epoch end)
    plt.plot(np.arange(1, n + 1), array3[0:n])  # train loss (on epoch end)
    plt.title("model loss")
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend(['train', 'validation','test'], loc="upper left")
    title = os.path.join(folder, name_f)
    plt.savefig(title, dpi=600)
    plt.close(fig)
