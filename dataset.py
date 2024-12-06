import os
import pandas as pd
import torch
from util import read_img_cv2
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from torchvision.transforms.functional import to_tensor
import torchvision.transforms as T
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#
def torchDataAugmentation(img, j):
    img_out = []
    if(j == 0):
        img_out = img
    elif (j == 1):
        img_out = T.functional.rotate(img, 90)
    elif (j == 2):
        img_out = T.functional.rotate(img, 180)
    elif (j == 3):
        img_out = T.functional.rotate(img, 270)
    elif (j == 4):
        img_out = T.functional.hflip(img)
    elif (j == 5):
        img_tmp = T.functional.rotate(img, 90)
        img_out = T.functional.hflip(img_tmp)
        del img_tmp
    elif (j == 6):
        img_out = T.functional.vflip(img)
    elif (j == 7):
        img_out = T.functional.rotate(img, 30)
    elif (j == 8):
        img_out = T.functional.rotate(img, -30)
        
    return img_out

#
#
#
def getVec(data):
    n = len(data)
    vec = []
    
    hist = np.zeros((101,1))
    for i in range(0, n):
        q = data.iloc[i].Q
        vec.append(q)
        index = int(np.ceil(q))
        hist[index] += 1
        
    return vec, hist
    
#
#
#
def filterData(data, groupaffine):
    out = []
    
    fn = []
    q_val = []
    lmax = []
    gpa = []
        
    n = len(data)
    for i in range(0, n):
        if '_P_0_' in data.iloc[i].Distorted:
           continue
           
        fn.append(data.iloc[i].Distorted)
        q_val.append(data.iloc[i].Q)
        lmax.append(data.iloc[i].Lmax)
        
        if groupaffine > 1:
            gpa.append(data.iloc[i].I)
 
    if groupaffine <= 1:
        d = {'Distorted': fn, 'Lmax': lmax, 'Q': q_val}
    else:
        d = {'Distorted': fn, 'Lmax': lmax, 'Q': q_val, 'I': gpa}
    out = pd.DataFrame(data=d)
    return out
    
#
#
#
def read_data_split(data_dir, group=None, groupaffine = 1):
    train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    train = filterData(train, groupaffine)
    train.sort_values(by=['Distorted'], inplace=True)

    val = pd.read_csv(os.path.join(data_dir, 'val.csv'))
    val = filterData(train, groupaffine)
    val.sort_values(by=['Distorted'], inplace=True)

    test = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    val = filterData(train, groupaffine)
    test.sort_values(by=['Distorted'], inplace=True)

    return train, val, test

#
#
#
def split_data(data_dir, random_state=42, group=None, groupaffine= 1):

    data = os.path.join(data_dir, 'data.csv')
    data = pd.read_csv(data)
    data.sort_values(by=['Distorted'], inplace=True)
    
    if group:
        print('Grouping')
        if groupaffine > 1:
           print('Groups transformations are online')
           n = len(data)
           img_fn = []
           q_val = []
           lmax = []
           gpa = []
           for i in range(0, n):
               tmp0 = data.iloc[i].Distorted
               tmp1 = data.iloc[i].Q
               tmp2 = data.iloc[i].Lmax
               
               for j in range(0, groupaffine):
                   img_fn.append(tmp0)
                   q_val.append(tmp1)
                   lmax.append(tmp2)
                   gpa.append(j)
           d = {'Distorted': img_fn, 'Lmax': lmax, 'Q': q_val, 'I': gpa}
           data = pd.DataFrame(data=d)
           group = group * groupaffine
        else:
            print('Groups are precomputed')
    
        data = [data[i:i + group] for i in range(0, len(data), group)]
    else:
        print('No grouping')
       
        
    #split data into 80% train, 10% validation, and 10% test
    train, valtest = train_test_split(data, test_size=0.2, random_state=random_state)
    val, test = train_test_split(valtest, test_size=0.5, random_state=random_state)
    
    if group:
        train = pd.concat(train)
        val = pd.concat(val)
        test = pd.concat(test)
    
    #
    #
    #
    print(len(train))
    q_tra, h_tra = getVec(train)
    q_val, h_val = getVec(val)
    q_tes, h_tes = getVec(test)
    
    plt.clf()
    sns.distplot(q_tra, kde=True, rug=True, bins=100)
    plt.savefig('hist_q_train0.png')
    plt.clf()
    sns.distplot(q_val, kde=True, rug=True, bins=100)
    plt.savefig('hist_q_val0.png')
    plt.clf()
    sns.distplot(q_tes, kde=True, rug=True, bins=100)
    plt.savefig('hist_q_test0.png')
    
    #
    #
    #
    
    train = filterData(train, groupaffine)
    val = filterData(val, groupaffine)
    test = filterData(test, groupaffine)
    
    #train = pd.concat(train)
    #val = pd.concat(val)
    #test = pd.concat(test)

    #
    #
    #
    q_tra, h_tra = getVec(train)
    q_val, h_val = getVec(val)
    q_tes, h_tes = getVec(test)

    plt.clf()
    sns.distplot(q_tra, kde=True, rug=True, bins=100)
    plt.savefig('hist_q_train1.png')
    plt.clf()
    sns.distplot(q_val, kde=True, rug=True, bins=100)
    plt.savefig('hist_q_val1.png')
    plt.clf()
    sns.distplot(q_tes, kde=True, rug=True, bins=100)
    plt.savefig('hist_q_test1.png')


    #if group:
    #    train = pd.concat(train)
    #    val = pd.concat(val)
    #    test = pd.concat(test)

    return train, val, test

#
#
#
class HdrVdpDataset(Dataset):
    
    #
    #
    #
    def __init__(self, data, base_dir, group = None, groupaffine=1, bScaling = False, grayscale = True):
        self.data = data
        self.base_dir = base_dir
        self.groupaffine = groupaffine
        self.group = group
        self.bScaling = bScaling
        self.grayscale = grayscale

    #
    #
    #
    def __getitem__(self, index):
        sample = self.data.iloc[index]
        stim = self.base_dir
        #stim = os.path.join(stim, 'stim')

        fn = os.path.join(stim, sample.Distorted)
        stim = read_img_cv2(fn, maxClip = sample.Lmax, grayscale = self.grayscale)
            
        if self.bScaling:
            q = torch.FloatTensor([sample.Q / 100.0])
        else:
            q = torch.FloatTensor([sample.Q])
            
        if self.bScaling:
            lmax = torch.FloatTensor([sample.Lmax / 10000.0])
        else:
            lmax = torch.FloatTensor([sample.Lmax])
            
        if self.group != None:
            if self.groupaffine > 1:
                stim = torchDataAugmentation(stim, sample.I)

        return stim, q, lmax

    #
    #
    #
    def __len__(self):
        return len(self.data)
