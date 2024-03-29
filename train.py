#
#Copyright (C) 2023 ISTI-CNR
#Licensed under the BSD 3-Clause Clear License (see license.txt)
#

import os
import sys
import argparse

import re
import glob2

import pandas as pd
import numpy as np

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from tqdm import tqdm, trange

import matplotlib.pyplot as plt
import seaborn as sns

from dataset import torchDataAugmentation
from util import plotGraph
from dataset import split_data, read_data_split, HdrVdpDataset
from model_classic import QNetC
from model_bn import QNetBN
from model_rz import QNetRZ
from model_res import QNetRes

#
# training for a single epoch
#
def train(loader, model, optimizer, args):
    model.train()
    
    
    progress = tqdm(loader)
    total_loss = 0.0
    counter = 0
    for stim, q, lmax in progress:
        if torch.cuda.is_available():
            stim = stim.cuda()
            q = q.cuda()
            lmax = lmax.cuda()
                   
        q_hat = model(stim, lmax)
        
        loss = F.l1_loss(q_hat, q)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        counter += 1
        
        progress.set_postfix({'loss': total_loss / counter})

    return total_loss / counter;

#
#this function simply evaluate the model
#
def evaluate(loader, model, args):
    model.eval()
    
    total_loss = 0
    counter = 0
    progress = tqdm(loader)
    targets = []
    predictions = []

    for stim, q, lmax in progress:
        with torch.no_grad():
            if torch.cuda.is_available():
                stim = stim.cuda()
                q = q.cuda()
                lmax = lmax.cuda()

            q_hat = model(stim, lmax)
            loss = F.l1_loss(q_hat, q)
        
            counter += 1
            total_loss += loss.item()
            
            targets.append(q)
            predictions.append(q_hat)
            
            progress.set_postfix({'loss': total_loss / counter})
     
    targets = torch.cat(targets, 0).squeeze()
    predictions = torch.cat(predictions, 0).squeeze()

    total_loss /= counter
     
    return total_loss, targets, predictions

#
#
#
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Q regressor',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('data', type=str, help='Path to data dir')
    parser.add_argument('-g', '--group', type=int, help='grouping factor for augmented dataset')
    parser.add_argument('-gpa', '--groupaffine', type=int, default=1, help='grouping affine')
    parser.add_argument('-s', '--scaling', type=bool, default=False, help='scaling')
    parser.add_argument('-btype', type=int, default = 0, help='Base dir of run to evaluate')
    parser.add_argument('-e', '--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('-b', '--batch', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('-r', '--runs', type=str, default='runs/', help='Base dir for runs')
    parser.add_argument('--resume', default=None, help='Path to initial weights')
    parser.add_argument('--grayscale', type=int, default=1, help='Grayscale')
    args = parser.parse_args()
  
    args.grayscale = (args.grayscale == 1)
  
    ### Prepare run dir
    params = vars(args)
    params['dataset'] = os.path.basename(os.path.normpath(args.data))
    
    results_str = os.path.basename(os.path.normpath(args.data))
    print('Group: ' + str(args.group))
    print('Group Affine: ' + str(args.groupaffine))
    print('E: ' + str(args.epochs))
    print('LR: ' + str(args.lr))
    print('Batch: ' + str(args.batch))
    print('Batch: ' + str(args.batch))
    print('Model type: ' + str(args.btype))
    print('Scaling: ' + str(args.scaling))
    
    run_name = 'q_{0[dataset]}_lr{0[lr]}_e{0[epochs]}_b{0[batch]}_t{0[btype]}_g{0[grayscale]}'.format(params)
    run_dir = os.path.join(args.runs, run_name) 
    ckpt_dir = os.path.join(run_dir, 'ckpt')
    
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
        os.makedirs(ckpt_dir)
        
    if not os.path.exists('results_'+results_str):
        os.makedirs('results_'+results_str)
    
    log_file = os.path.join(run_dir, 'log.csv')
    param_file = os.path.join(run_dir, 'params.csv')
    pd.DataFrame(params, index=[0]).to_csv(param_file, index=False)
    
    ### Load Data
    if os.path.exists(os.path.join(args.data, 'train.csv')):
        print('Precomputed train/validation/test')
        train_data, val_data, test_data = read_data_split(args.data, args.group, args.groupaffine)
    else:
        print('Computing train/validation/test')
        train_data, val_data, test_data = split_data(args.data, group=args.group, groupaffine = args.groupaffine)
        train_data.to_csv(os.path.join(args.data, "train.csv"), ',')
        val_data.to_csv(os.path.join(args.data, "val.csv"), ',')
        test_data.to_csv(os.path.join(args.data, "test.csv"), ',')
        
        train_data.to_csv(os.path.join(run_dir, "train.csv"), ',')
        val_data.to_csv(os.path.join(run_dir, "val.csv"), ',')
        test_data.to_csv(os.path.join(run_dir, "test.csv"), ',')

    #create the loader for the training set
    train_data = HdrVdpDataset(train_data, args.data, group = args.group, groupaffine = args.groupaffine, bScaling = args.scaling, grayscale = args.grayscale)
    train_loader = DataLoader(train_data, shuffle=True, batch_size=args.batch, num_workers=8, pin_memory=True)
    #create the loader for the validation set
    val_data = HdrVdpDataset(val_data, args.data, group = args.group, groupaffine = args.groupaffine, bScaling = args.scaling, grayscale = args.grayscale)
    val_loader = DataLoader(val_data, shuffle=False, batch_size=args.batch, num_workers=8, pin_memory=True)
    #create the loader for the testing set
    test_data = HdrVdpDataset(test_data, args.data, group = args.group, groupaffine = args.groupaffine, bScaling = args.scaling, grayscale = args.grayscale)
    test_loader = DataLoader(test_data, shuffle=False, batch_size=1, num_workers=8, pin_memory=True)

    if args.grayscale:
        n_in = 1
    else:
        n_in = 3
        
    params_size_net = None
    
    out_str = ''
    if args.btype == 0:
        model = QNetC(n_in, 1, params_size = params_size_net)
        out_str = 'c'
    elif args.btype == 1:
        model = QNetBN(n_in, 1, params_size = params_size_net, layer_norm = 0)
        out_str = 'bn'
    elif args.btype == 2:
        model = QNetRZ(n_in, 1, params_size = params_size_net)
        out_str = 'rz'
    elif args.btype == 3:
        model = QNetRes(n_in, 1, params_size = params_size_net, whichResnet = 18)
        out_str = 'res18'

    #create the model
    if(torch.cuda.is_available()):
        model = model.cuda()

    #create the optmizer
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, patience=5, factor=0.5, verbose=True)
    log = pd.DataFrame()
    
    #training loop
    best_mse = None
    a_t = []
    a_v = []
    a_te = []

    start_epoch = 1
    if args.resume != None:
    
       if args.resume == 'same':
          ckpt_dir_r = ckpt_dir
       else:
          ckpt_dir_r = os.path.join(args.resume, 'ckpt')
       ckpts = glob2.glob(os.path.join(ckpt_dir_r, '*.pth'))
       assert ckpts, "No checkpoints to resume from!"
    
       def get_epoch(ckpt_url):
           s = re.findall("ckpt_e(\d+).pth", ckpt_url)
           epoch = int(s[0]) if s else -1
           return epoch, ckpt_url
    
       start_epoch, ckpt = max(get_epoch(c) for c in ckpts)
       print('Checkpoint:', ckpt)
       ckpt = torch.load(ckpt)
       model.load_state_dict(ckpt['model'])
       start_epoch = ckpt['epoch']
       best_mse = ckpt['mse_val']
    
    
    for epoch in trange(start_epoch, args.epochs + 1):
        cur_loss = train(train_loader, model, optimizer, args)
        val_loss, targets_v, predictions_v = evaluate(val_loader, model, args)
        test_loss,  targets_t, predictions_t = evaluate(test_loader, model, args)

        metrics = {'epoch': epoch}
        metrics['mse_train'] = cur_loss
        metrics['mse_val'] = val_loss
        metrics['mse_test'] = test_loss
        log = log.append(metrics, ignore_index=True)
        log.to_csv(log_file, index=False)
        
        a_t.append(cur_loss)
        a_v.append(val_loss)
        a_te.append(test_loss)

        if (best_mse is None) or (val_loss < best_mse) or (epoch == args.epochs):
            delta = (targets_t - predictions_t)
            errors = delta.cpu().numpy()
            targets_t = targets_t.cpu().numpy()
            predictions_t = predictions_t.cpu().numpy()
                        
            sz = errors.shape
            errors = np.reshape(errors, (sz[0], 1))
            predictions_t = np.reshape(predictions_t, (sz[0], 1))
            targets_t = np.reshape(targets_t, (sz[0], 1))
            mtx = np.concatenate((targets_t, predictions_t, errors), axis=1)
            
            np.savetxt(os.path.join(run_dir, 'errors_' + out_str + '.txt'), mtx, fmt='%f')
            np.savetxt(os.path.join('results_'+results_str, 'errors_' + out_str + '.txt'), mtx, fmt='%f')            
            plt.clf()
            sns.distplot(errors, kde=True, rug=True)
            plt.savefig('results_'+results_str+'/hist_errors_test_' +  out_str + '.png')
            plt.savefig(os.path.join(run_dir, 'hist_errors_test_' +  out_str + '.png'))

            name_f = 'plot_' + out_str + '.png'
            plotGraph(a_t, a_v, a_te, 'results_'+results_str, name_f)
            plotGraph(a_t, a_v, a_te, run_dir, name_f)
            best_mse = val_loss
            print(ckpt_dir)
            ckpt = os.path.join(ckpt_dir, 'ckpt_e{}.pth'.format(epoch))
            torch.save({
                'epoch': epoch,
                'type': args.btype,
                'mse_train': cur_loss,
                'mse_val': val_loss,
                'mse_test': test_loss,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }, ckpt)
        scheduler.step(val_loss)
