import sys
sys.path.append('../MTL/')
import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms as T
import torchvision.datasets as dset
from tqdm import tqdm
import torchvision
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import roc_auc_score as AUROC
from sklearn.metrics import average_precision_score as AUPR
from model import *
from dataset import *
import argparse

def performance4(model,loader,num,batch_size=32):
    '''
    Compute the performance
    '''
    model.eval()
    rep=1
    device = 'cuda:0'
    with torch.no_grad():
        for index in range(rep):
            y_pre,y_true = [],[]
            for i,(x,y,_) in tqdm(enumerate(loader)):
                x,y = x.to(device),y.to(device)
                y_ = model(x1=x.float(),num=num)
                if num not in [4,5,6]:
                    y_pre = y_pre + list(y_.detach().cpu().numpy())
                else :
                    y_pre = y_pre + list(y_.detach().cpu().numpy()[:,0])
                y_true = y_true + list(y.cpu().numpy()[:,0])
            y_pre,y_true = np.array(y_pre),np.array(y_true)
            if num not in [4,5,6]:
                y_pre[y_pre>=0.5] = 1
                y_pre[y_pre<0.5] = 0

    if num not in [4,5,6]:
        result1,result2 = AUROC(y_true,y_pre),AUPR(y_true,y_pre)
    else :
        result1 = spearmanr(y_true,y_pre)[0]
        result2 = MSE(y_pre,y_true)
    return result1,result2

def train_single(model,epochs,loader_train,loader_test,optimizer,criterion,num,task=-1,record=False,record_model=False):
    '''
    num : integer, 1-8 representing 8 datasets
    '''
    device = 'cuda:0'
    model = model.to(device)
    for e in range(epochs):
        print('begin epoch: ',e+1)
        model.train()
        for i,(x,y,_) in tqdm(enumerate(loader_train)):
            x,y = x.to(device),y.to(device)

            y_ = model(x1=x.float(),num=num)
            loss = criterion(y_,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    if record:
        a,b = performance4(model,loader_test,num)
        if task == -1:
            with open('results/STL_{}.txt'.format(str(num)),'a') as f:
                f.write('\n')
                f.write(str(a)+','+str(b) + '\n')
        else :
            #1:dti,2:d_response,3:d_se
            with open('results/STL_task_{}.txt'.format(str(task)),'a') as f:
                f.write('\n')
                f.write(str(a)+','+str(b) + '\n')

    if record_model:
        torch.save(model.state_dict(),'models/STL_{0}.pth'.format(str(num)))

def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--epochs',type=int,default=20,help='the number of epoch')
    parser.add_argument('--batch_size',type=int,default=256,help='batch size')
    parser.add_argument('--record',type=int,default=0,help='0:not record,1:record,whether to record the performances')
    parser.add_argument('--model',type=int,default=0,help='the model to use')
    parser.add_argument('--single',type=int,default=-1,help='the dataset you want to train,1-9')
    parser.add_argument('--task',type=int,default=-1,help='1-3, to train the task with datasets combined. 1:dti,2:d_response,3:d_se')
    parser.add_argument('--record_model',type=int,default=0,help='0:not to record,1:record the model')
    args = parser.parse_args()
    batch_size = args.batch_size
    epoch = args.epochs
    record = True if args.record == 1 else False
    model_num = args.model
    single = args.single
    record_model = True if args.record_model == 1 else False
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = eval('MTL_'+str(model_num)+'()')

    optimizer = optim.Adam(model.parameters(),lr = 1e-3)
    criterions = [nn.BCELoss(),nn.MSELoss(),nn.BCELoss()]
    paths = ['../Drug_target/stitch/','../Drug_target/drugbank/','../Drug_target/repurposing_hub/',
            '../Drug_response/pdx/',
            '../Drug_response/gdsc/','../Drug_response/ccle/',
            '../Drug_se/sider/','../Drug_se/offside/']

    name = paths[single-1].split('/')[2]
    paths2 = [i+'train.npy' for i in paths]
    paths = [i+'test.npy' for i in paths]
    dset_train,dset_test = dataset_single(paths2[single-1],batch_size),dataset_single(paths[single-1],batch_size=32)
    loader_train,loader_test = DataLoader(dset_train,shuffle=True,batch_size=batch_size),DataLoader(dset_test,shuffle=False,batch_size=32)
    criterion = criterions[(single-1) // 3]
    if args.task ==-1:
        #train single dataset, 1-8:8 datasets
        train_single(model,epoch,loader_train,loader_test,optimizer,criterion,args.single,record=record,record_model=record_model)  #single is num
    else :
        # train task with datasets combined. (e.g. dti:stitch+drugbank+repur)
        paths = ['../Drug_target/stitch/','../Drug_target/drugbank/','../Drug_target/repurposing_hub/',
        '../Drug_response/pdx/',
        '../Drug_response/gdsc/','../Drug_response/ccle/',
        '../Drug_se/sider/','../Drug_se/offside/']
        paths_train = [i+'train.npy' for i in paths]
        paths_test = [i+'test.npy' for i in paths]
        dset_train = dataset_whole(paths_train,args.task,args.batch_size)
        dset_test = dataset_whole(paths_test,args.task,args.batch_size)
        loader_train = DataLoader(dset_train,shuffle=True,batch_size=args.batch_size)
        loader_test = DataLoader(dset_test,shuffle=False,batch_size=32)
        if args.task == 1:
            num = 1
            criterion = criterions[0]
        elif args.task == 2:
            num = 4
            criterion = criterions[1]
        else :
            num = 7
            criterion = criterions[2]
        assert args.task in [1,2,3]
        train_single(model,epoch,loader_train,loader_test,optimizer,
                     criterion,args.task,record=record,record_model=record_model,task=args.task)

if __name__ == '__main__':
    main()
