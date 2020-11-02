import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
import torchvision.datasets as dset
from tqdm import tqdm
import torchvision
import numpy as np
from dataset import *
from model import *
from scipy.stats import mode
import argparse

def main():
##1 boost 2
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model',type=int,help='index of model')
    parser.add_argument('--batch_size',type=int,default=256)
    parser.add_argument('--num1',type=int,help='source task:1-dti,2-d_response,3-drug_se')
    parser.add_argument('--num2',type=int,help='target task')
    parser.add_argument('--epochs',type=int,default=20)
    parser.add_argument('--dynamic',type=str,default='',help='Dynamic weight adjustment strategy,options are LBTW and DWA')
    parser.add_argument('--info',type=str,help='the information you want to write in the output file')
    parser.add_argument('--record',type=int,default=1,help='whether to record the performance,options are 1 and 0')
    parser.add_argument('--weight1',type=float,default=1,help='initial weight for source task')
    parser.add_argument('--weight2',type=float,default=1,help='initial weight for target task')

    args = parser.parse_args()
    num1,num2 = args.num1,args.num2
    model_num = args.model
    info = args.info
    record = True if args.record == 1 else False
    # model = MTL_20()
    model = eval('MTL_'+str(model_num)+'()')
    model = model.to('cuda:0')
    batch_size,device=args.batch_size,'cuda:0'
    optimizer = optim.Adam(model.parameters(),lr = 1e-3)
    criterions = [nn.BCELoss(),nn.MSELoss(),nn.BCELoss()]
    paths = ['../Drug_target/stitch/','../Drug_target/drugbank/','../Drug_target/repurposing_hub/',
                '../Drug_response/pdx/',
                '../Drug_response/gdsc/','../Drug_response/ccle/',
                '../Drug_se/sider/','../Drug_se/offside/']
    paths_all = [i+'data.npy' for i in paths]
    paths_all[0] = paths[0]+'data_900.npy'
    dset1 = dataset_whole(paths_all,args.num1,args.batch_size) #use entire source task data

    paths2 = [i+'train.npy' for i in paths]
    dset2_train = dataset_whole(paths2,args.num2,args.batch_size) #training set of target task
    paths = [i+'test.npy' for i in paths]
    dset2_test = dataset_whole(paths,args.num2,32) #test set of target task
    loader1 = DataLoader(dset1,shuffle=True,batch_size=args.batch_size)
    loader2_train = DataLoader(dset2_train,shuffle=True,batch_size=args.batch_size)
    loader2_test = DataLoader(dset2_test,shuffle=False,batch_size=32)
    num_to_name = {1:'stitch',2:'drugbank',3:'repurposing_hub',4:'pdx',5:'gdsc',6:'ccle',7:'sider',8:'offside'}

    if num1 == 2:
        criterion1 = criterions[1]
        num1 = 4
    elif num1 ==1:
        criterion1 = criterions[0]
        num1 = 1
    else:
        criterion1 = criterions[2]
        num1 = 7
    if num2 == 2:
        criterion2 = criterions[1]
        num2 = 4
    elif num2 == 1:
        criterion2 = criterions[0]
        num2 = 1
    else:
        criterion2 = criterions[2]
        num2 = 7

    def performance4(model,loader,num,batch_size=32):
        '''
        Compute the performance
        '''
        model.eval()
        rep=1
    #     dset1,dset2 = loader_tests[1],loader_tests[2]
    #     len1,len2 = len(dset1),len(dset2)
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


    def TRAIN2(model,loader1,loader2_train,loader2_test,epochs=20):
        '''
        loader1: dataloader for source task
        loader2_train(test): dataloader for the training(test) set of target task
        '''
        model.train()
        running_loss = 0
        loss1_,loss2_,loss3_=0,0,0
        weight1,weight2 = args.weight1,args.weight2
        len1 = len(loader1)
        count1,count2 = 0,0
        loader = iter(loader1)     #gives batch samples of source task
        loss1_train,loss2_train = [],[]
        i=-1                      #iteration index (used for DWA)
        for e in range(epochs):
            for _,(x2,y2,_) in tqdm(enumerate(loader2_train)):
                i += 1
                if count1 >= len1:
                    loader = iter(loader1)
                    count1 = 0
                count1,count2= count1+1,count2+1
                x,y,_ = loader.next()
                x,y,x2,y2 = x.to(device),y.to(device),x2.to(device),y2.to(device)
                y_,y_2,ms = model(x1=x.float(),x2=x2.float(),num1=num1,num2=num2)
                loss1 = criterion1(y_,y)
                loss2 = criterion2(y_2,y2)
                if args.dynamic == 'LBTW':
                    if i == 0:
                        loss1_0,loss2_0 = loss1.item(),loss2.item()
        #                 print([loss1_0,loss2_0])
                    if i >=1:
        #                 print(np.sqrt(loss1.item()/loss1_0))
                        weight1 = np.sqrt(loss1.item()/loss1_0)
                        weight2=np.sqrt(loss2.item()/loss2_0)
                        sum_ = weight1 + weight2
                        weight1 = weight1/sum_
                        weight2 = weight2/sum_
        #                     weights = weights / np.sum(weights)
        #                 print([loss1.item(),loss2.item()])

                if args.dynamic == 'DWA':
                    loss1_train.append(loss1.item())
                    loss2_train.append(loss2.item())
                    if i >= 2:
                        weight1 = float(loss1_train[i-1])/float(loss1_train[i-2])
                        weight2 = float(loss2_train[i-1])/float(loss2_train[i-2])
                        weights = np.array([weight1,weight2])
                        sum_ = np.sum(np.exp(weights/2))
                        weights = 2 * np.exp(weights/2) / sum_
                        weight1 = weights[0]
                        weight2 = weights[1]

                loss = weight1 * loss1 + weight2*loss2
                running_loss+= loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if record:
            a,b = performance4(model,loader2_test,num2)
            with open('results/{}_t2t.txt'.format(str(args.num2)),'a') as f:
                f.write('\n')
                f.write(str(args.num1)+str(args.num2)+str(args.num2)+'\n')
                f.write(str(a)+','+str(b) + '\n')

    TRAIN2(model,loader1,loader2_train,loader2_test,args.epochs)


if __name__ == '__main__':
    main()
