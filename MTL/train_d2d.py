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


##1 boost 2
def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model',type=int,help='index of model')
    parser.add_argument('--batch_size',type=int,default=256)
    parser.add_argument('--num1',type=int,help='source dataset') #1-8:[stitch,drugbank,repur,pdx,gdsc,ccle,sider,offsides]
    parser.add_argument('--num2',type=int,help='target dataset')
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
    paths_all[0] = paths[0]+'data_900.npy' #paths to the entire dataset

    paths2 = [i+'train.npy' for i in paths]
    paths = [i+'test.npy' for i in paths]
    paths = [dataset_single(i,32) for i in paths]           #test
    paths2 = [dataset_single(i,batch_size) for i in paths2] #train
    paths_all = [dataset_single(i,batch_size) for i in paths_all]  #entire dataset
    loader_singles = [DataLoader(i,shuffle=False,batch_size=32) for i in paths]     #test
    loader_singles2 = [DataLoader(i,shuffle=True,batch_size=batch_size,num_workers=0) for i in paths2]  #train
    loader_all = [DataLoader(i,shuffle=True,batch_size=batch_size) for i in paths_all] #source

    loader1_train,loader1_test = loader_singles2[num1-1],loader_singles[num1-1]
    loader2_train,loader2_test = loader_singles2[num2-1],loader_singles[num2-1]
#     args.iter1 = len(loader1_train)*20
#     args.iter2 = len(loader2_train)*20
#     iteration = args.iter2+5
    num_to_name = {1:'stitch',2:'drugbank',3:'repurposing_hub',4:'pdx',5:'gdsc',6:'ccle',7:'sider',8:'offside'}

    if num1 in [4,5,6]:
        criterion1 = criterions[1]
    elif num1 in [-1,1,2,3]:
        criterion1 = criterions[0]
    else:
        criterion1 = criterions[2]
    if num2 in [4,5,6]:
        criterion2 = criterions[1]
    elif num2 in [-1,1,2,3]:
        criterion2 = criterions[0]
    else:
        criterion2 = criterions[2]

    def performance4(model,loader,num,batch_size=32):
        model.eval()
        rep=1
        y_pres = np.zeros((len(loader)*batch_size,rep))
        y_trues = np.zeros((len(loader)*batch_size,1))
        used = []
        with torch.no_grad():
            for index in range(rep):
                for i,(x,y,z) in tqdm(enumerate(loader)):
                    x,y = x.to(device),y.to(device)
                    used = used + list(z.detach().cpu().numpy()[:,0])
                    y_ = model(x1=x.float(),num=num)
    #                 if num == num1:
    #                     y_ = y_ + model2(x1=x.float(),num=num)
    #                 else :
    #                     y_ = y_ + model3(x1=x.float(),num=num)
                    if num not in [4,5,6]:
                        y_[y_>=0.5] = 1
                        y_[y_<0.5] = 0
                    for k in range(batch_size):
    #                     y_pres[z[k,0].item(),index] = y_.detach().cpu().numpy()[k,:].argmax() if num not in [4,5,6] else y_.detach().cpu().numpy()[k,0]
                        y_pres[z[k,0].item(),index] = y_.detach().cpu().numpy()[k,0]
                        if index == 0:
                            y_trues[z[k,0].item(),0] = y.cpu().numpy()[k,0]
        if num not in [4,5,6]:
            y_pre = mode(y_pres,axis=1)[0][:,0]
            y_true = y_trues[:,0]
            result1,result2 = AUROC(y_true,y_pre),AUPR(y_true,y_pre)
        else :
            y_pre = np.mean(y_pres,axis=1)
            y_true = y_trues[:,0]
            result1 = spearmanr(y_true,y_pre)[0]
            result2 = MSE(y_pre,y_true)
        return result1,result2


    def TRAIN2(model,epochs):
        model.train()
        running_loss = 0
        loss1_,loss2_,loss3_=0,0,0
        weight1,weight2 = args.weight1,args.weight2
        len1,len2 = len(loader_all[num1-1]),len(loader_singles2[num2-1])
        count1,count2 = 0,0
        loader1,loader2 = iter(loader_all[num1-1]),iter(loader_singles2[num2-1]) #loader1 is the source dataset
        loss1_train,loss2_train = [],[]
        i=-1 #index of iteration
        for e in range(epochs):
            for _,(x2,y2,_) in tqdm(enumerate(loader_singles2[num2-1])):  #iterate through the target dataset
                i += 1
                if count1 >= len1:
        #             loader1 = iter(loader_singles2[num1-1])
                    loader1 = iter(loader_all[num1-1])
                    count1 = 0
                count1,count2= count1+1,count2+1
                x,y,_ = loader1.next()
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
            a,b = performance4(model,loader_singles[args.num2-1],args.num2)
            with open('results/{}_d2d.txt'.format(str(args.num2)),'a') as f:
                f.write('\n')
                f.write(str(args.num1)+str(args.num2)+str(args.num2)+'\n')
                f.write(str(a)+','+str(b) + '\n')

    TRAIN2(model,args.epochs)

if __name__ == '__main__':
    main()
