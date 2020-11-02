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
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model',type=int,help='index of model')
    parser.add_argument('--batch_size',type=int,default=256)
    parser.add_argument('--epochs',type=int,default=20)
    parser.add_argument('--dynamic',type=str,default='',help='Dynamic weight adjustment strategy,options are LBTW and DWA')
    parser.add_argument('--info',type=str,help='the information you want to write in the output file')
    parser.add_argument('--T',type=float,default=2,help='The T value in DWA stategy')
    parser.add_argument('--record',type=int,default=1,help='whether to record the performance,options are 1 and 0')
    parser.add_argument('--record_model',type=int,default=1,help='whether to record the model')

    args = parser.parse_args()
    record_model = True if args.record_model == 1 else False
    model_num = args.model
    info = args.info
    record = True if args.record == 1 else False
    # model = MTL_20()
    model = eval('MTL_'+str(model_num)+'()')
    model = model.to('cuda:0')
    batch_size,device=args.batch_size,'cuda:0'
    optimizer = optim.Adam(model.parameters(),lr = 1e-3)
    criterions = [nn.BCELoss(),nn.MSELoss(),nn.BCELoss()] ##here
    paths = ['../Drug_target/stitch/','../Drug_target/drugbank/','../Drug_target/repurposing_hub/',
                '../Drug_response/pdx/',
                '../Drug_response/gdsc/','../Drug_response/ccle/',
                '../Drug_se/sider/','../Drug_se/offside/']
    paths2 = [i+'train.npy' for i in paths]
    paths = [i+'test.npy' for i in paths]
    paths = [dataset_single(i,32) for i in paths]           #test
    paths2 = [dataset_single(i,batch_size) for i in paths2] #train
    loader_singles = [DataLoader(i,shuffle=False,batch_size=32) for i in paths]           #test
    loader_singles2 = [DataLoader(i,shuffle=True,batch_size=batch_size,num_workers=0) for i in paths2] #train
    num_to_name = {1:'stitch',2:'drugbank',3:'repurposing_hub',4:'pdx',5:'gdsc',6:'ccle',7:'sider',8:'offside'}
    criterion1 = criterions[0]
    criterion2 = criterions[1]
    criterion3 = criterions[2]
    lens = [len(loader_singles2[i]) for i in range(8)]   #number of batchs in one epoch
    lens = [k*args.epochs for k in lens]       #total number of batchs for args.epochs
    max_len = np.max(lens)

    def performance4(model,loader,num,batch_size=32):
        '''
        compute the performance metrics
        loader: dataloader for test set
        num : index of dataset you want to evaluate. 1-8 refers to the 'num_to_name' dict above
        '''
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


    def TRAIN2(model,iteration):
        '''
        Training method.
        iteration : max iteration for largest dataset.
        '''
        model.train()
        running_loss = 0
        loss1_,loss2_,loss3_=0,0,0

        #keep the length of each dataset  (can use list to simplify the code)
        len1,len2 = len(loader_singles2[0]),len(loader_singles2[1])
        count1,count2 = 0,0
        loader1,loader2 = iter(loader_singles2[0]),iter(loader_singles2[1])
        len3,len4 = len(loader_singles2[2]),len(loader_singles2[3])
        count3,count4 = 0,0
        loader3,loader4 = iter(loader_singles2[2]),iter(loader_singles2[3])
        len5,len6 = len(loader_singles2[4]),len(loader_singles2[5])
        count5,count6 = 0,0
        loader5,loader6 = iter(loader_singles2[4]),iter(loader_singles2[5])
        len7,len8 = len(loader_singles2[6]),len(loader_singles2[7])
        count7,count8 = 0,0
        loader7,loader8 = iter(loader_singles2[6]),iter(loader_singles2[7])
        loss1_train,loss2_train = [],[]
        loss3_train,loss4_train = [],[]
        loss5_train,loss6_train = [],[]
        loss7_train,loss8_train = [],[]
        weights=np.array([1.1,1,1,1,1,1,1,1])  #initial weight
        for i in range(iteration):
            model.train()
            #to keep sampling data from each dataset:
            if count1 >= len1:
                loader1 = iter(loader_singles2[0])
                count1 = 0
            if count2 >= len2:
                loader2 = iter(loader_singles2[1])
                count2= 0
            if count3 >= len3:
                loader3 = iter(loader_singles2[2])
                count3 = 0
            if count4 >= len4:
                loader4 = iter(loader_singles2[3])
                count4 = 0
            if count5 >= len5:
                loader5 = iter(loader_singles2[4])
                count5 = 0
            if count6 >= len6:
                loader6 = iter(loader_singles2[5])
                count6 = 0
            if count7 >= len7:
                loader7 = iter(loader_singles2[6])
                count7 = 0
            if count8 >= len8:
                loader8 = iter(loader_singles2[7])
                count8 = 0
            count1,count2= count1+1,count2+1
            count3,count4,count5,count6,count7,count8 = count3+1,count4+1,count5+1,count6+1,count7+1,count8+1
            #sampling
            x1,y1,_ = loader1.next()
            x2,y2,_ = loader2.next()
            x3,y3,_ = loader3.next()
            x4,y4,_ = loader4.next()
            x5,y5,_ = loader5.next()
            x6,y6,_ = loader6.next()
            x7,y7,_ = loader7.next()
            x8,y8,_ = loader8.next()
            x1,y1,x2,y2 = x1.to(device),y1.to(device),x2.to(device),y2.to(device)
            x3,y3,x4,y4 = x3.to(device),y3.to(device),x4.to(device),y4.to(device)
            x5,y5,x6,y6 = x5.to(device),y5.to(device),x6.to(device),y6.to(device)
            x7,y7,x8,y8 = x7.to(device),y7.to(device),x8.to(device),y8.to(device)
            y_1,y_2,y_3,y_4,y_5,y_6,y_7,y_8 = model(x1.float(),x2.float(),x3.float(),x4.float(),x5.float(),x6.float(),x7.float(),x8.float())
    #         y_1,y_2,y_3,y_4,y_5,y_7,y_8 = model(x1=x1.float(),x2=x2.float(),x3=x3.float(),x4=x4.float(),x5=x5.float(),x7=x7.float(),x8=x8.float())   #####here
            loss1 = criterion1(y_1,y1)
            loss2 = criterion1(y_2,y2)
            loss3 = criterion1(y_3,y3)
            loss4 = criterion2(y_4,y4)
            loss5 = criterion2(y_5,y5)
            loss6 = criterion2(y_6,y6)   ###here
            loss7 = criterion3(y_7,y7)
            loss8 = criterion3(y_8,y8)
            if args.dynamic == 'DWA':
                loss1_train.append(loss1.item())
                loss2_train.append(loss2.item())
                loss3_train.append(loss3.item())
                loss4_train.append(loss4.item())
                loss5_train.append(loss5.item())
                loss6_train.append(loss6.item()) ###here
                loss7_train.append(loss7.item())
                loss8_train.append(loss8.item())
                if i >= 2:
                    for k in range(8):
    #                     if k+1 == 6:
    #                         continue  ###here
                        weights[k] = float(eval('loss{0}_train[i-1]'.format(str(k+1))))/float(eval('loss{0}_train[i-2]'.format(str(k+1))))
                    sum_ = np.sum(np.exp(weights/args.T))
                    weights = 8 * np.exp(weights/args.T) / sum_
            loss = weights[0]*loss1+weights[1]*loss2+weights[2]*loss3+weights[3]*loss4+weights[4]*loss5+weights[5]*loss6+weights[6]*loss7+weights[7]*loss8
            running_loss+= loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if not record:
                if i % 5000 == 0 and i != 0:
                    a,b = performance4(model,loader_singles[1-1],1)
                    print(str(a)+','+str(b) + '\n')
            if record:
                if i == lens[0]:
                    a,b = performance4(model,loader_singles[1-1],1)
                    with open('{}_MTL.txt'.format(str(1)),'a') as f:
                        f.write('\n')
                        f.write(args.info+'\n')
                        f.write(str(a)+','+str(b) + '\n')
                    if record_model:
                        torch.save(model.state_dict(),'models/MTL_{0}.pth'.format(str(1)))
                elif i == lens[1]:
                    a,b = performance4(model,loader_singles[2-1],2)
                    with open('{}_MTL.txt'.format(str(2)),'a') as f:
                        f.write('\n')
                        f.write(args.info+'\n')
                        f.write(str(a)+','+str(b) + '\n')
                    if record_model:
                        torch.save(model.state_dict(),'models/MTL_{0}.pth'.format(str(2)))
                elif i == lens[2]:
                    a,b = performance4(model,loader_singles[3-1],3)
                    with open('{}_MTL.txt'.format(str(3)),'a') as f:
                        f.write('\n')
                        f.write(args.info+'\n')
                        f.write(str(a)+','+str(b) + '\n')
                    if record_model:
                        torch.save(model.state_dict(),'models/MTL_{0}.pth'.format(str(3)))
                elif i == lens[3]:
                    a,b = performance4(model,loader_singles[4-1],4)
                    with open('{}_MTL.txt'.format(str(4)),'a') as f:
                        f.write('\n')
                        f.write(args.info+'\n')
                        f.write(str(a)+','+str(b) + '\n')
                    if record_model:
                        torch.save(model.state_dict(),'models/MTL_{0}.pth'.format(str(4)))
                if i == lens[4]:
                    a,b = performance4(model,loader_singles[5-1],5)
                    with open('{}_MTL.txt'.format(str(5)),'a') as f:
                        f.write('\n')
                        f.write(args.info+'\n')
                        f.write(str(a)+','+str(b) + '\n')
                    if record_model:
                        torch.save(model.state_dict(),'models/MTL_{0}.pth'.format(str(5)))
                if i == lens[5]:
                    a,b = performance4(model,loader_singles[6-1],6)
                    with open('{}_MTL.txt'.format(str(6)),'a') as f:
                        f.write('\n')
                        f.write(args.info+'\n')
                        f.write(str(a)+','+str(b) + '\n')
                    if record_model:
                        torch.save(model.state_dict(),'models/MTL_{0}.pth'.format(str(6)))
                if i == lens[6]:
                    a,b = performance4(model,loader_singles[7-1],7)
                    with open('{}_MTL.txt'.format(str(7)),'a') as f:
                        f.write('\n')
                        f.write(args.info+'\n')
                        f.write(str(a)+','+str(b) + '\n')
                    if record_model:
                        torch.save(model.state_dict(),'models/MTL_{0}.pth'.format(str(7)))
                if i == lens[7]:
                    a,b = performance4(model,loader_singles[8-1],8)
                    with open('{}_MTL.txt'.format(str(8)),'a') as f:
                        f.write('\n')
                        f.write(args.info+'\n')
                        f.write(str(a)+','+str(b) + '\n')
                    if record_model:
                        torch.save(model.state_dict(),'models/MTL_{0}.pth'.format(str(8)))

    TRAIN2(model,max_len+1)


if __name__ == '__main__':
    main()
