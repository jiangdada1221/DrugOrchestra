import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import roc_auc_score as AUROC
from sklearn.metrics import average_precision_score as AUPR
from scipy.stats import spearmanr
# from torchvision import transforms as T
# import torchvision.datasets as dset

class MTL_26(nn.Module):
    '''
    Receive up to 3 input channels, used for investigating the transferability
    '''
    def __init__(self,single=False):
        super(MTL_26,self).__init__()
        self.d = nn.Linear(300,512)
#         self.d2 = nn.Linear(300,512)
        self.f1,self.f2,self.f3 = nn.Sequential(nn.Linear(800,256),nn.ReLU(),nn.Linear(256,128),nn.ReLU()),nn.Sequential(nn.Linear(300,256),nn.ReLU(),nn.Linear(256,128),nn.ReLU()),nn.Sequential(nn.Linear(300,256),nn.ReLU(),nn.Linear(256,128),nn.ReLU())
        self.dd = nn.Linear(512,128)
        self.c1,self.c2,self.c3 = self.classifier(256,1),self.classifier(256,1),self.classifier(256,1)
        self.name = 'MTL_26 '
    def classifier(self,In,out):
        return nn.Sequential(
                             nn.Linear(In,64),nn.ReLU(),
                             nn.Linear(64,16),nn.ReLU(),
                            nn.Linear(16,out))
    def forward(self,x1,x2=None,x3=None,num=-1,num1=-1,num2=-1,num3=-1):
        num_to_c = {1:self.c1,2:self.c2,3:self.c3}
        num_to_c2 = {1:self.c1,2:self.c1,3:self.c1,4:self.c2,5:self.c2,6:self.c2,7:self.c3,8:self.c3}
        num_to_f = {1:self.f1,2:self.f1,3:self.f1,4:self.f2,5:self.f2,6:self.f2,7:self.f3,8:self.f3}
        if x2!=None and x3 == None:
            assert num1!= -1 and num2 != -1

            d1,d2 = x1[:,0:300],x2[:,0:300]
            f1,f2 = x1[:,300:],x2[:,300:]

            x1,x2 = nn.ReLU()(self.d(d1)),nn.ReLU()(self.d(d2))
            x1 = nn.ReLU()(self.dd(x1))
            x2 = nn.ReLU()(self.dd(x2))
            f1,f2 = num_to_f[num1](f1),num_to_f[num2](f2)
            c1,c2 = num_to_c2[num1],num_to_c2[num2]
            x1,x2 = torch.cat((x1,f1),1),torch.cat((x2,f2),1)
            res1,res2 = c1(x1),c2(x2)
            if num1 not in [4,5,6]:
                res1 = nn.Sigmoid()(res1)
            if num2 not in [4,5,6]:
                res2 = nn.Sigmoid()(res2)
            return res1,res2,0
        elif x3!= None:
            #3 tasks
            d1,d2,d3 = x1[:,0:300],x2[:,0:300], x3[:,0:300]
            f1,f2,f3 = x1[:,300:],x2[:,300:],x3[:,300:]
            x1,x2,x3 = nn.ReLU()(self.d(d1)),nn.ReLU()(self.d(d2)),nn.ReLU()(self.d(d3))
            x1 = nn.ReLU()(self.dd(x1))
            x2 = nn.ReLU()(self.dd(x2))
            x3 = nn.ReLU()(self.dd(x3))
            f1,f2,f3 = num_to_f[num1](f1),num_to_f[num2](f2),num_to_f[num3](f3)
            c1,c2,c3 = num_to_c2[num1],num_to_c2[num2],num_to_c2[num3]
            x1,x2,x3 = torch.cat((x1,f1),1),torch.cat((x2,f2),1),torch.cat((x3,f3),1)
            return nn.Sigmoid()(c1(x1)),(c2(x2)),nn.Sigmoid()(c3(x3)),0
        else :
            assert num != -1
            d,f = x1[:,0:300],x1[:,300:]
            d = nn.ReLU()(self.d(d))
            d = nn.ReLU()(self.dd(d))
            f = num_to_f[num](f)
            c = num_to_c2[num]
            x1 = torch.cat((d,f),1)
            if num not in [4,5,6]:
                return nn.Sigmoid()(c(x1))
            else :
                return c(x1)

class MTL_33(nn.Module):
    '''
    Used for computing the inner transferability for drug target datasets
    '''
    def __init__(self,single=False):
        super(MTL_33,self).__init__()
        self.d = nn.Linear(300,512)
#         self.d2 = nn.Linear(300,512)
        self.f1,self.f2,self.f3 = nn.Sequential(nn.Linear(800,256),nn.ReLU(),nn.Linear(256,128),nn.ReLU()),nn.Sequential(nn.Linear(800,256),nn.ReLU(),nn.Linear(256,128),nn.ReLU()),nn.Sequential(nn.Linear(800,256),nn.ReLU(),nn.Linear(256,128),nn.ReLU())
        self.dd = nn.Linear(512,128)
        self.c1,self.c2,self.c3 = self.classifier(256,1),self.classifier(256,1),self.classifier(256,1)
        self.name = 'MTL_26-target '
    def classifier(self,In,out):
        return nn.Sequential(
                             nn.Linear(In,64),nn.ReLU(),
                             nn.Linear(64,16),nn.ReLU(),
                            nn.Linear(16,out))
    def forward(self,x1,x2=None,x3=None,num=-1,num1=-1,num2=-1,num3=-1):
        num_to_c = {1:self.c1,2:self.c2,3:self.c3}
        num_to_c2 = {1:self.c1,2:self.c2,3:self.c3}
        num_to_f = {1:self.f1,2:self.f2,3:self.f3}
        if x2!=None and x3 == None:
            assert num1!= -1 and num2 != -1

            d1,d2 = x1[:,0:300],x2[:,0:300]
            f1,f2 = x1[:,300:],x2[:,300:]

            x1,x2 = nn.ReLU()(self.d(d1)),nn.ReLU()(self.d(d2))
            x1 = nn.ReLU()(self.dd(x1))
            x2 = nn.ReLU()(self.dd(x2))
            f1,f2 = num_to_f[num1](f1),num_to_f[num2](f2)
            c1,c2 = num_to_c2[num1],num_to_c2[num2]
            x1,x2 = torch.cat((x1,f1),1),torch.cat((x2,f2),1)
            res1,res2 = c1(x1),c2(x2)
            if num1 not in [4,5,6]:
                res1 = nn.Sigmoid()(res1)
            if num2 not in [4,5,6]:
                res2 = nn.Sigmoid()(res2)
            return res1,res2,0
        elif x3!= None:
            d1,d2,d3 = x1[:,0:300],x2[:,0:300], x3[:,0:300]
            f1,f2,f3 = x1[:,300:],x2[:,300:],x3[:,300:]
            x1,x2,x3 = nn.ReLU()(self.d(d1)),nn.ReLU()(self.d(d2)),nn.ReLU()(self.d(d3))
            x1 = nn.ReLU()(self.dd(x1))
            x2 = nn.ReLU()(self.dd(x2))
            x3 = nn.ReLU()(self.dd(x3))
            f1,f2,f3 = num_to_f[num1](f1),num_to_f[num2](f2),num_to_f[num3](f3)
            c1,c2,c3 = num_to_c2[num1],num_to_c2[num2],num_to_c2[num3]
            x1,x2,x3 = torch.cat((x1,f1),1),torch.cat((x2,f2),1),torch.cat((x3,f3),1)
            return nn.Sigmoid()(c1(x1)),nn.Sigmoid()(c2(x2)),nn.Sigmoid()(c3(x3)),0
        else :
            assert num != -1
            d,f = x1[:,0:300],x1[:,300:]
            d = nn.ReLU()(self.d(d))
            d = nn.ReLU()(self.dd(d))
            f = num_to_f[num](f)
            c = num_to_c2[num]
            x1 = torch.cat((d,f),1)
            if num not in [4,5,6]:
                return nn.Sigmoid()(c(x1))
            else :
                return c(x1)

class MTL_34(nn.Module):
    '''
    Used for computing the inner transferability for drug response datasets
    '''
    def __init__(self,single=False):
        super(MTL_34,self).__init__()
        self.d = nn.Linear(300,512)
#         self.d2 = nn.Linear(300,512)
        self.f1,self.f2,self.f3 = nn.Sequential(nn.Linear(300,256),nn.ReLU(),nn.Linear(256,128),nn.ReLU()),nn.Sequential(nn.Linear(300,256),nn.ReLU(),nn.Linear(256,128),nn.ReLU()),nn.Sequential(nn.Linear(300,256),nn.ReLU(),nn.Linear(256,128),nn.ReLU())
        self.dd = nn.Linear(512,128)
        self.c1,self.c2,self.c3 = self.classifier(256,1),self.classifier(256,1),self.classifier(256,1)
        self.name = 'MTL_34 -response '
    def classifier(self,In,out):
        return nn.Sequential(
                             nn.Linear(In,64),nn.ReLU(),
                             nn.Linear(64,16),nn.ReLU(),
                            nn.Linear(16,out))
    def forward(self,x1,x2=None,x3=None,num=-1,num1=-1,num2=-1,num3=-1):
        num_to_c = {1:self.c1,2:self.c2,3:self.c3}
        num_to_c2 = {4:self.c1,5:self.c2,6:self.c3}
        num_to_f = {4:self.f1,5:self.f2,6:self.f3}
        if x2!=None and x3 == None:
            assert num1!= -1 and num2 != -1

            d1,d2 = x1[:,0:300],x2[:,0:300]
            f1,f2 = x1[:,300:],x2[:,300:]

            x1,x2 = nn.ReLU()(self.d(d1)),nn.ReLU()(self.d(d2))
            x1 = nn.ReLU()(self.dd(x1))
            x2 = nn.ReLU()(self.dd(x2))
            f1,f2 = num_to_f[num1](f1),num_to_f[num2](f2)
            c1,c2 = num_to_c2[num1],num_to_c2[num2]
            x1,x2 = torch.cat((x1,f1),1),torch.cat((x2,f2),1)
            res1,res2 = c1(x1),c2(x2)
            return res1,res2,0
        elif x3!= None:
            d1,d2,d3 = x1[:,0:300],x2[:,0:300], x3[:,0:300]
            f1,f2,f3 = x1[:,300:],x2[:,300:],x3[:,300:]
            x1,x2,x3 = nn.ReLU()(self.d(d1)),nn.ReLU()(self.d(d2)),nn.ReLU()(self.d(d3))
            x1 = nn.ReLU()(self.dd(x1))
            x2 = nn.ReLU()(self.dd(x2))
            x3 = nn.ReLU()(self.dd(x3))
            f1,f2,f3 = num_to_f[num1](f1),num_to_f[num2](f2),num_to_f[num3](f3)
            c1,c2,c3 = num_to_c2[num1],num_to_c2[num2],num_to_c2[num3]
            x1,x2,x3 = torch.cat((x1,f1),1),torch.cat((x2,f2),1),torch.cat((x3,f3),1)
            return (c1(x1)),(c2(x2)),(c3(x3)),0
        else :
            assert num != -1
            d,f = x1[:,0:300],x1[:,300:]
            d = nn.ReLU()(self.d(d))
            d = nn.ReLU()(self.dd(d))
            f = num_to_f[num](f)
            c = num_to_c2[num]
            x1 = torch.cat((d,f),1)
            if num not in [4,5,6]:
                return nn.Sigmoid()(c(x1))
            else :
                return c(x1)

class MTL_35(nn.Module):
    '''
    Used for drug side-effect
    '''
    def __init__(self,single=False):
        super(MTL_35,self).__init__()
        self.d = nn.Linear(300,512)
#         self.d2 = nn.Linear(300,512)
        self.f1,self.f2,self.f3 = nn.Sequential(nn.Linear(300,256),nn.ReLU(),nn.Linear(256,128),nn.ReLU()),nn.Sequential(nn.Linear(300,256),nn.ReLU(),nn.Linear(256,128),nn.ReLU()),nn.Sequential(nn.Linear(300,256),nn.ReLU(),nn.Linear(256,128),nn.ReLU())
        self.dd = nn.Linear(512,128)
        self.c1,self.c2,self.c3 = self.classifier(256,1),self.classifier(256,1),self.classifier(256,1)
        self.name = 'MTL_26 _se version '
    def classifier(self,In,out):
        return nn.Sequential(
                             nn.Linear(In,64),nn.ReLU(),
                             nn.Linear(64,16),nn.ReLU(),
                            nn.Linear(16,out))
    def forward(self,x1,x2=None,x3=None,num=-1,num1=-1,num2=-1,num3=-1):
        num_to_c = {1:self.c1,2:self.c2,3:self.c3}
        num_to_c2 = {7:self.c1,8:self.c2}
        num_to_f = {7:self.f1,8:self.f2}
        if x2!=None and x3 == None:
            assert num1!= -1 and num2 != -1

            d1,d2 = x1[:,0:300],x2[:,0:300]
            f1,f2 = x1[:,300:],x2[:,300:]

            x1,x2 = nn.ReLU()(self.d(d1)),nn.ReLU()(self.d(d2))
            x1 = nn.ReLU()(self.dd(x1))
            x2 = nn.ReLU()(self.dd(x2))
            f1,f2 = num_to_f[num1](f1),num_to_f[num2](f2)
            c1,c2 = num_to_c2[num1],num_to_c2[num2]
            x1,x2 = torch.cat((x1,f1),1),torch.cat((x2,f2),1)
            res1,res2 = c1(x1),c2(x2)
            if num1 not in [4,5,6]:
                res1 = nn.Sigmoid()(res1)
            if num2 not in [4,5,6]:
                res2 = nn.Sigmoid()(res2)
            return res1,res2,0
        elif x3!= None:
            d1,d2,d3 = x1[:,0:300],x2[:,0:300], x3[:,0:300]
            f1,f2,f3 = x1[:,300:],x2[:,300:],x3[:,300:]
            x1,x2,x3 = nn.ReLU()(self.d(d1)),nn.ReLU()(self.d(d2)),nn.ReLU()(self.d(d3))
            x1 = nn.ReLU()(self.dd(x1))
            x2 = nn.ReLU()(self.dd(x2))
            x3 = nn.ReLU()(self.dd(x3))
            f1,f2,f3 = num_to_f[num1](f1),num_to_f[num2](f2),num_to_f[num3](f3)
            c1,c2,c3 = num_to_c2[num1],num_to_c2[num2],num_to_c2[num3]
            x1,x2,x3 = torch.cat((x1,f1),1),torch.cat((x2,f2),1),torch.cat((x3,f3),1)
            return nn.Sigmoid()(c1(x1)),nn.Sigmoid()(c2(x2)),0,0
        else :
            assert num != -1
            assert num in [7,8]
            d,f = x1[:,0:300],x1[:,300:]
            d = nn.ReLU()(self.d(d))
            d = nn.ReLU()(self.dd(d))
            f = num_to_f[num](f)
            c = num_to_c2[num]
            x1 = torch.cat((d,f),1)

            if num not in [4,5,6]:
                return nn.Sigmoid()(c(x1))

class MTL_36(nn.Module):
    '''
    Used for training 8 datasets together
    '''
    def __init__(self,single=False):
        super(MTL_36,self).__init__()
        self.d = nn.Linear(300,512)
#         self.d2 = nn.Linear(300,512)
        self.f1,self.f2,self.f3 = nn.Sequential(nn.Linear(800,256),nn.ReLU(),nn.Linear(256,128),nn.ReLU()),nn.Sequential(nn.Linear(800,256),nn.ReLU(),nn.Linear(256,128),nn.ReLU()),nn.Sequential(nn.Linear(800,256),nn.ReLU(),nn.Linear(256,128),nn.ReLU())
    ###### change
#         self.f4,self.f5,self.f6 = nn.Sequential(nn.Linear(300,256),nn.ReLU(),nn.Linear(256,128),nn.ReLU()),nn.Sequential(nn.Linear(300,256),nn.ReLU(),nn.BatchNorm1d(256),nn.Linear(256,128),nn.ReLU(),nn.BatchNorm1d(128)),nn.Sequential(nn.Linear(300,256),nn.ReLU(),nn.BatchNorm1d(256),nn.Linear(256,128),nn.ReLU(),nn.BatchNorm1d(128))
        self.f4,self.f5,self.f6 = nn.Sequential(nn.Linear(300,256),nn.ReLU(),nn.Linear(256,128),nn.ReLU()),nn.Sequential(nn.Linear(300,256),nn.ReLU(),nn.Linear(256,128),nn.ReLU()),nn.Sequential(nn.Linear(300,256),nn.ReLU(),nn.Linear(256,128),nn.ReLU())
        self.f7,self.f8 = nn.Sequential(nn.Linear(300,256),nn.ReLU(),nn.Linear(256,128),nn.ReLU()),nn.Sequential(nn.Linear(300,256),nn.ReLU(),nn.Linear(256,128),nn.ReLU())
        self.dd = nn.Linear(512,128)
        self.c1,self.c2,self.c3 = self.classifier(256,1),self.classifier(256,1),self.classifier(256,1)
        self.c4,self.c5,self.c6 = self.classifier(256,1),self.classifier(256,1),self.classifier(256,1)
        self.c7,self.c8 = self.classifier(256,1),self.classifier(256,1)
        self.name = 'MTL_36 - train all together '
    def classifier(self,In,out):
        return nn.Sequential(
                             nn.Linear(In,64),nn.ReLU(),
                             nn.Linear(64,16),nn.ReLU(),
                            nn.Linear(16,out))
    def forward(self,x1,x2=None,x3=None,x4=None,x5=None,x6=None,x7=None,x8=None,num=-1):
        num_to_c = {1:self.c1,2:self.c2,3:self.c3}
        num_to_c2 = {1:self.c1,2:self.c2,3:self.c3,4:self.c4,5:self.c5,6:self.c6,7:self.c7,8:self.c8}
        num_to_f = {1:self.f1,2:self.f2,3:self.f3,4:self.f4,5:self.f5,6:self.f6,7:self.f7,8:self.f8}
        if type(x2) != type(None):
            d1,d2,d3,d4 = x1[:,0:300],x2[:,0:300],x3[:,0:300],x4[:,0:300]
            d5,d6,d7,d8 = x5[:,0:300],x6[:,0:300],x7[:,0:300],x8[:,0:300]
            f1,f2,f3,f4 = x1[:,300:],x2[:,300:],x3[:,300:],x4[:,300:]
            f5,f6,f7,f8 = x5[:,300:],x6[:,300:],x7[:,300:],x8[:,300:]

            x1,x2,x3,x4 = nn.ReLU()(self.d(d1)),nn.ReLU()(self.d(d2)),nn.ReLU()(self.d(d3)),nn.ReLU()(self.d(d4))
            x5,x6,x7,x8 = nn.ReLU()(self.d(d5)),nn.ReLU()(self.d(d6)),nn.ReLU()(self.d(d7)),nn.ReLU()(self.d(d8))
            x1,x2,x3,x4 = nn.ReLU()(self.dd(x1)),nn.ReLU()(self.dd(x2)),nn.ReLU()(self.dd(x3)),nn.ReLU()(self.dd(x4))
            x5,x6,x7,x8 = nn.ReLU()(self.dd(x5)),nn.ReLU()(self.dd(x6)),nn.ReLU()(self.dd(x7)),nn.ReLU()(self.dd(x8))
            f1,f2,f3 = self.f1(f1),self.f2(f2),self.f3(f3)
            f4 = self.f4(f4)
            f5,f6,f7,f8 = self.f5(f5),self.f6(f6),self.f7(f7),self.f8(f8)
            x1,x2,x3,x4 = torch.cat((x1,f1),1),torch.cat((x2,f2),1),torch.cat((x3,f3),1),torch.cat((x4,f4),1)
            x5,x6,x7,x8 = torch.cat((x5,f5),1),torch.cat((x6,f6),1),torch.cat((x7,f7),1),torch.cat((x8,f8),1)
            return nn.Sigmoid()(self.c1(x1)),nn.Sigmoid()(self.c2(x2)),nn.Sigmoid()(self.c3(x3)),self.c4(x4),self.c5(x5),self.c6(x6),nn.Sigmoid()(self.c7(x7)),nn.Sigmoid()(self.c8(x8))

        else :
            assert num != -1
            d,f = x1[:,0:300],x1[:,300:]
            d = nn.ReLU()(self.d(d))
            d = nn.ReLU()(self.dd(d))
            f = num_to_f[num](f)
            c = num_to_c2[num]
            x1 = torch.cat((d,f),1)
            if num not in [4,5,6]:
                return nn.Sigmoid()(c(x1))
            else :
                return c(x1)
