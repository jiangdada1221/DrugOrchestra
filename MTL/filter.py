import numpy as np
import pandas as pd
import argparse
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem.Fingerprints import FingerprintMols
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

def main():

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--threshold',type=float,default=0.0,help='threshold of Tanimoto score,0.6-0.9')
    args = parser.parse_args()
    print('The threshold you are using is :',args.threshold)
    paths_train = ['../Drug_target/stitch/train.npy','../Drug_target/repurposing_hub/train.npy','../Drug_target/drugbank/train.npy','../Drug_response/pdx/train.npy','../Drug_response/gdsc/train.npy','../Drug_response/ccle/train.npy','../Drug_se/sider/train.npy','../Drug_se/offside/train.npy']
    paths_test = ['../Drug_target/stitch/test.npy','../Drug_target/repurposing_hub/test.npy','../Drug_target/drugbank/test.npy','../Drug_response/pdx/test.npy','../Drug_response/gdsc/test.npy','../Drug_response/ccle/test.npy','../Drug_se/sider/test.npy','../Drug_se/offside/test.npy']

    paths_original = ['../Drug_target/stitch/','../Drug_target/repurposing_hub/','../Drug_target/drugbank/','../Drug_response/pdx/','../Drug_response/gdsc/','../Drug_response/ccle/','../Drug_se/sider/','../Drug_se/offside/']
    paths_smile = [i+'drug_smile.csv' for i in paths_original]
    paths_vector = [i+'drug_embedding.csv' for i in paths_original]
    model_num = 36
    def filter_test(d_train,d_test,d2s,threshold):
        #filter drugs in test based on threshold
        #d_train / d_test : a set containg the drugs in train / test
        #d2s : a dict mapping drug to SMILES string
        #returns a set after filtering
        d_train,d_test = list(d_train),list(d_test)
        mol_train = [Chem.MolFromSmiles(d2s[x]) for x in d_train]
        mol_test = [Chem.MolFromSmiles(d2s[x]) for x in d_test]
        fps_train = [FingerprintMols.FingerprintMol(x) for x in mol_train]
        fps_test= [FingerprintMols.FingerprintMol(x) for x in mol_test]
        to_keep = []
        for i in range(len(d_test)):
            remove = False
            for fp in fps_train:
                score = DataStructs.FingerprintSimilarity(fp,fps_test[i])
                if score >= threshold:
    #                 to_delete.append(d_test[i])
                    remove = True
            if remove == False:
                to_keep.append(d_test[i])
        return set(to_keep)

    def performance4(model,loader,num,batch_size=32):
        #compute performance
        model.eval()
        rep=1
        y_pres = np.zeros((len(loader)*batch_size,rep))
        y_trues = np.zeros((len(loader)*batch_size,1))
    #     dset1,dset2 = loader_tests[1],loader_tests[2]
    #     len1,len2 = len(dset1),len(dset2)
        used = []
        with torch.no_grad():
            for index in range(rep):
                for i,(x,y,z) in tqdm(enumerate(loader)):
                    x,y = x.to(device),y.to(device)
                    used = used + list(z.detach().cpu().numpy()[:,0])
                    y_ = model(x1=x.float(),num=num)
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

    #Begin the filtering:
    k=0    #do the filtering for stitch
    data_train= np.load(paths_train[k])
        #vector to drug
    m_file,s_file = pd.read_csv(paths_vector[k]),pd.read_csv(paths_smile[k])
    v2d,d2s = dict(),dict()  #mapping of embedding vector to drug, mapping of drug to SMILES string
    for i in range(m_file.shape[0]):
        row = m_file.loc[i].values
        v2d[str(np.round(row[1:11].astype(np.float),6))] = row[0] #6 digits
    col1,col2 = s_file['cid'],s_file['SMILE']
    for i in range(len(col1)):
        d2s[col1[i]] = col2[i]
    v_in_train,d_in_train = set(),set()
    for i in range(data_train.shape[0]):
        v_in_train.add(str(np.round(data_train[i,0:10],6)))
    for v in v_in_train:
        d_in_train.add(v2d[v])
    del data_train

    ## to get drugs in test:
    data_test=  np.load(paths_test[k])
    v_in_test,d_in_test = set(),set()
    for i in range(data_test.shape[0]):
        v_in_test.add(str(np.round(data_test[i,0:10],6)))
    for v in v_in_test:
        d_in_test.add(v2d[v])

    #create filtered test set
    d_needed = filter_test(d_in_train,d_in_test,d2s,args.threshold) #return a set containing drugs needed in test
    indexes = []
    for i in range(data_test.shape[0]):
        d = v2d[str(np.round(data_test[i,0:10],6))]
        if d in d_needed:
            indexes.append(i)
    new_test = data_test[indexes]
    np.save(paths_test[k],new_test)

    #do the filtering for the rest datasets
    for k in range(1,8):
        data_train= np.load(paths_train[k])
        #vector to drug
        m_file,s_file = pd.read_csv(paths_vector[k]),pd.read_csv(paths_smile[k])
        v2d,d2s = dict(),dict()
        for drug in m_file.columns.values:
            v2d[str(np.round(m_file[drug].values[0:10],6))] = drug #6 digits
        col1,col2 = s_file[s_file.columns[0]],s_file[s_file.columns[1]]
        if k == 6:
            col2 = s_file[s_file.columns[2]]
        for i in range(len(col1)):
            d2s[col1[i]] = col2[i]
        v_in_train,d_in_train = set(),set()
        for i in range(data_train.shape[0]):
            v_in_train.add(str(np.round(data_train[i,0:10],6)))
        for v in v_in_train:
            d_in_train.add(v2d[v])
        del data_train
        ## to get drugs in test:
        data_test=  np.load(paths_test[k])
        v_in_test,d_in_test = set(),set()
        for i in range(data_test.shape[0]):
            v_in_test.add(str(np.round(data_test[i,0:10],6)))
        for v in v_in_test:
            d_in_test.add(v2d[v])

        d_needed = filter_test(d_in_train,d_in_test,d2s,args.threshold) #return a set containing drugs needed in test
        indexes = []
        for i in range(data_test.shape[0]):
            d = v2d[str(np.round(data_test[i,0:10],6))]
            if d in d_needed:
                indexes.append(i)
        new_test = data_test[indexes]
        np.save(paths_test[k],new_test)
        del data_test

    ## begin computing the performances:
    paths = ['../Drug_target/stitch/','../Drug_target/drugbank/','../Drug_target/repurposing_hub/',
                '../Drug_response/pdx/',
                '../Drug_response/gdsc/','../Drug_response/ccle/',
                '../Drug_se/sider/','../Drug_se/offside/']
    paths = [i+'test.npy' for i in paths]
    paths = [dataset_single(i,32) for i in paths]#test
    loader_singles = [DataLoader(i,shuffle=True,batch_size=32) for i in paths] #test
    device = 'cuda:0'

    for i in range(8):
        model = MTL_36()
        model.load_state_dict(torch.load('models/MTL_{0}.pth'.format(i+1)))
        model = model.to(device)
        model.eval()
        a,b = performance4(model,loader_singles[i],i+1)
        with open('results/MTL_AD_{}.txt'.format(i+1),'a') as f:
            f.write('\n')
            f.write(str(args.threshold)+'\n')
            f.write(str(a)+','+str(b)+'\n')
        model.load_state_dict(torch.load('models/STL_{0}.pth'.format(i+1)))
        model = model.to(device)
        a,b = performance4(model,loader_singles[i],i+1)
        with open('results/STL_AD_{}.txt'.format(i+1),'a') as f:
            f.write('\n')
            f.write(str(args.threshold)+'\n')
            f.write(str(a)+','+str(b)+'\n')

if __name__ == '__main__':
    main()
