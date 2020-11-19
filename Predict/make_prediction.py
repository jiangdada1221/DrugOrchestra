import sys
sys.path.append('../MTL/')
import os
import torch
import pickle
import collections
import math
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
# from rdkit import DataStructs
# from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from GNN_loader import mol_to_graph_data_obj_simple
from GNN_model import GNN_graphpred
from GNN_model import GNN
import torch
from model import *

## This file is used for making predictions
## The input is a SMILES string and will output the predicted targets,responses,side-effects by the MTL model
## Note that, to use the pretrained model for drug feature extracting, you may need additional packages. Please refer to https://github.com/snap-stanford/pretrain-gnns/

def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--SMILES',type=str,help='the input SMILES string')
    parser.add_argument('--path_to_gnn_model',type=str,help='the path to the pretrained GNN model; the model can be download from https://github.com/snap-stanford/pretrain-gnns/blob/master/chem/model_gin/supervised_contextpred.pth ')    
    parser.add_argument('--path_to_MTL',type=str,help='path to the pretrained MTL model;A sample model can be download from https://drive.google.com/file/d/1SbKfPve-befnRR77VRpDE003w0_Jg04A/view?usp=sharing')
    args = parser.parse_args()

    pre_drug = args.path_to_gnn_model #location of the pretrained model for drug embedding
    SMILES = args.SMILES
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = GNN(5,300)
    model.load_state_dict(torch.load(pre_drug))  #load the pretrained model for drug embedding
    model = model.to(device)
    model.eval()
    #build graph from SMILES string
    rdkit_mol = AllChem.MolFromSmiles(SMILES)
    data = mol_to_graph_data_obj_simple(rdkit_mol)  #to graph 
    drug_vector = torch.mean(model(data.to(device)).cpu().detach().numpy(),axis=0) #obtain the drug embedding

    pre_MTL = args.path_to_MTL #location of the pretrained MTL model
    MTL_model = MTL_36()
    MTL_model.load_state_dict(torch.load(pre_MTL))
    MTL_model = MTL_model.to(device)
    MTL_model.eval()

    #to make predictions for targets
    files = ['../Drug_target/stitch/gene_vectors.csv','../Drug_target/drugbank/gene_vectors.csv','../Drug_target/repurposing_hub/gene_vectors.csv']
    # drug_vector = drug_vector.to(device)
    dtis = [[],[],[]]  # store the predicted dti
    for i,file in enumerate(files) :
        gene_vecs = pd.read_csv(file)
        for gene in gene_vecs.columns.values:
            gene_vec = gene_vecs[gene].values
            input_vec = torch.reshape(torch.from_numpy(np.concatenate((drug_vector,gene_vec))),(1,-1))
            input_vec = input_vec.to(device)
            score = MTL_model(x1=input_vec,num = i+1).item()                        
            if score >= 0.5:
                dtis[i].append(gene)
    with open('predicted_dti.txt','w') as f:
        names = ['STITCH','Repurposing Hub','Drugbank']
        for i,dti in enumerate(dtis):
            f.write('Predicted targets based on {}\n'.format(names[i]))        
            for t in dti:
                f.write(str(t)+'\n')
            f.write('\n')    

    #to make predictions for response
    files = ['../Drug_response/pdx/ccl_feature.csv','../Drug_response/gdsc/ccl_feature.csv','../Drug_response/ccle/ccl_feature.csv']
    means = [130.28,0.843,12.878]  #the response data used for training is the z-score value of original response
    stds = [175.239,0.1947,2.574]  #need to convert them to original data
    # drug_vector = drug_vector.to(device)
    responses = [[],[],[]]  # store the predicted response
    ccls = [[],[],[]]
    for i,file in enumerate(files) :
        ccl_vecs = pd.read_csv(file)
        for ccl in ccl_vecs.columns.values[1:]:
            ccl_vec = ccl_vecs[ccl].values
            input_vec = torch.reshape(torch.from_numpy(np.concatenate((drug_vector,ccl_vec))),(1,-1))
            input_vec = input_vec.to(device)
            score = MTL_model(x1=input_vec,num = i+4).item()                        
            responses[i].append(score*stds[i]+means[i])
            ccls[i].append(ccl)
    with open('predicted_response.txt','w') as f:
        names = ['PDX','GDSC','CCLE']    
        for i,res in enumerate(responses):
            f.write('Predicted responses based on {}\n'.format(names[i]))        
            f.write('ccl(PDX),response\n')
            for k in range(len(ccls[i])):
                f.write(str(ccls[i][k])+','+str(responses[i][k])+'\n')
            f.write('\n')    

    #to make predictions for response
    files = ['../Drug_se/disease_embedding.csv']*2
    # drug_vector = drug_vector.to(device)
    ses = [[],[]]  # store the predicted side effects
    for i,file in enumerate(files) :
        se_vecs = pd.read_csv(file)
        for se in se_vecs.columns.values[1:]:
            se_vec = se_vecs[se].values
            input_vec = torch.reshape(torch.from_numpy(np.concatenate((drug_vector,se_vec))),(1,-1))
            input_vec = input_vec.to(device)
            score = MTL_model(x1=input_vec,num = i+7).item()                        
            if score >= 0.5:
                ses[i].append(se)
    with open('predicted_se.txt','w') as f:
        names = ['SIDER','OFFSIDES']    
        for i,se in enumerate(ses):
            f.write('Predicted side effect based on {}\n'.format(names[i]))        
            for s in se:
                f.write(str(s)+'\n')
            f.write('\n')    

if __name__ == '__main__':
    main()
