import numpy as np
import os
import torch
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
import numpy as np

class dataset_single(Dataset):
    '''
    Dataset for the single dataset. 
    Input is the path to the dataset
    '''
    def __init__(self,path,batch_size):
        self.data = np.lib.format.open_memmap(path)
#         self.num = num
        self.batch_size= batch_size
    def __len__(self):  
        length = self.data.shape[0]
        return (length // self.batch_size) * self.batch_size
#         return length
#         return min(self.data.shape[0]-1,4992)
    def __getitem__(self,idx):
        x = torch.from_numpy(np.round(self.data[idx][:-1],5))
        
        y = self.data[idx][-1]
        if y == 1 or y == -1:
            y = torch.FloatTensor([0]) if y == -1 else torch.FloatTensor([1])  
        else :
            y = torch.FloatTensor([y])
        Id = torch.LongTensor([idx])
        return (x,y,Id)
    
    def get_multiple(self,ids):
        #get multiple sample (used for randomly selecting samples)
        xs,ys,zs = [],[],[]
        if type(ids) != int:        
            for index in range(len(ids)):
                x,y,z = self.__getitem__(index)
                x,y,z = x.view((1,-1)),y.view((1,-1)),z.view((1,-1))
                xs.append(x)
                ys.append(y)
                zs.append(z)
            return torch.cat(xs,0),torch.cat(ys,0),torch.cat(zs,0)
        else :
            needed = 0
            while len(xs) != ids:
                index = np.random.randint(self.__len__())
                x,y,z = self.__getitem__(index)
            
                if y.item() == needed:
                    x,y,z = x.view((1,-1)),y.view((1,-1)),z.view((1,-1))
                    xs.append(x)
                    ys.append(y)
                    zs.append(z)
                    needed = 1 if needed == 0 else 1
            return torch.cat(xs,0),torch.cat(ys,0),torch.cat(zs,0)


class dataset_whole(Dataset):
    '''
    Dataset for one task (with datasets combined)
    '''
    def __init__(self,paths,task,batch_size):
        '''
        paths: a list containing paths to all the datasets
        task : a integer indicating which task you want to use. 1-dti,2-d_response,3:adr
        '''
        paths = paths[(task-1)*3:task*3]
        self.dsets = [dataset_single(i,batch_size) for i in paths]
        if task == 3:
            assert len(paths) == 2 #only 2 datasets in drug_se task
    def __len__(self):
        return sum([len(i) for i in self.dsets])
    def __getitem__(self,idx):
        len1,len2 = len(self.dsets[0]),len(self.dsets[1])
        if idx >= len1:
            if idx >= len1+len2:
                return self.dsets[2].__getitem__(idx-len1-len2)
            else :
                return self.dsets[1].__getitem__(idx-len1)
        else :
            return self.dsets[0].__getitem__(idx)
        
    

# class dataset_all(Dataset):
#     '''
#     The dataset for the whole training/test combined
#     The input should be Drug_target/Drug_response/Drug_se
#     Data is stored as .npy file
    
#     x.shape= [batch_size,1100/600]
#     y.shape = [batch_size,1]
#     '''
#     def __init__(self,dset_name,test = False):
#         self.dset_name = dset_name
#         self.test = test
#         file = 'train.npy'
#         if test == True:
#             file = 'test.npy'
#         assert dset_name in ['Drug_target','Drug_response','Drug_se']
#         if dset_name == 'Drug_target':
#             ds = ['repurposing_hub/'+file,'drugbank/'+file,'stitch/'+file]
            
#         elif dset_name == 'Drug_response':
#             ds = ['pdx/'+file,'gdsc/'+file,'ccle/'+file]
#         else :
#             assert dset_name == 'Drug_se'
#             ds = ['sider/'+file,'offside/'+file]
            
#         if dset_name == 'Drug_se':
#             d1,d2 = np.lib.format.open_memmap('../'+dset_name+'/'+ds[0]),np.lib.format.open_memmap('../'+dset_name+'/'+ds[1])
#             self.dset = [d1,d2]
#         else :
#             d1,d2,d3 = np.lib.format.open_memmap('../'+dset_name+'/'+ds[0]),np.lib.format.open_memmap('../'+dset_name+'/'+ds[1]),np.lib.format.open_memmap('../'+dset_name+'/'+ds[2])
#             self.dset = [d1,d2,d3]
    
#     def __len__(self):
#         count = 0
#         for d in self.dset:
#             count += d.shape[0]
#             #count
#         return count
# #         return 10000
    
#     def __getitem__(self,idx):
        
#         if self.dset_name == 'Drug_se':
#             len1 = self.dset[0].shape[0]
#             if idx >= len1:
#                 x = torch.from_numpy(self.dset[1][idx-len1][:-1])
#                 y = torch.LongTensor([0]) if self.dset[1][idx-len1][-1] == -1 else torch.LongTensor([1])
#                 z = 7
#             else:
#                 x = torch.from_numpy(self.dset[0][idx][:-1])
#                 y = torch.LongTensor([0]) if self.dset[0][idx][-1] == -1 else torch.LongTensor([1])
#                 z = 6
#         else :
#             len1 = self.dset[0].shape[0]
#             len2 = self.dset[1].shape[0]
#             if  idx < len1:
#                 x = torch.from_numpy(self.dset[0][idx][:-1])
#                 y = self.dset[0][idx][-1]
#                 z = 0 if self.dset_name =='Drug_target' else 3
#             elif idx>=len1+len2:
#                 x = torch.from_numpy(self.dset[2][idx-len1-len2][:-1])
#                 y = self.dset[2][idx-len1-len2][-1]
#                 z = 2 if self.dset_name == 'Drug_target' else 5
#             else :
#                 x = torch.from_numpy(self.dset[1][idx-len1][:-1])
#                 y = self.dset[1][idx-len1][-1]
#                 z = 1 if self.dset_name == 'Drug_target' else 4
#             if self.dset_name == 'Drug_target':
#                 y = torch.LongTensor([0]) if y == -1 else torch.LongTensor([1])
#             else :
#                 y = torch.FloatTensor([y])
#         z = torch.LongTensor([z])
#         return (x,y,z)
#     def get_multiple(self,ids):
#         xs,ys,zs = [],[],[]
#         if type(ids) != int:        
#             for index in range(len(ids)):
#                 x,y,z = self.__getitem__(index)
#                 x,y,z = x.view((1,-1)),y.view((1,-1)),z.view((1,-1))
#                 xs.append(x)
#                 ys.append(y)
#                 zs.append(z)
#             return torch.cat(xs,0),torch.cat(ys,0),torch.cat(zs,0)
#         else :
#             needed = 0
#             while len(xs) != ids:
#                 index = np.random.randint(self.__len__())
#                 x,y,z = self.__getitem__(index)
            
#                 if y.item() == needed:
#                     x,y,z = x.view((1,-1)),y.view((1,-1)),z.view((1,-1))
#                     xs.append(x)
#                     ys.append(y)
#                     zs.append(z)
#                     needed = 1 if needed == 0 else 1
#             return torch.cat(xs,0),torch.cat(ys,0),torch.cat(zs,0)

    
