File description:

dataset.py : containing dataset classes
model.py :
  containing models. MTL_26,MTL_36,MTL_33,MTL_34,MTL_35 have the same
  structure but with different input. MTL_26 is used for investigating
  transferability across tasks(datasets),MTL_33-35 are for inner tasks(datasets)
  and MTL_36 is used for training the 8 datasets together

filter.py:
  used for computing the performances after filtering based on different
  Tanimoto score
process.py : to performance random split for the data
train_all.py : train 8 datasets simultaneously
train_d2d.py : investigating transferability of dataset to dataset
train_t2t.py : investigating transferability of task to task


How to use (belows scripts are just examples, please see the py files for documentation):

First, download the data from https://drive.google.com/file/d/1tzsZwk0exESwq1hoLii0SI5MWB-k5BxC/view?usp=sharingput
Put the data inside the right directory.

Before running the code for training, split the data first:

  python process.py --number ? --use_old ? # '?' need to be replaced by specific argument

To perform MTL on 8 datasets:

  python train_all.py --model 36 --dynamic ? --info ?

To investigate transferability :

  python train_d2d.py --model 26 --num1 ? --num2 ? --weight1 1 --weight2 1 --dynamic ?
  #across datasets, num1 represents source dataset,num2 represents target dataset
  
  

  python train_d2d.py --model 33 --num1 ? --num2 ? --weight1 1 --weight2 1 --dynamic ?
  #inner datasets within same task. e.g. num1-1(STITCH),num2-7(SIDER)

  python train_t2t.py --num1 ? --num2 ? --model 26 --dynamic ?
  #transferability of task to task

To explore the applicability domain :
  Download the data for this part from https://drive.google.com/file/d/1JvGDiNMAqWJb4Ya0c7ahV9fM1MCjtkxA/view?usp=sharing
  Put the files in the right directory and then run :

  python filter.py --threshold ?

If you want to get the drug embedding by your own, you can refer to http://snap.stanford.edu/gnn-pretrain/ 
