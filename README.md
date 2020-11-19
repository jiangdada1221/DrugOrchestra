# DrugOrchestra
DrugOrchestra is a multi-task learning neural model used for jointly training tasks of drug target prediction, drug response prediction, drug side effect prediction. It is proven to be better compared to single-task learning (training only one task) under the same training conditions

## Packages and environment
torch==1.4.0 <br />
torchvision==0.5.0 <br />
pandas==1.1.3 <br />
numpy==1.18.5 <br />
rdkit==2017.09.1 <br />
tqdm==4.48.0 <br />
scipy=1.4.1 <br />
sklearn==0.23.2 <br />
python==3.7.6<br />
cuda==9.2

## Data
The data after thresholding and feature extraction are available in <br /> 
https://drive.google.com/file/d/1JvGDiNMAqWJb4Ya0c7ahV9fM1MCjtkxA/view?usp=sharing .
 <br />

The data used for training and testing are available in <br />
https://drive.google.com/file/d/1tzsZwk0exESwq1hoLii0SI5MWB-k5BxC/view?usp=sharingput . <br />

## How to run the code
Please refer to the https://github.com/jiangdada1221/DrugOrchestra/tree/master/MTL to see examples. <br />

## To make predictions
Run the script in Predict folder by: <br />
<br />
python make_predictions.py --SMILES arg1 --path_to_gnn_model arg2 --path_to_MTL arg3 <br />
<br />
arg1 is the input SMILES string you want to use to make predictions <br />
arg2 is the path to the pretrained GNN model for drug embedding extraction <br />
args is the path to the pretrained MTL model <br />
By running this script, you will get output of prediced targets,response,side effects for the given drug <br />


## Correspondence
If you have any question, feel free to contact jiangdada12344321@gmail.com
