# getGPLVM.py is a script, that implements the GP-LVM optimization.
# This script calls the function defined in the bGPLVMOptimizer
# python script. The three step procedure is described in the paper.
#
# This code is available under a GPL v3.0 license and comes without
# any explicit or implicit warranty.
#
# (C) Wilfried Wöber 2020 <wilfried.woeber@technikum-wien.at>
import numpy as np  #Numpy :-)
import sys
sys.path.append("../Python/")                  #Get own GPC stuff (GPC_nfCV.py)
from bGPLVMOptimizer import step1_PCA
from bGPLVMOptimizer import step2_bGPLVM_IDP
from bGPLVMOptimizer import step3_bGPLVM_latentDim
import GPy
from sklearn import preprocessing       #Data preprocessing
#--------------------#
#--- Define paths ---#
#--------------------#
path_data   = "../Data/design.csv"      #Flattened image vectors
data_raw    = np.loadtxt(path_data,delimiter=",")           #Load data from file
#---------------------#
#--- Do processing ---#
#---------------------#
PCADim  =   step1_PCA(data_raw,explanationTH=0.75)  #Get initial latent Dimension from PCA
IDP     =   step2_bGPLVM_IDP(  design_raw = data_raw,
                            PCAdim = PCADim, 
                            IDPRange = [10,20,50,75,100, 125, 150,200], 
                            IDP_LL_TH = 0.95,
                            doPlot = True)
lDim    =   step3_bGPLVM_latentDim(data_raw, IDP, [2,5,10,20,30,40,50,75,100,125,150], doPlot=True)
#-----------------------------#
#--- Train optimized model ---#
#-----------------------------#
iterations=10000                                            #Number of maximum iteration for modelling
print("Train optimized model")
print("Standartize data...")
scaler = preprocessing.StandardScaler().fit(data_raw)     #Create and train scaler
design = scaler.transform(data_raw)                       #Scale data
print("Start modelling")
while(True):
    model_LVM = GPy.models.BayesianGPLVM(Y               = design,      #Dataset - the original dimension
                                         input_dim       = lDim,        #Latent dimension
                                         num_inducing    = IDP)         #Number of induction points
    model_return = model_LVM.optimize(messages=False, max_iters   = iterations)  #Do modelling 
    #--- We do this till convergency ---#
    if(model_return.status!="Errorb'ABNORMAL_TERMINATION_IN_LNSRCH'"):
        break
#--- We now have finalized model ---#
pre_str = "optModel_"
projection_test, projection_var = model_LVM.infer_newX(design) #Do prediction for dataset
projection = projection_test.mean                                   #Get mean value of dataset
#---------------------#
#--- Store results ---#
#---------------------#
np.savetxt(pre_str+"bgp_features_train.csv", projection, delimiter=',')
np.savetxt(pre_str+"bgp_ARDValues.csv", model_LVM['rbf.lengthscale'], delimiter=',')
np.savetxt(pre_str+"bgp_loglik.csv", model_LVM.log_likelihood()[0,:], delimiter=',')
np.save(pre_str+'bgp_model.npy', model_LVM.param_array)
