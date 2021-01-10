# FeatureVariance.py estiamtes the heatmaps for GP-LVM. It
# is based on the estimated models from getGPLVM.py.
#
# This code is available under a GPL v3.0 license and comes without
# any explicit or implicit warranty.
#
# (C) Wilfried Wöber 2020 <wilfried.woeber@technikum-wien.at>
import sys                  #System stuff
sys.path.append('../Python/')  #Add path to project library
from bGPLVM import bGPLVM   #GPy wrapper
import numpy as np          #You should know that
import matplotlib.pyplot as plt #We aim to plot something...
image_dim = (224,224)
import os                   #For bash stuff
#--------------------------------------#
#--- Load IDP and optimal dimension ---#
#--------------------------------------#
latentDim        = np.loadtxt("./optModel_bgp_lDim.csv", dtype=int)
nrInd            = np.loadtxt("./optModel_bgp_nrInd.csv",dtype=int)
# --> note, we used 90% rule
print("Use %d inducing points and %d latent dimensions" % (nrInd,latentDim))
#-------------------------#
#--- Init bGPLVM model ---#
#-------------------------#
data_Model	= "./optModel_bgp_model.npy"
data_latent	= "./optModel_bgp_features_train.csv"
dataFolder	= "../Data/design.csv"
#--- Create bGPLVM model ---#
model =bGPLVM(  dataFolder,         #Path to training data
                data_latent,        #Extracted features
              	data_Model,         #Path to model data
                "",                 #Sampels from faulty classes
                "",                 #Path to excluded featues
                nrInd,              #Number of inducing pts
                latentDim,          #Number of latent dimensions
                (image_dim[1],image_dim[0]))       #Reshaping image
#--------------------------------------#
#--- Estimate variance per features ---#
#--------------------------------------#
os.system("mkdir Heatmaps")
model.plotVarHeatmaps(prefix="./Heatmaps/")
