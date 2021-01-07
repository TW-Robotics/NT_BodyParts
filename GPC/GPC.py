# The GPC.py script runs the GPC for Bayes GP-LVM and Procrustes 
# features. GPy is used. This script is called using:
#
# python GPC.py <data-file.csv> <feature-ignore-list.csv>(optional)
#
# This script generates the files for our statistical analysis.
#
# This code is available under a GPL v3.0 license and comes without
# any explicit or implicit warranty.
#
# (C) Wilfried Wöber 2020 <wilfried.woeber@technikum-wien.at>

#-------------------------#
#--- Run main programm ---#
#-------------------------#
import sys                              #Use system stuff
sys.path.append("../Python/")           #Get own GPC stuff (GPC_nfCV.py)
from GPC_nfCV import GPC_nfCV           #Get the class
import numpy as np                      #Numpy ;)
import pandas as pd                     #We need that for novel data loading
from sklearn.preprocessing import StandardScaler    #Scale input data
#-------------------------------------#
#--- Check arguments from terminal ---#
#-------------------------------------#
if((len(sys.argv) < 2) or (len(sys.argv)>3) ):
    print("Please call GPC script with: the path to the data file and the path to the feature selection")
    sys.exit()
path_data = sys.argv[1]     #Get Path to data
if(len(sys.argv) == 3):
    Selection_path      = sys.argv[2] 
    Selection           = np.genfromtxt(Selection_path, delimiter=',',dtype=int)
else:
    Selection_path      = "" 
k_fold=10                   #k-fold cross validation
print("Got path to data: %s" % path_data)
print("Path to feature selection: %s" % Selection_path)
print("Use %d for k-fold cross validation" % k_fold)
#--------------------------------------#
#--- Load files from data generator ---#
#--------------------------------------#
data_file = pd.read_csv(path_data, header=None).as_matrix()    #Read data including sample ID
# The format is for each row: sampleID (string), label (int), data vector (double)
samples_iteration   = np.genfromtxt("../Data/NoReplace_20205809_085859.csv", delimiter=' ',dtype=int)  #Load data from R script
print("Do %d iterations" % samples_iteration.shape[1])
#-------------------#
#--- Create data ---#
#-------------------#
sample_ID = data_file[:,0]                      #The sample ID
data_raw = data_file[:,1:data_file.shape[1]]    #The label followed by data (BGPLVM features)
#------------------#
#--- Do looping ---#
#------------------#
for ITERATION in range (0,samples_iteration.shape[1]):
    print("Iteration %d" % ITERATION)
    #--- Create data ---#
    index = samples_iteration[:,ITERATION]          #Get sampled random indices
    X_train = data_raw[index,1:data_raw.shape[1]]   #Get BGPLVM data
    if(Selection_path != ""):
        X_train = X_train[:,Selection]
    y_train = data_raw[index,0].astype(int)         #Get labels
    #-----------------------------#
    #--- Do data preprocessing ---#
    #-----------------------------#
    scaler_train = StandardScaler()
    scaler_train.fit(X_train)
    X_train = scaler_train.transform(X_train)
    #-------------------------#
    #--- Do classification ---#
    #-------------------------#
    n_fold_CV = GPC_nfCV(X_train,y_train, k_fold)
    GPC_return=n_fold_CV.run(str(ITERATION))
    data_iteration = np.concatenate(    (sample_ID[index].reshape((209,1)), #Sample name
                                        y_train.reshape((209,1)),           #True class
                                        GPC_return[1].astype(int),          #Predicted class
                                        GPC_return[2]),                     #Class probability
                                    axis=1)
    prefix = str(ITERATION)
    #--- Create csv file for later processing ---#
    #ttarg: true target;
    #ptarg: predicted target
    #probs0-5: class probabilities
    data_list = {   "ttarg":y_train.reshape((209,)), 
                    "ptarg":GPC_return[1].astype(int).reshape((209,)), 
                    "probs0":GPC_return[2][:,0].reshape((209,)),
                    "probs1":GPC_return[2][:,1].reshape((209,)),
                    "probs2":GPC_return[2][:,2].reshape((209,)),
                    "probs3":GPC_return[2][:,3].reshape((209,)),
                    "probs4":GPC_return[2][:,4].reshape((209,)),
                    "probs5":GPC_return[2][:,5].reshape((209,))}
    print("ttarg dim:", data_list['ttarg'].shape)
    print("ptarg dim:", data_list['ptarg'].shape)
    print("prob0 dim:", data_list['probs0'].shape)
    #df = pd.DataFrame(data=data_iteration)
    df = pd.DataFrame(data=data_list)
    df = df[["ttarg", "ptarg", "probs0", "probs1", "probs2", "probs3", "probs4", "probs5"]]
    df.to_csv(prefix+"_dataMatrix.csv", index=False)
