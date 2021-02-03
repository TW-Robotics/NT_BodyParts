# This scripts prepares the calculated classification results
# from GPC and CNN for pie and ranking visualization.
#
# In this script, we load the trainign matrix from the GPC
# and CNN, get the lengthscale values from the GPC and convert
# them. The convertion is done be invert the values, store the 
# inverted lengthscale values (=relevance) and get the mean values 
# over all k's in the k-fold cross validation.
# 
# We store two relevance values, namely the mean of all entries in the
# relevance matrix and the mean of overall col mean. 
#
# This code is available under a GPL v3.0 license and comes without
# any explicit or implicit warranty.
#
# (C) Wilfried Woeber 2020 <wilfried.woeber@technikum-wien.at>
import numpy as np 
import os
import pandas as pd
#------------------------#
#--- global functions ---#
#------------------------#
# Create a folder for a given dataset
def mkdir(classifier):
    os.system("mkdir data/data_"+classifier) #Creates a folder for given classifier
# Returns the name of the classifiers data folder
def getdir(classifier):
    return("data/data_"+classifier)
# Converts data to a pandas DF with names
def convertARD(M):
    myDF = pd.DataFrame(relevance_local)    #Convert to pandas DF with false col name
    #--- Create column names ---#
    colnames = ['ard_1']    #Default name
    for i in range(2,M.shape[1]+1):
        colnames.extend(['ard_'+str(i)])    #Extend name list
    myDF.columns = colnames     #Changed colum names
    return(myDF)
#------------------------------#
#--- Main processing starts ---#
#------------------------------#
nr_classes = 6          #Number of populations
k_fold = range(0,10)    #Range of ks
iterations = 10         #Number of iterations
os.system("mkdir data")	#Created folder for generated data
#--- Create initial data ---#
data_1 = {'name': "BGPLVM_noResampling_manual", 'folder':"../../GPC/GPLVM/",        'isCNN': False, 'nrFeatures':14}
data_2 = {'name': "BGPLVM_noResampling_top14",  'folder':"../../GPC/GPLVM_full/",   'isCNN': False, 'nrFeatures':14}
data_3 = {'name': "Procrustes_noResampling",    'folder':"../../GPC/Procrustes/",   'isCNN': False, 'nrFeatures':28}
data_4 = {'name': "CNN_Aug_noResampling",       'folder':"../../CNN/Classification/Augmented/", 'isCNN': True, 'nrFeatures':0}
DF = pd.DataFrame(data=data_1,index=[0])
DF = DF.append(data_2,ignore_index=True)
DF = DF.append(data_3,ignore_index=True)
DF = DF.append(data_4,ignore_index=True)
#--- Prepare system ---#
os.system("rm -rf data/data*")  #Remove old data if exists
#---------------------------------#
#--- Prepare data and get data ---#
#---------------------------------#
for i in range(0,len(DF)): #Loop over all defined classifiers
    data = DF.loc[i,]   #Get classifier info
    name = data[2]      #Name of classifier
    path = data[0]      #Path to classifier data
    isCNN= bool(data[1])    #See if classifier was an CNN
    nrFeatures=data[3]  #Number of features of classifier
    #--- Do processing ---#
    # 1: Create data folder
    mkdir(name)     #Create folder with classifier name
    # 2: Calculate relevance
    RELEVANCE_SUM = np.zeros((iterations,nrFeatures))   #Create memory for all values
    for ITERATION in range(0,iterations):       #We got 10 iterations -> 0..9
        DF_classifier_result = pd.read_csv(path+str(ITERATION)+'_dataMatrix.csv')
        #--- Store here ---#
        DF_classifier_result.to_csv(getdir(name)+"/"+str(ITERATION)+"_classifierResult.csv", index=False)
        #--- Get relevance if available (no CNN) ---#
        if(not isCNN):
            relevance_local = np.zeros((len(k_fold),nrFeatures))            #Memory for later use - the col mean values
            relevance_per_population = np.zeros((nr_classes,nrFeatures))    #Memory for mean of each entry in the relevance matrix
            #To get the relevance we need to loop over all k's 
            for k in k_fold:    #Loop over all k's
                LS_of_k = np.genfromtxt(path+str(ITERATION)+'_foldNr_'+str(k)+'_LS.csv',delimiter=' ')
                #LS_of_k consists of the LS values of the GPC --> mxd matrix, where m is the number of populations ans d is the data dimension (here, 14)
                inv_LS_of_k = 1./LS_of_k            #Invert the lengthscale values
                colMeans = np.mean(inv_LS_of_k,0)   #Get col mean
                #--- Store data ---#
                relevance_per_population = relevance_per_population+inv_LS_of_k #Add relevance values together
                relevance_local[k,:] = colMeans                                 #Store col means
            #--- Post process relevance values ---#
            MEAN_relevance_per_population = relevance_per_population/len(k_fold)    #Get mean relevance values
            np.savetxt(getdir(name)+"/"+str(ITERATION)+"_MeanRelevance_perPopulation.csv",MEAN_relevance_per_population, delimiter=',')
            DF_ARD = convertARD(relevance_local)                        #Convert to pandas DF
            DF_ARD_mean = pd.DataFrame((np.mean(DF_ARD,0))).transpose() #Get col means
            DF_ARD_mean.to_csv(getdir(name)+"/"+str(ITERATION)+"_Relevance.csv",index=False)
            RELEVANCE_SUM[ITERATION,:]=np.array(DF_ARD_mean)
        #End if is CNN
    #End Iteration loop
    if(not isCNN):
        np.savetxt(getdir(name)+"/_RelevanceSUM.csv",RELEVANCE_SUM, delimiter=',')
