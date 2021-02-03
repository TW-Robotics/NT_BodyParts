# This scripts converts the HMC relevance files to sorted relevance 
# files.
#
# This code is available under a GPL v3.0 license and comes without
# any explicit or implicit warranty.
#
# (C) Wilfried Woeber 2020 <wilfried.woeber@technikum-wien.at>
import numpy as np 
import os
import pandas as pd
data_1 = {'name': "BGPLVM_noResampling_manual", 'folder':"./resdata/BGPLVM_NoReplace",     'isProc':False}
data_2 = {'name': "BGPLVM_noResampling_top14",  'folder':"./resdata/BGPLVM_NoReplace_14",  'isProc':False}
data_3 = {'name': "Procrustes_noResampling",    'folder':"./resdata/Procrustes_NoReplace", 'isProc':True} #Actually 28 features...
#data_4 = {'name': "CNN_Aug_noResampling",       'folder':"../../CNN/Classification/Augmented/", 'isCNN': True, 'nrFeatures':0, 'isProc':False}
DF = pd.DataFrame(data=data_1,index=[0])
DF = DF.append(data_2,ignore_index=True)
DF = DF.append(data_3,ignore_index=True)
#--- Create target order ---#
for i in range(0,len(DF)):
    name=DF.loc[i]['name']          #Get name of dataset
    data_path=DF.loc[i]['folder']   #Path to data
    isProc = DF.loc[i]['isProc']    #Procrustes flag
    #--- Get cols to seek in the data ---#
    mxDim = 14
    if(isProc):
        mxDim=28
    #--- Create order ---#
    targorder = []                  #Memory for target order
    for n in range(0,mxDim):
        targorder.append("ard_"+str(n))
    for n in range(0,mxDim):
        targorder.append("nonlin_ard_"+str(n))
    for n in range(0,mxDim):
        targorder.append("lin_ard_"+str(n))
    #--- Load file ---#
    for ITERATION in range(0,10):
        relevance = pd.read_csv(data_path+"/"+str(ITERATION)+"_Relevance_.csv")
        #--- Create corrected file ---#
        relevance_real = []
        for n in range(0,len(targorder)):
            relevance_real.append(float(relevance[targorder[n]]))    #Integrate ARD value as a float
        DF_relevance = pd.DataFrame(data=relevance_real, index=targorder)
        DF_relevance = DF_relevance.transpose()
        DF_relevance.to_csv(data_path+"/"+str(ITERATION)+"_Relevance.csv",index=False)
