## make GPC and CNN output compatible with HMC-MLP.
##
## load libraries for data handling. 
import numpy as np
import pandas as pd
import copy
import re
import os
os.system("mkdir res_GPC_CNN")
resdir="./res_GPC_CNN/"     ## target location of results files
srcdir="./"                 ## source location of results
## source locations
allsrcs=["../../GPC/GPLVM/{0}_dataMatrix.csv",  
         "../../Vizualizations/PieVisualization/data/data_BGPLVM_noResampling_manual/{0}_Relevance.csv",
         "../../GPC/GPLVM_full/{0}_dataMatrix.csv",
         "../../Vizualizations/PieVisualization/data/data_BGPLVM_noResampling_top14/{0}_Relevance.csv",
         "../../CNN/Classification/Augmented/{0}_dataMatrix.csv",
         "../../GPC/Procrustes/{0}_dataMatrix.csv",
         "../../Vizualizations/PieVisualization/data/data_Procrustes_noResampling/{0}_Relevance.csv",
]
## target names
alltrgnams=["rshfl_gpred_gpc_it{0}_allpreds.csv",
            "rshfl_gpred_gpc_it{0}_allardres.csv",
            "rshfl_gpall_gpc_it{0}_allpreds.csv",
            "rshfl_gpall_gpc_it{0}_allardres.csv",
            "rshfl_cnn_it{0}_allpreds.csv",
            "rshfl_prc_gpc_it{0}_allpreds.csv",         
            "rshfl_prc_gpc_it{0}_allardres.csv",
]
## we have now got to load the data, reorder columns and write them to
## the target name
rgx_pred=re.compile(".*_allpreds.csv")
def s2t(sbase,snam,tbase,tnam):
    ## convert source to target format for one file
    targorder=["ttarg", "ptarg", "probs0", "probs1", "probs2", "probs3", "probs4", "probs5"]
    idata=pd.read_csv(sbase+snam, sep=',')
    if rgx_pred.match(tnam):
        idata=idata[targorder]
    idata.to_csv(tbase+tnam, sep=",", index=False)
## we run the conversion for all entries in allsrcs and alltrgnams and
## within for all files with numbers 0..9 (including boundaries)
for cit in range(len(allsrcs)):
    for simdx in range(10):
        s2t(srcdir, allsrcs[cit].format(simdx), resdir, alltrgnams[cit].format(simdx))
## We now copy the HMC results in this folder
os.system("cp -r ../../HMC/Code/Data/resdata res_HMC")
