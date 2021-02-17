# coding: utf-8
# Evaluation of Tilapia GP-LVM features and GPA transformed landmarks
# The approach relies on reshuffling with identical sample indices used accross all competing 
# methods (GPC for GP-LVM features and GPA transformed landmarks and CNNs on images) to obtain 
# probabiliteis for class, class labels and ARD assessments. We compare the results with 
# Neals HMC-MLP against Gaussian process classification results and CNN / deep learning based results.
# # Supplement to "Inferring Ethiopian Nile tilapia (Oreochromis niloticus) morphotypes with machine learning"
# 
# This code is available under a GPL v3.0 license and comes without any explicit or implicit warranty.
# 
# (C) P. Sykacek 2020 <peter@sykacek.net>
import numpy as np
import pandas as pd
#----------------------------------#
#--- Initially convert the data ---#
#----------------------------------#
import os
os.system("rm -rf ./resdata")
os.system("cp -r ./res_HMC resdata")    #Copy HMC results
os.system("cp ./res_GPC_CNN/*cnn* ./resdata/.")
os.system("cp ./res_GPC_CNN/*gpc* ./resdata/.")
os.system("cp ./res_GPC_CNN/*allardres* ./resdata/.")
targorder=["ttarg", "ptarg", "probs0", "probs1", "probs2", "probs3", "probs4", "probs5"]
for NAME in ('prc','gpred','gpall'):
    for i in range(0,10):
        test=pd.read_csv("./resdata/rshfl_"+NAME+"_it"+str(i)+"__allpredres.csv", sep=',')
        test=test[targorder]
        test.to_csv("./resdata/rshfl_"+NAME+"_it"+str(i)+"__allpredres.csv", sep=",", index=False)

## we have patterns for the different input data 
## 1) procrustes landmarks
## 2) 14 top ranked GP-LVM dimensions
## 3) 14 selected GP-LVM dimensions which 
##    have no noticeable technical variation 
##    (to avoid clever Hans effects)
##
## For every input modality we resample 10 times by reshuffling
##
## Every instance is assessed by 10 fold cv with
##
## Neals HMC-MLP (20 hidden units level 1 ARD) (60 result files)
## GPC from the Ghahramani group               (60 result files)
## 
## Images are analysed directly using a
## CNN Implementation from the Tensorflow Python binding. 
## (20 result files)
## for data loading and metrics calculation we use the evalres.py 
## library functions
import evalres as evr
## use the code to load all prediction results and produce a
## respective dataframe.
resdir="./resdata/"
## 1) load the experimental description
infodf=evr.info2df(evr.infomap)
## the iteration parterns in files is 0..9 (including)
idsperpatt=list(range(10))
allresdict=evr.loadallres(resdir, infodf , idsperpatt)
## we may now convert the results to metrics.  This needs to consider
## the dofferent samüple orders in bootstrap and reshuffle. providing
## samplesel as specified will analyse 'Bootstrapped' resampled data
## and 'Reshuffled' resampled data as separate groups. The latter is
## required as McNemar can not compare different sample orders.#
samplesel={'reshuffle':{'resampling':'Reshuffled'}}
annotatedresults=evr.resdict2metrics(allresdict, infodf, selinfo=samplesel)
## annotatedresults is a dataframe which contains the following 
## metrics (column names)
## 'Acc': generalisation accuracy, 'MI': mutual information, 
##
## 'Sig': McNemar significance when comparing against the least 
## performing competitor in each group (that is calculated separately 
## for 'Bootstrapped' and 'Reshuffled' data.

## To avoid problems with visualisation we threshold McNemar 
## Significance levlels below 0.999
##affectedrows=annotatedresults.index[annotatedresults["Sig"]>0.999]
##print(affectedrows)
##annotatedresults.loc[affectedrows, "Sig"]=0.999
## add a column with logit transformed significance values
annotatedresults_A=annotatedresults['A']
annotatedresults_B=annotatedresults['B']
#annotatedresults["logitSig"]=evr.logit(annotatedresults_A['Sig'].tolist()) #evr.logit(annotatedresults["Sig"].tolist())
## comparison with default predictor
defcomp=evr.resvsdefpred(allresdict)
## We will now analyse the reshuffled experiments only
#rsdf=annotatedresults_B.iloc[(annotatedresults_B['resampling']=='Reshuffled').tolist(), :]
## We will now prepare visualisation of gne acc by multiplying with 100
rsdf = annotatedresults_A
rsdf.loc[:,"Acc"]=rsdf.loc[:,"Acc"]*100
## We now add names
predmethod=[]
for INDEX in rsdf.index:
    if INDEX == 'netprcshfl':
        predmethod.append('GPA+HMC-MLP')
    elif INDEX == 'netgptopshfl':
        predmethod.append('Top GP-LVM+HMC-MLP')
    elif INDEX == 'netgpselshfl':
        predmethod.append('Sel GP-LVM+HMC-MLP')
    elif INDEX == 'gpcgpselshfl':
        predmethod.append('Sel GP-LVM+GPC')
    elif INDEX == 'gpcgptopshfl':
        predmethod.append('Top GP-LVM+GPC')
    elif INDEX == 'gpcprcshfl':
        predmethod.append('GPA+GPC')
    elif INDEX == 'cnnshfl':
        predmethod.append('CNV+CNN')
rsdf["predmethod"]=predmethod       #Add to existing data frame
import matplotlib.pyplot as plt
## and can finally produce a boxplot for generlisation acciracy over methods
fh=4        # figure height
fw=4*fh     # and width
lbsz=20     # font size of axis legends
ttsz=28     # and plot title
# ### Generate boxplot for generalisation accuracy
# ### Average metrics per group
rsdf.groupby("predmethod").mean()
#rsdf.groupby("predmethod").mean().to_latex("allmetrics.tex")
ax=rsdf.boxplot(column='Acc', by="predmethod", figsize=(fw, fh)) # use pandas
plt.suptitle("")      ## remove the default "By ... text"
## add annotations
plt.title("", fontsize=ttsz)
plt.xlabel("Analysis methods", fontsize=lbsz)
plt.ylabel("Correctly classified [%]", fontsize=lbsz)
## generate plot and inline visualisation
plt.savefig("tilapia_genacc.pdf", bbox_inches='tight')
#plt.show()
# ### Mutual information
ax=rsdf.boxplot(column='MI', by="predmethod", figsize=(fw, fh)) # use pandas
plt.suptitle("")      ## remove the default "By ... text"
## add annotations
plt.title("", fontsize=ttsz)
plt.xlabel("Analysis methods", fontsize=lbsz)
plt.ylabel("Channel capacity [bit]", fontsize=lbsz)
## generate plot and inline visualisation
plt.savefig("tilapia_mutinf.pdf", bbox_inches='tight')
#plt.show()
# ### McNemar significance comparing against GPA+GPC
## McNemar significance is displayed on logit scale to enlarge differences
## Including GPA+GPC makes no sense -> we remove it from the dataframe.
pval =[]    #Memory for p-values of MCNemars test
for key in allresdict.keys():
    for idx in range(0,10):
        base_res=allresdict['gpcprcshfl']
        curr_res=allresdict[key]
        na,nb=evr.lab2cnt(base_res[idx]["ptarg"],curr_res[idx]["ptarg"],curr_res[idx]["ttarg"])
        pval.append(np.minimum(evr.mcnemar(na,nb),0.999))
## Convert to logits for visualization
logitSig=[] #Logit memory
for sig in pval:
    logitSig.append(evr.logit(sig))
rsdf["logitSig"]=logitSig   #Append to DF
rsdf.set_index("predmethod", inplace=True, drop=False)
rsdf_FULL = rsdf.copy()
rsdf.drop(index="GPA+GPC", inplace=True)
rsdf.reset_index(inplace=True, drop=True)
## we may now go on as above
ax=rsdf.boxplot(column='logitSig', by="predmethod", figsize=(fw, fh)) # use pandas
plt.suptitle("")      ## remove the default "By ... text"
## add annotations
plt.title("", fontsize=ttsz)
plt.xlabel("Analysis methods", fontsize=lbsz)
plt.ylabel("logit(p-value)", fontsize=lbsz)
## generate plot and inline visualisation
plt.savefig("tilapia_mcnemar.pdf", bbox_inches='tight')
#plt.show()
rsdf.set_index("predmethod", inplace=True, drop=False)
rsdf.loc["GPA+HMC-MLP"]
#--- Vizualization ---#
result = pd.DataFrame({'predmethod':rsdf_FULL.index, 'ACC': np.array(rsdf_FULL['Acc']), 'MI': np.array(rsdf_FULL['MI']), 'logitSig': [k[0] for k in np.array(rsdf_FULL['logitSig'])]})
result.groupby("predmethod").mean()
result.groupby("predmethod").mean().to_latex("allmetrics.tex")
import matplotlib.pyplot as plt
from pandas.plotting import table
ax = plt.subplot(222, frame_on=False) # no visible frame
ax.xaxis.set_visible(False)  # hide the x axis
ax.yaxis.set_visible(False)  # hide the y axis
table(ax, result.groupby("predmethod").mean(),loc='center')  # where df is your data frame
plt.savefig('table.pdf')
os.system("pdfcrop --margins '0 0 0 0' --clip table.pdf table.pdf")
