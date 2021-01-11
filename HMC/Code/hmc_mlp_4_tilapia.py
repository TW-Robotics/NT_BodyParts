## python script for generating HMC-MLP results on top GP-LVM,
## selected GP-LVM and GPA inputs.  Depends on R. Neals HMC-MLP being
## available. To aid reproducing our results, this code is provided as
## supplement to the publication "Inferring Ethiopian Nile tilapia
## (Oreochromis niloticus) morphotypes with machine learning"
## 
## This python 3 script should be called as
## 1) for analysing GPA transformed landmarks:
## python hmc_mlp4tilapia.py "procrustes"
## 2) for analysing the modes of the (by GP-LVM ARD asssessed) 14 top ranked GP-LVM dimensions. 
## python hmc_mlp4tilapia.py "gplvm_all"
## 3) for analysing the modes of the (by GP-LVM ARD asssessed) 14 selected GP-LVM dimensions.
## python hmc_mlp4tilapia.py "gplvm_red"
##
##
## Note: fold testing is done using maxthreads parallel tasks on a SMD
## machine - you may have to adjust the variable below if you have
## less than 10 cores available. The script depends on ./runardnet.sh
## and ./prepdata.py being available and R. Neals HMC software
## (https://www.cs.toronto.edu/~radford/fbm.software.html) installed
## and a respective path set.
##
## This code is available under a GPL v3.0 license and comes without
## any explicit or implicit warranty.
##
## (C) P. Sykacek 2020 <peter@sykacek.net> 

## load libraries for data handling. 
## We use subprocess on runardnet.sh 
## for doing the dispatch
import subprocess as sbp
import numpy as np
import pandas as pd
import sys ## for argv
## the next two are for file handling
import fnmatch
import os
## directories
indatadir="../../Data/"
## created folders
os.system("mkdir Data")
os.system("cd Data; mkdir subsampdir")
os.system("cd Data; mkdir hmclog")
os.system("cd Data; mkdir resdata")

ssdir="./Data/subsampdir/"
logdir="./Data/hmclog/"
resdir="./Data/resdata/"
## script to run neals hmc-mlp 10 fold assessment
hmc_script="./runardnet.sh"
## GPA transformed procrustes data
prcfnam="../../Procrustes/PROCRUSTES_DATA.csv"
## GP-LVM modes
gplvmnam="../../GPLVM/BGPLVM_DATA.csv"
## columns with selected gp-lvm features (excludes variance in the
## background).
gpselfnam="unselectedFeatures.csv"
gplvm_selfromX=pd.read_csv(indatadir+gpselfnam, header=None).values.flatten().tolist()
## 10 iterations of reshuffling sample indices 
id4reshuffle="NoReplace_20205809_085859.csv"
## 10 iterations of resampled (with replacement) sample indices
id4resample="replace_20205809_085859.csv"
## decide about which data to use
doprocrustes="procrustes" in sys.argv   # procrustes features 
dogplvmall="gplvm_all" in sys.argv      # all gp-lvm dimensions 
dogplvmred="gplvm_red" in sys.argv      # gp-lvm discarding the first two dimensions
if doprocrustes:
    ## load preocrustes data
    data=pd.read_csv(prcfnam, header=None)
    sampnams=list(data.iloc[:,0])       ## unique sample names in first column
    y_orig=np.array(list(data.iloc[:,1]))    ## sample lables - integer location idx
    ## procrustes transformed locations as numpy array
    ## (mark1.x, mark1.y, mark2.x, ...) 
    X_orig=data.iloc[:,2:].to_numpy(copy=True)   
else:
    ## load the gp-lvm modes
    data=pd.read_csv(gplvmnam, header=None)
    ## prepare the gp-lvm data:
    sampnams=list(data.iloc[:,0])       ## unique sample names in first column
    y_orig=np.array(list(data.iloc[:,1]))    ## sample lables - integer location idx
    ## modes of GP-lvm inferred latent representation as numpy array 
    X_fromfile=data.iloc[:,2:].to_numpy(copy=True)
    ## take top 14
    X_orig=X_fromfile[:,0:14]
    ## in case of dogplvmred we drop the first two clumns in X_orig as
    ## these data are most likely of technical nature.
    if dogplvmred:
        ## take selected 14 as stored in gplvm_selfromX
        X_orig=X_fromfile[:,gplvm_selfromX]

allflsmpdx=pd.read_csv(indatadir+id4reshuffle, header=None, sep=" ").to_numpy()
print("Randomisation by reshuffle selected!")

## testing interface between subprocess.call and flag based script
## control - works as shown!
## sbp.call(['./test.sh', '-z', 'zval', '-a', '1'])

## we are now ready to run neals hmc-mlp
## -> we call runardnet.sh which is in directory 
## ./runardnet.sh -h $nohidden -a $ard1 -f $nofolds -m $maxthreads -n $hmciter -i "$simdir""$data_all" -d $simdir -l $lognamebase_all_ard1 -z Y

## general settings
nohidden=20  # one hidden layer MLP with 20 tanh units 
ardlevel=1   # use ARD level 1 
nofolds=10    # 10 folds which are exectuted in parallel
maxthreads=10 # 10 threads for fold processing
hmciter=2500    # 10 hmc samples for testing, 2500 for priduction
simdir=logdir # storage of all hmc output (drawn samples and processing logs)
donormalise='Y' ## we apply a normalisation step (all input columns to zero mean and unit std deviation).
dpreshuffle='N' ## we reshuffle here and do not repeat that in the
                ## script as we want identical order in all tested
                ## algorithms

## below we use X and y 
X=X_orig
y=y_orig

## specify lognambase and fnamebase in dependence of feature data
## (completed in response to randomisation iteration)

if doprocrustes:
    lognambase="rshfl_prc_it{0}_"
    fnambase=ssdir+"prc_rshfl_{0}.txt"
elif dogplvmall:
    lognambase="rshfl_gpall_it{0}_"
    fnambase=ssdir+"gpall_rshfl_{0}.txt"
else:
    lognambase="rshfl_gpred_it{0}_"
    fnambase=ssdir+"gpred_rshfl_{0}.txt"

## we may now loop over the resampling iterations
nresample= allflsmpdx.shape[1]  ## number of resampling indices
for cit in range(nresample):
    print("Doing procrustes resample iteration {0}".format(cit))
    ## first step: we prepare the data by applying the resampling
    csmpdx=allflsmpdx[:,cit]
    Xcit=X[csmpdx,:]
    ycit=y[csmpdx]
    ycit.shape=(len(ycit),1)
    data=pd.DataFrame(np.concatenate((Xcit,ycit), axis=1))
    cnams=list(data)
    nocols=len(cnams)
    data=data.astype({cnams[nocols-1]:"int32"})
    fnam=fnambase.format(cit)
    ## np.savetxt(fnam, data, delimiter=" ") -> does not work because
    ## we need integer class labels hence the pandas route!
    data.to_csv(fnam, sep=' ', index=False, header=False) ## produce a data file which is compatible with Neals HMC.
    lognambs=lognambase.format(cit)
    ## call neal sampling
    ##sbp.call('./runardnet.sh', '-h', "{0}".format(nohidden), '-a', "{0}".format(ardlevel),
    ##         '-f', "{0}".format(nofolds), '-m', "{0}".format(maxthreads),
    ##         '-n', "{0}".format(hmciter), '-i', fnam, '-d', logdir, '-l', lognambs,
    ##         '-z', 'Y', '-s', 'N')
    sbp.call('./runardnet.sh -h '+"{0}".format(nohidden) + ' -a '+"{0}".format(ardlevel) +
             ' -f '+ "{0}".format(nofolds) + ' -m '+ "{0}".format(maxthreads)+
             ' -n '+ "{0}".format(hmciter)+ ' -i '+ fnam+ ' -d '+ logdir+ ' -l '+ lognambs+
             ' -z '+ 'Y'+ ' -s '+ 'N', shell=True)

## we may now copy the collected results from logdir to resdir
f2move=[fnam for fnam in os.listdir(logdir) if fnmatch.fnmatch(fnam, '*_allpredres.csv') or fnmatch.fnmatch(fnam, '*_allardres.csv')]
for fnam in f2move:
    os.replace(logdir+fnam, resdir+fnam)

## we may now analyse the bootstrap resampled features
allflsmpdx=pd.read_csv(indatadir+id4resample, header=None, sep=" ").to_numpy()
print("Randomisation by bootstrap selected!")
if doprocrustes:
    lognambase="rsmp_prc_it{0}_"
    fnambase=ssdir+"prc_rsmp_{0}.txt"
elif dogplvmall:
    lognambase="rsmp_gpall_it{0}_"
    fnambase=ssdir+"gpall_rsmp_{0}.txt"
else:
    lognambase="rsmp_gpred_it{0}_"
    fnambase=ssdir+"gpred_rsmp_{0}.txt"

## we may now loop over the resampling iterations
nresample= allflsmpdx.shape[1]  ## number of resampling indices
for cit in range(nresample):
    print("Doing procrustes resample iteration {0}".format(cit))
    ## first step: we prepare the data by applying the resampling
    csmpdx=allflsmpdx[:,cit]
    Xcit=X[csmpdx,:]
    ycit=y[csmpdx]
    ycit.shape=(len(ycit),1)
    data=pd.DataFrame(np.concatenate((Xcit,ycit), axis=1))
    cnams=list(data)
    nocols=len(cnams)
    data=data.astype({cnams[nocols-1]:"int32"})
    fnam=fnambase.format(cit)
    ## np.savetxt(fnam, data, delimiter=" ") -> does not work because
    ## we need integer class labels hence the pandas route!
    data.to_csv(fnam, sep=' ', index=False, header=False) ## produce a data file which is compatible with Neals HMC.
    lognambs=lognambase.format(cit)
    ## call neal sampling
    ##sbp.call('./runardnet.sh', '-h', "{0}".format(nohidden), '-a', "{0}".format(ardlevel),
    ##         '-f', "{0}".format(nofolds), '-m', "{0}".format(maxthreads),
    ##         '-n', "{0}".format(hmciter), '-i', fnam, '-d', logdir, '-l', lognambs,
    ##         '-z', 'Y', '-s', 'N')
    sbp.call('./runardnet.sh -h '+"{0}".format(nohidden) + ' -a '+"{0}".format(ardlevel) +
             ' -f '+ "{0}".format(nofolds) + ' -m '+ "{0}".format(maxthreads)+
             ' -n '+ "{0}".format(hmciter)+ ' -i '+ fnam+ ' -d '+ logdir+ ' -l '+ lognambs+
             ' -z '+ 'Y'+ ' -s '+ 'N', shell=True)

## we may now copy the collected results from logdir to resdir
f2move=[fnam for fnam in os.listdir(logdir) if fnmatch.fnmatch(fnam, '*_allpredres.csv') or fnmatch.fnmatch(fnam, '*_allardres.csv')]
for fnam in f2move:
    os.replace(logdir+fnam, resdir+fnam)
