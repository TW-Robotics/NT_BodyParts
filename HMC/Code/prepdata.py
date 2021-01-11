#!../../Python/VE/bin/python
## Note: adjust the previous line to your environment or call as python prepdata.py ... arguments list ....
##
## prepdata.py is a dependency of ./runardnet.sh and ***NOT called directly***.
##
## The path to the python interpreter needs to match the location of
## the python environment you are going to use.
##
## To aid reproducing our results, this code is provided as supplement
## to the publication "Inferring Ethiopian Nile tilapia (Oreochromis
## niloticus) morphotypes with machine learning"
##
## The code depends on ./runardnet.sh and ./prepdata.py being
## available and R. Neals HMC software
## (https://www.cs.toronto.edu/~radford/fbm.software.html) installed
## and a respective path set.
##
## This code is available under a GPL v3.0 license and comes without
## any explicit or implicit warranty.
##
## (C) P. Sykacek 2020 <peter@sykacek.net>

## analyse data and check for mode.
import sys
import pandas as pd
import numpy as np
import subprocess as spc
import time
import os
import stat

isint = lambda val: type(val)==type(1) or np.round(val)==val

def all_int(inarr):
    ## all_int checks whether all entries in inarr are of type integer.
    inarr=inarr.flatten().tolist()
    return all([isint for entry in inarr])

def nsmplflds2splits(nosmpl, nofolds):
    ## we convert the number of samples and the number of folds
    ## to a list of ranges with start and end of the test data.
    ## This is compatible with Neals "numin" utility which is used
    ## for loading data. the indices we need for controlinmg fold
    ## iterations are 1 based.
    allstart=[]
    allend=[]
    cstart=0
    while nosmpl > 0:
        cspan=int(nosmpl/nofolds)
        cend=cstart+cspan-1
        allstart.append(cstart)
        allend.append(cend)
        cstart=cend+1
        nosmpl=nosmpl-cspan
        nofolds=nofolds-1
    tstids=1+np.transpose(np.array([allstart, allend]))
    return tstids

if __name__ == "__main__":
    # execute if run as a script
    if len(sys.argv)!=10:
        print(sys.argv)
        print(len(sys.argv))
        raise(ValueError("Call prepdata ifname simdir lognambase nohidden ardlevel (0,1 or 2) <hmc-iterations> nrfolds donormalise [Y/N] doreshuffle [Y/N]"))
    arglist=sys.argv
    for i in range(len(sys.argv)):
        print("{0}: {1}".format(i, arglist[i]))
    ifnam=arglist[1]
    simdir=arglist[2]
    lognambase=arglist[3]
    ## add simdir if there is no directory on lognambase
    if not os.path.dirname(lognambase):
        lognambase=simdir+lognambase
    logdir=os.path.dirname(lognambase)
    nohidden=int(arglist[4])
    ardlevel=int(arglist[5])
    hmc_it=int(arglist[6])
    hmc_burnin=int(0.25*hmc_it) ## ignored for predicitions
    nofolds=int(arglist[7])
    donormalise=arglist[8]=="Y" or arglist[8]=="y" or arglist[8]=="J" or arglist[8]=="j"
    doreshuffle=arglist[9]=="Y" or arglist[9]=="y" or arglist[9]=="J" or arglist[9]=="j"
    ## define prior settings for net-spüec in dependence of ARD level
    if ardlevel==0:
        ## no ard
        ih="x0.2:0.5"
        bh="0.1:0.5"
        ho="x0.05:0.5"
        io="x0.2:0.5"
        bo="1"
    elif ardlevel==1:
        ## ARD with one precision per parameter group used on input to hidden and input to output
        ih="x0.2:0.5:1"
        bh="0.1:0.5"
        ho="x0.05:0.5"
        io="x0.2:0.5:1"
        bo="1"
    elif ardlevel==2:
        ## ARD with one precision per parameter group and coefficient (only input to hidden)
        ih="x0.2:0.5:1:5"
        bh="0.1:0.5"
        ho="x0.05:0.5"
        io="x0.2:0.5:1"
        bo="1"

    ## print the call parameters
    print("./prepdata.py {0} {1} {2} {3} {4} {5} {6}".format(ifnam, simdir, lognambase, nohidden, ardlevel, hmc_it, nofolds))
    # read input data in Neal HMC compatible format.
    alldata=pd.read_csv(ifnam, sep=" ", header=None)
    nosamples=alldata.shape[0]
    nocols=alldata.shape[1]
    noin=nocols-1
    print("input data shape: {0}".format(alldata.shape))
    print("nr. samples: {0} nr. inputs: {1}".format(nosamples, noin))
    # in Neals format the last column contains the targets
    y=np.array(alldata.iloc[:,nocols-1])
    # analyse the data
    isclass=all_int(y)
    if isclass:
        ## we have a classification problem and need to find out how
        ## many classes we have ansd also relabel the data. This is
        ## done irrespective whether it is needed as it is at worst
        ## preserving the data.
        y=np.array([int(elem) for elem in y])
        unqlab=np.sort(list(set(y)))
        for idx, val in enumerate(unqlab):
            # set y[y==val]=idx -> this will relabel using consecutive labels.
            # and enforce a 0 based labeling of classes
            y[y==val]=idx
        if len(unqlab)>2:
            modelspc="class"
            target="class"
            noout=1
            maxint=len(unqlab)
        else:
            modelspc="binary"
            target="binary"
            noout=1
            maxint=2
    else:
        modelspc="real"
        regnoise="0.05:0.5" ## includes the prior for the noise level
        target="real"
        noout=1
    ## need to set the sghape of y such that the concatenation below works.
    y.shape=(len(y), 1)
    ## extract all regressors
    X=np.array(alldata.iloc[:,0:noin])
    ## normlise by removing location
    if donormalise:
        mnX=np.mean(X, axis=0)
        X=X-mnX
        ## and scale
        sdX=np.std(X, axis=0)
        sdX[sdX<np.finfo(float).eps]=np.finfo(float).eps
        X=X/sdX
    ## construct a range of row indices for reshuffling the data
    idre=np.arange(X.shape[0])  ## init to allow for reshuffling be optional!
    if doreshuffle:
        np.random.shuffle(idre)
    #print(idre)
    #print(X.shape)
    #print(y.shape)
    newdata=np.concatenate((X, y), axis=1)
    #print(newdata.shape)
    newdata=newdata[idre,:]
    #print(newdata.shape)
    ## save newdata as ifnam
    data=pd.DataFrame(newdata)
    if isclass:
        ## we have a classifier and use int32 as data type for the last column (otherwise data-spec raises an error).
        cnams=list(data)
        nocols=len(cnams)
        data=data.astype({cnams[nocols-1]:"int32"})
    # use pandas to_csv tp create the normalised data
    data.to_csv(ifnam, sep=' ', index=False, header=False)
    ## next we calculate one based row ranges for the folds 
    datacoords=nsmplflds2splits(nosamples, nofolds)
    datatrtst=datacoords.shape[0]==1 ## we have only one fold and use the data for training and testing
    ## loop over datacoords to get all folds set up
    allmclogs=[] # store the hmc logfile names for calling the final net-mc in a bash script
    for rwdx in range(datacoords.shape[0]):
        ## unless we have only one fold covering all data as training
        ## and test, the subsequent values denote the range of test
        ## data in the input file.
        startt=datacoords[rwdx,0]  # start of test data range 
        endt=datacoords[rwdx,1]    # end of test data range
        clgfnam="{0}_net_{1}.log".format(lognambase, rwdx)
        ## keep the names of the log files
        allmclogs.append(os.path.basename(clgfnam))
        ## execute the net-spec command
        if modelspc=="class":
            print(["net-spec", clgfnam, "{0}".format(noin), "{0}".format(nohidden),
                   "{0}".format(maxint), "/", "ih={0}".format(ih), "bh={0}".format(bh),
                   "ho={0}".format(ho), "io={0}".format(io), "bo={0}".format(bo)])
            spc.run(["net-spec", clgfnam, "{0}".format(noin), "{0}".format(nohidden),
                     "{0}".format(maxint), "/", "ih={0}".format(ih), "bh={0}".format(bh),
                     "ho={0}".format(ho), "io={0}".format(io), "bo={0}".format(bo)])
        else:
            print(["net-spec", clgfnam, "{0}".format(noin), "{0}".format(nohidden),
                   "{0}".format(noout), "/", "ih={0}".format(ih), "bh={0}".format(bh),
                   "ho={0}".format(ho), "io={0}".format(io), "bo={0}".format(bo)])
            spc.run(["net-spec", clgfnam, "{0}".format(noin), "{0}".format(nohidden),
                     "{0}".format(noout), "/", "ih={0}".format(ih), "bh={0}".format(bh),
                     "ho={0}".format(ho), "io={0}".format(io), "bo={0}".format(bo)])
        ## execute model-spec
        if isclass:
            print("modelspc: {0}".format(modelspc))
            spc.run(["model-spec", clgfnam, modelspc])
        else:
            ## we have a regression model
            print("modelspc: {0} {1}".format(modelspc, regnoise))
            spc.run(["model-spec", clgfnam, modelspc, regnoise])
        print("datatrtst: {0}".format(datatrtst))
        ## execute data-spec
        if datatrtst:
            ## we have only one fold and use the training data for inference and predicition
            if isclass:
                ## solution for classification
                spc.run(["data-spec", clgfnam, "{0}".format(noin), "{0}".format(noout),
                         "{0}".format(maxint), "/", "{0}@{1}:{2}".format(ifnam, startt, endt), ".",
                         "{0}@{1}:{2}".format(ifnam, startt, endt), "." ])
            else:
                ## solution for regression (without maxint)
                spc.run(["data-spec", clgfnam, "{0}".format(noin), "{0}".format(noout), "/",
                         "{0}@{1}:{2}".format(ifnam, startt, endt), ".", 
                         "{0}@{1}:{2}".format(ifnam, startt, endt), "." ])
        else:
            ## we have separate training and test instances accpording
            ## to numin.html we may use negative indices:
            ## @-<from>:<to> to exclude samples from training
            if isclass:
                ## solution for classification
                print(["data-spec", clgfnam, "{0}".format(noin), "{0}".format(noout),
                         "{0}".format(maxint), "/", "{0}@-{1}:{2}".format(ifnam, startt, endt), ".",
                         "{0}@{1}:{2}".format(ifnam, startt, endt), "." ])
                spc.run(["data-spec", clgfnam, "{0}".format(noin), "{0}".format(noout),
                         "{0}".format(maxint), "/", "{0}@-{1}:{2}".format(ifnam, startt, endt), ".",
                         "{0}@{1}:{2}".format(ifnam, startt, endt), "."])
            else:
                ## solution for regression (without maxint)
                spc.run(["data-spec", clgfnam, "{0}".format(noin), "{0}".format(noout), "/",
                         "{0}@-{1}:{2}".format(ifnam, startt, endt), ".",
                         "{0}@{1}:{2}".format(ifnam, startt, endt), "." ])
        ## initialise rand-seed with current Unix time in seconds
        rndseed=int(time.time())
        spc.run(["rand-seed", clgfnam, "{0}".format(rndseed)])
        ## initilise network from a random draw of parameters from the prior
        ## doing so allows for rerunning analyses and testing for convergence.
        ## the final "0" is for max-index (see data-gen.html)
        spc.run(["net-gen", clgfnam, "0"])
        ## specify the mc initialisation the commands follow Neals
        ## DELVE script settings which are independent of other
        ## settings always the same.
        spc.run(["mc-spec", clgfnam, "repeat", "50", "sample-noise", "heatbath", "hybrid", "10", "0.1"])
        spc.run(["net-mc", clgfnam, "1"])
        spc.run(["mc-spec", clgfnam, "repeat", "50", "sample-sigmas", "heatbath", "hybrid", "10", "0.1"])
        spc.run(["net-mc", clgfnam, "2"])
        ## final mc-spec to prepare the log file for the actual
        ## HMC-MCMC which happens in a script in parallel for all
        ## folds.
        spc.run(["mc-spec", clgfnam, "repeat", "10", "sample-sigmas",
                 "heatbath", "0.9", "hybrid", "90:5", "0.1", "negate",
                 "/", "leapfrog", "8"])
        ## construct net-mc and predicition script which does most of
        ## the simulation and can be called from a bash script
        mconam="{0}_{1}_runmc.sh".format(lognambase, rwdx)
        predoutnam="{0}_{1}_preds.txt".format(lognambase, rwdx)
        adroutnam="{0}_{1}_inputard.txt".format(lognambase, rwdx)
        with open(mconam,"w") as of:
            of.write("#!/bin/bash\n")
            of.write("# automatically generated HMC-MLP simulation script\n")
            of.write("# based on R. Neals extensive documentation and\n")
            of.write("# DELVE ARD examples.\n")
            of.write("#\n")
            of.write("# (C) P. Sykacek 2020 <peter@sykacek.net> \n")
            of.write("# HMC-MLP simulations  \n")
            of.write("net-mc {0} {1}\n".format(clgfnam, hmc_it))
            of.write("# true targets and mean probabilities or real valued model prdictions \n")
            of.write("net-pred tn {0} {1}: > {2}\n".format(clgfnam,  hmc_burnin, predoutnam))
            of.write("# first level hyper parameters for input to hidden and (linear) \n")
            of.write("# input to output weights (in dependence of prior this will show ARD) \n")
            of.write("# smaller values show less relevance. One olumn per input (two blocks). \n")   
            of.write("net-tbl th1@h4@ {0} > {1}".format(clgfnam, adroutnam))
        ##
        ## change mode of mconam to executable
        st = os.stat(mconam)
        os.chmod(mconam, st.st_mode | stat.S_IEXEC)
        ## write a log file of the preparatory step which aids parsing of results.
        resdir=os.path.dirname(lognambase)
        predstail="_preds.txt"
        ardtail="_inputard.txt"
        ## we also write target and noout.
        log4parsenam=lognambase+"_resparse.txt"
        ## we have still got to adjust noout in case we have a multiclass problem:
        if target=="class":
            noout=maxint
        with open(log4parsenam,"w") as of:
            of.write("RESDIR:\t{0}\n".format(resdir))     ## directory with results
            of.write("PREDS:\t{0}\n".format(predstail))   ## pattern for pterdiction files
            of.write("ARD:\t{0}\n".format(ardtail))       ## pattern for ARD information
            of.write("LOGDIR:\t{0}\n".format(logdir))     ## directory with net-mc log files
            of.write("LOGFILES:\t{0}\n".format(",".join(allmclogs))) ## comma separated list of log file names of simulation
            of.write("BURNIN:\t{0}\n".format(hmc_burnin)) ## we discard 25% of the samples. (Here for assessing ARD)
            of.write("TARGS:\t{0}\n".format(target))      ## txype of target
            of.write("MEANCOLS:\t{0}\n".format(noout))    ## columns of predictions (1 for regression and binary classification and no of classes in miulticlass)



if 0:
    ih="x0.2:0.5:1"
    bh="0.1:0.5"
    ho="x0.05:0.5"
    io="x0.2:0.5:1"
    bo="1"
    clgfnam="hmc_mlp.log"
    noin="#(inputs)"
    nohidden=20
    maxint=6
    print(" ".join(["net-spec", clgfnam, "{0}".format(noin), "{0}".format(nohidden),
                    "{0}".format(maxint), "/", "ih={0}".format(ih), "bh={0}".format(bh),
                    "ho={0}".format(ho), "io={0}".format(io), "bo={0}".format(bo)]))
