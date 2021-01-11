#!/opt/intel/ipy3.19.v5/intelpython3/bin/python
## to use in a shell script in my ipy3 environment (on wall) call as:
##
## ExecStart=/opt/intel/ipy3.19.v5/intelpython3/bin/python ./readmcres.py ... with proper set arguments ...
##
## note that the path needs to match the location of the python
## environment you are going to use. 
##
## (C) P. Sykacek 2020 <peter@sykacek.net>

## read and process Neals HMC-MLP results. We focus on ARD and predictions. 
import sys
import pandas as pd
import numpy as np
import subprocess as spc
import time
import os
import fnmatch
import stat
import collections as coll
import re
## regular expressions and lambda functions for parsing the logfile we
## create during hmc preparation
## results directors containing all files
rgx_isrdir = re.compile('RESDIR:.*')
isresdir = lambda instr:rgx_isrdir.match(instr)!=None
rgx_getrdir = re.compile('(?<=RESDIR:).*') # use as res=rgx_getgid.search(line) goid=res.group(0)
getresdir = lambda instr:((rgx_getrdir.search(instr)).group(0)).strip()

## patterm for files containing predicitions
rgx_ispreds = re.compile('PREDS:.*')
ispreds = lambda instr:rgx_ispreds.match(instr)!=None
rgx_getpreds = re.compile('(?<=PREDS:).*') # use as res=rgx_getgid.search(line) goid=res.group(0)
getpreds = lambda instr:((rgx_getpreds.search(instr)).group(0)).strip()

## pattern for files with ARD weights of inputs
rgx_isard = re.compile('ARD:.*')
isard = lambda instr:rgx_isard.match(instr)!=None
rgx_getard = re.compile('(?<=ARD:).*') # use as res=rgx_getgid.search(line) goid=res.group(0)
getard = lambda instr:((rgx_getard.search(instr)).group(0)).strip()

## pattern for log file directory
rgx_islogdir = re.compile('LOGDIR:.*')
islogdir = lambda instr:rgx_islogdir.match(instr)!=None
rgx_getlogdir = re.compile('(?<=LOGDIR:).*') # use as res=rgx_getgid.search(line) goid=res.group(0)
getlogdir = lambda instr:((rgx_getlogdir.search(instr)).group(0)).strip()

## pattern for net-mc log files
rgx_islogfiles = re.compile('LOGFILES:.*')
islogfiles = lambda instr:rgx_islogfiles.match(instr)!=None
rgx_getlogfiles = re.compile('(?<=LOGFILES:).*') # use as res=rgx_getgid.search(line) goid=res.group(0)
getlogfiles = lambda instr:[fnam.strip() for fnam in (((rgx_getlogfiles.search(instr)).group(0)).strip()).split(',')]

## pattern for burnin counter
rgx_isburnin = re.compile('BURNIN:.*')
isburnin = lambda instr:rgx_isburnin.match(instr)!=None
rgx_getburnin = re.compile('(?<=BURNIN:).*') # use as res=rgx_getgid.search(line) goid=res.group(0)
getburnin = lambda instr:((rgx_getburnin.search(instr)).group(0)).strip()

## pattern for target type
rgx_istrgtyp = re.compile('TARGS:.*')
istrgtyp = lambda instr:rgx_istrgtyp.match(instr)!=None
rgx_gettrgtyp = re.compile('(?<=TARGS:).*') # use as res=rgx_getgid.search(line) goid=res.group(0)
gettrgtyp = lambda instr:((rgx_gettrgtyp.search(instr)).group(0)).strip()

## pattern for number of meancols entries:
rgx_ismeans = re.compile('MEANCOLS:.*')
ismeans = lambda instr:rgx_ismeans.match(instr)!=None
rgx_getmeans = re.compile('(?<=MEANCOLS:).*') # use as res=rgx_getgid.search(line) goid=res.group(0)
getmeans = lambda instr:((rgx_getmeans.search(instr)).group(0)).strip()

## define a datatype which holds information from the log file created
## by prepdata.py
##
## resdir:     location (directory) with HMC results
##
## predfpatt:  trailing pattern of predicition results ("_preds.txt")
##
## ardfpatt:   trailing pattern of input ARD information
##             ("_inputard.txt")
##
## nrburnin:   nr of burnin samples
##
## targettyp:  type of target model - "real" for univariate regression
##             models; "binary" for binary classification (0/1
##             targets) and "class" for 1-of-c classification.
##
## nrmeancols: nr of mean columns (1 for "real" and "binary" and
##             <nr. of classes> for "class"
LogInfo=coll.namedtuple("LogInfo","resdir predfpatt ardfpatt logdir logfiles nrburnin targettyp nrmeancols")
def readhmcinfo(logfnam):
    ## readhmcinfo reads a log file logfnam which is constructed by
    ## prepdata and stores all information in an object of type
    ## LogInfo
    ##
    ## IN
    ##
    ## logfnam: a log file name whcih is read and parsed.
    ##
    ## OUT
    ##
    ## loginfo: a LogInfo object which contains the information
    ##          extracted from the log file.
    ##
    ## (C) P. Sykacek 2020 <peter@sykacek.net>
    with open(logfnam) as of:
        cline=of.readline()
        while cline != "":
            if isresdir(cline):
                resdir=getresdir(cline)
            elif ispreds(cline):
                predfpatt=getpreds(cline)
            elif isard(cline):
                ardfpatt=getard(cline)
            elif islogdir(cline):
                logdir=getlogdir(cline)
            elif islogfiles(cline):
                logfiles=getlogfiles(cline)
            elif isburnin(cline):
                nrburnin=int(getburnin(cline))
            elif istrgtyp(cline):
                targettyp=gettrgtyp(cline)
            elif ismeans(cline):
                nrmeancols=int(getmeans(cline))
            cline=of.readline()
    ## done with parsing: we return the parameterised LogInfo object
    return LogInfo(resdir=resdir, predfpatt=predfpatt,
                   ardfpatt=ardfpatt, logdir=logdir,
                   logfiles=logfiles, nrburnin=nrburnin,
                   targettyp=targettyp, nrmeancols=nrmeancols)

# testing to load a log file which summarizes the entire simulation
# (in general an n-fold test)

# loginfo=readhmcinfo("testlogf.txt")

## fucntions for reading net-pred simulation output

tonum=lambda instr:[float(part) for part in instr.split()]

def readres(fnam):
    ## readres reads output files from hmc-mlp net-pred tn commands
    ## for binary and 1-of-c classification and regression models.
    ## the function returns a tuple with case numbers, true targets and
    ## predictions.  Predictions are the mode in case of regression
    ## and expected probabilities in case of classification.
    ##
    ## IN
    ##
    ## fnam:   file name with path
    ##
    ## OUT
    ##
    ## (smpno,:    sample number
    ##  trtrgs,:   true targets
    ##  means):    predicted targets - for 1-of-c a [nsmp x c] matrix
    ##
    ## (C) 2020 P. Sykacek <peter@sykacek.net>

    ccnt=0
    ctrt=1
    cpred=[]
    getcp=True
    smpno=[]
    trtrgs=[]
    means=[]
    with open(fnam) as ifl:
        for cline in ifl:
            try:
                ## can we convert cline to numetical values?
                numvals=tonum(cline)
                ##print(cline)
                ##print(len(numvals))
                if len(numvals)>2:
                    ## if we can't convert we raise an error otherwise we get here
                    if getcp:
                        novals=len(numvals)
                        if novals>4:
                            cpred=slice(2,(novals-1))
                        else:
                            cpred=2
                        getcp=False
                    ## store the data
                    smpno.append(numvals[ccnt])
                    trtrgs.append(numvals[ctrt])
                    means.append(numvals[cpred])
            except ValueError:
                pass
    ## done we may return the values as tupple of lists (in case of
    ## 1-of-c means is a list of lists)
    return (smpno, trtrgs, means)
        
## analysing net-pred output for getting probabilities and predictions.
PredStats=coll.namedtuple("PredStats", "mode ttarg ptarg probs")
## PredStats predicition statistics of inferred model
##
## mode: type of target model - "real" for univariate regression
##       models; "binary" for binary classification (0/1 targets) and
##       "class" for 1-of-c classification.
## ttarg: true target (one dim vector: 0/1, 0..(c-1) or real) 
##
## ptarg: predicted target (similar to above)
##
## probs: (a one dim column vector with probability of 1 in binary
##        case and a [nsample x c] dim matrix with row normalised
##        probabilities in case of 1-of-c classification.

def netlog2preds(loginfo):
    ## netlog2preds converts the LOGDIR and LOGFILES information
    ## (loginfo.logdir and loginfo.logfiles) in a runardnet.sh
    ## generated simulation log to a file with prediction results.
    ## This version maintains in the predicted outputs the order of
    ## logfiles as they appear in the LOGFILES: entry (always
    ## appending) we use net-pred for generating temporary prediction
    ## result files which are subsequently deleted.
    ##
    ## IN
    ##
    ## loginfo: object of type LogInfo describing the log information
    ##          of an MLP-HMC fold experiment.
    ## 
    ## OUT
    ##
    ## predstats: object of type PredStats sumarising prediction results.
    ##
    ## (C) P. Sykacek 2020 <peter@sykacek.net>
    resdir=loginfo.resdir
    if resdir[len(resdir)-1] != "/":
        resdir=resdir+"/"
    logdir=loginfo.logdir
    if logdir[len(logdir)-1] != "/":
        logdir=logdir+"/"
    ## convert log filenames to predition outputs 
    allpredfiles=[os.path.splitext(lognam)[0]+'.txt' for lognam in loginfo.logfiles]
    ## we have now got to execute net-pred for all logfiles to produce
    ## the prediciton outputs
    nofiles=len(allpredfiles)
    for fno in range(nofiles):
        ## we execute all net-pred commands in line with the call used in runardnet.sh
        clognam=logdir+loginfo.logfiles[fno]
        cresfnam=resdir+allpredfiles[fno]
        burnin=loginfo.nrburnin
        print("logf:{0} resf:{1} burnin:{2}".format(clognam, cresfnam, burnin))
        spc.run("net-pred tn {0} {1}: > {2}".format(clognam, burnin, cresfnam), shell=True)
    ## we have now all predictions in the allpredfiles in directory
    ## loginfo.resdir and can thus load and combine them.
    allsmpno=[]
    alltrtrgs=[]
    allmeans=[]
    print("collecting")
    for fnam in allpredfiles:
        fnam=resdir+fnam ## add directory
        print("from file:{0}".format(fnam))
        (smpno, trtrgs, means)=readres(fnam) ## and load the data.
        allsmpno=allsmpno+smpno
        alltrtrgs=alltrtrgs+trtrgs
        allmeans=allmeans+means
        ## remove the file
        os.remove(fnam)
    print("generating predstats")
    ## done reading we construct and return a PredStats instance
    if loginfo.targettyp=="real":
        predstats=PredStats(mode=loginfo.targettyp, ttarg=np.array(alltrtrgs), ptarg=np.array(allmeans), probs=None)
    elif loginfo.targettyp=="binary":
        probs=np.array(allmeans)
        ptarg=np.zeros(probs.shape)
        ptarg[probs>0.5]=1
        predstats=PredStats(mode=loginfo.targettyp, ttarg=np.array(alltrtrgs), ptarg=ptarg, probs=probs)
    else:
        ## 1-of-c
        probs=np.array(allmeans)
        ptarg=np.argmax(probs, axis=1)
        predstats=PredStats(mode=loginfo.targettyp, ttarg=np.array(alltrtrgs), ptarg=ptarg, probs=probs)
    return predstats
    
    
def netpredstats(loginfo):
    ## netpredstats collects HMC-MLP prediction results for the
    ## specification provided as loginfo.
    ##
    ## IN
    ## 
    ## loginfo: object of type LogInfo describing the log information
    ##          of an MLP-HMC fold experiment.
    ##
    ## OUT
    ##
    ## predstats: object of type PredStats sumarising prediction results.
    ##
    ## (C) P. Sykacek 2020 <peter@sykacek.net> 

    ## get all predicition files (there are more than one).
    allfiles=[]
    resdir=loginfo.resdir
    if resdir[len(resdir)-1] != "/":
        resdir=resdir+"/"
    for fnam in os.listdir(resdir):
        if fnmatch.fnmatch(fnam, '*'+loginfo.predfpatt):
            ## we collect file with path name
            allfiles.append(resdir+fnam)
    ## os.listdir seemingly introduices modifications in the order of
    ## log files (probably using the time of creation). We can thus
    ## not expect that allfiles contains the folds obtained on the
    ## same data using different runs in the same order. This renders
    ## the calculation of statistics like McNemar impossible, as that
    ## requires pairing of samples between the compared classifiers.
    ## The solution is to sort allfiles alphabetically.
    allfiles.sort()
    ## we can now read the files and collect true targets and predictions
    allsmpno=[]
    alltrtrgs=[]
    allmeans=[]
    for fnam in allfiles:
        (smpno, trtrgs, means)=readres(fnam)
        allsmpno=allsmpno+smpno
        alltrtrgs=alltrtrgs+trtrgs
        allmeans=allmeans+means
        ## remove the file to avoid that we read it again if several
        ## simulations are stored in the same location
        os.remove(fnam)
    ## done reading we construct and return a PredStats instance
    if loginfo.targettyp=="real":
        predstats=PredStats(mode=loginfo.targettyp, ttarg=np.array(alltrtrgs), ptarg=np.array(allmeans), probs=None)
    elif loginfo.targettyp=="binary":
        probs=np.array(allmeans)
        ptarg=np.zeros(probs.shape)
        ptarg[probs>0.5]=1
        predstats=PredStats(mode=loginfo.targettyp, ttarg=np.array(alltrtrgs), ptarg=ptarg, probs=probs)
    else:
        ## 1-of-c
        probs=np.array(allmeans)
        ptarg=np.argmax(probs, axis=1)
        predstats=PredStats(mode=loginfo.targettyp, ttarg=np.array(alltrtrgs), ptarg=ptarg, probs=probs)
    return predstats

def netardstats(loginfo):
    ## function netardstats collects the ARD statisics according to
    ## the specification in loginfo. The data which we load from the
    ## respective files has two matrices with values describing
    ## "widths" (standard deviation equivalents) characterising input
    ## specific hyperparameters. The first column in the input file
    ## contains an HMC iteration counter. This is followed by n-input
    ## columns which contain widths which can be interpreted as common
    ## std. devs of Gaussian priors of all input specific to hidden
    ## MLP parameters. The second n-input number of columns have the
    ## same meaning, however for linear input to output mappings. We
    ## may thus obtain tow separate ARD characteristics per input: one
    ## guiding the importance of nonlinearity for every input and one
    ## gueiding the importance of linearity. These parameters are
    ## collected separately for every input allowing for a good
    ## characterisation of ARD which is not available from one ARD
    ## parameter allone as inputs with small widths in parameters
    ## affecting nonlinear projections may still be of linear
    ## importance. Proper ARD of an input is thus characterised by the
    ## square root of both widths squared. Note that the values
    ## returned are the averages accross all folds after discarding
    ## the first loginfo.nrburnin samples.
    ##
    ## IN
    ##
    ## loginfo: an object of type LogInfo which describes a fold run
    ##          of mlp-hmc that allows for ARD weights to be extracted
    ##          for input to hidden and input to output mappings.
    ## 
    ## OUT:
    ##
    ## (in2hard : a (<nr of in>, ) vector of average input to hidden widths
    ##  in2oard : a (<nr of in>, ) vector of average input to output widths
    ##  in2ard) : a (<nr of in>, ) vector of resulting ARD weights (square root
    ##            of the sum of squares of the above widths). 
    ##
    ## (C) P. Sykacek 2020 <peter@sykacek.net>#
    
    ## get all predicition files (there are more than one).
    allfiles=[]
    resdir=loginfo.resdir
    if resdir[len(resdir)-1] != "/":
        resdir=resdir+"/"
    for fnam in os.listdir(resdir):
        if fnmatch.fnmatch(fnam, '*'+loginfo.ardfpatt):
            ## we collect file with path name
            allfiles.append(resdir+fnam)
    ## we can now read the files and collect true targets and predictions
    allardvals=[]
    for fnam in allfiles:
        ## open file, loop through rows and start collecting values after we counted to loginfo.nrburnin
        ## use tonum(row) to get a list of values.
        with open(fnam) as infile:
            nrow=1
            for row in infile:
                if nrow > loginfo.nrburnin:
                    res=tonum(row)
                    ##print(type(res))
                    ##print(len(res))
                    allardvals.append(res)
                nrow=nrow+1
        ## remove the file to avoid that we read it again if several
        ## simulations are stored in the same location
        os.remove(fnam)
    ## convert allardvals to a numpy array
    allardvals=np.array(allardvals)
    ##print(allardvals.shape)
    ## remove the first column which contains the MCMC iteration
    ## counter and take the sample average
    avardvals=np.mean(allardvals[:, 1:], axis=0)
    nin=int(avardvals.shape[0]/2)
    ##
    ## the first nin values are the average input to hidden widths
    in2hard = avardvals[0:nin]
    ##
    ## the second nin values are the average input to output widths.
    in2oard = avardvals[nin:]
    ## ssquare root of the above widths squared as ARD values
    in2ard=np.sqrt(in2hard**2+in2oard**2)
    return (in2hard, in2oard, in2ard)

if __name__ == "__main__":
    # execute if run as a script
    if len(sys.argv)!=4:
        raise(ValueError("Call readmcres logfilename outfname resmode (resmode: 'ard' for input relevance, 'pred' or 'log2pred' for predictions)"))
    arglist=sys.argv
    logfnam=arglist[1]
    outfnam=arglist[2]
    resmode=arglist[3]
    print("####### log:{0}   out:{1}".format(logfnam, outfnam))
    ## read loginfo from logfnam
    loginfo=readhmcinfo(logfnam)
    if resmode=="ard":
        ## we extract ARD values for inputs
        (in2hard, in2oard, in2ard)=netardstats(loginfo)
        nin=in2ard.shape[0]
        header=["ard_{0}".format(ict) for ict in range(nin)]
        header=header+["nonlin_ard_{0}".format(ict) for ict in range(nin)]
        header=header+["lin_ard_{0}".format(ict) for ict in range(nin)]
        ## concatenate ardvals and cast elements to lists of length one
        allard=[[elem] for elem in in2ard.tolist()+in2hard.tolist()+in2oard.tolist()]
        ztmp=zip(header, allard)
        dtmp=dict(ztmp)
        #print(dtmp)
        arddf=pd.DataFrame(dict(zip(header, allard)))
        arddf.to_csv(outfnam, index=False)
    elif resmode=="pred":
        ## we extract predictions
        predstats=netpredstats(loginfo)
        ## predstats=PredStats(mode=loginfo.targettyp, ttarg=np.array(alltrtrgs), ptarg=ptarg, probs=probs)
        if predstats.mode=="real":
            ## regression model
            dtdict={"ttarg":predstats.ttarg.tolist(),
                    "ptarg":predstats.ptarg.tolist()}
        elif predstats.mode=="binary":
            ## binary classification
            dtdict={"ttarg":predstats.ttarg.tolist(),
                    "ptarg":predstats.ptarg.tolist(),
                    "probs":predstats.probs.tolist()}            
        else:
            ## 1-of-c classification
            dtdict={"ttarg":predstats.ttarg.tolist(),
                    "ptarg":predstats.ptarg.tolist()}
            for col in range(predstats.probs.shape[1]):
                ## iterate over columns and append probabilities for class col
                dtdict["probs{0}".format(col)]=predstats.probs[:,col].tolist()
        ## we can now convert dtdict to a dataframe and write it to a csv file.
        dtdf=pd.DataFrame(dtdict)
        if predstats.mode!="real":
            ## we convert the class labels to integers
            dtdf=dtdf.astype({"ttarg":"int32", "ptarg":"int32"})
        dtdf.to_csv(outfnam, index=False)
    elif resmode=="log2pred":
        ## new option to convert log files directly to predicitons
        ## involves use of Neals net-pred
        ## we extract predictions
        predstats=netlog2preds(loginfo)
        if predstats.mode=="real":
            ## regression model
            dtdict={"ttarg":predstats.ttarg.tolist(),
                    "ptarg":predstats.ptarg.tolist()}
        elif predstats.mode=="binary":
            ## binary classification
            dtdict={"ttarg":predstats.ttarg.tolist(),
                    "ptarg":predstats.ptarg.tolist(),
                    "probs":predstats.probs.tolist()}            
        else:
            ## 1-of-c classification
            dtdict={"ttarg":predstats.ttarg.tolist(),
                    "ptarg":predstats.ptarg.tolist()}
            for col in range(predstats.probs.shape[1]):
                ## iterate over columns and append probabilities for class col
                dtdict["probs{0}".format(col)]=predstats.probs[:,col].tolist()
        ## we can now convert dtdict to a dataframe and write it to a csv file.
        dtdf=pd.DataFrame(dtdict)
        if predstats.mode!="real":
            ## we convert the class labels to integers
            dtdf=dtdf.astype({"ttarg":"int32", "ptarg":"int32"})
        dtdf.to_csv(outfnam, index=False)
    else:
        raise(ValueError("Option {0} unknown; use ard for input relevance or pred for predicitions.".format(resmode)))
