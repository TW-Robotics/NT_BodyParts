## evalres.py contains definitions and code which are used to evaluate
## metrics of classification performance.  It is imported by the
## Jupyter notebook eval_hmc_mlp_4_tilapia.ipynb which generates the
## HMC-MLP metrics for evaluating Nile tilapia classification of six
## Ethiopian lakes. To aid reproducing our results, this code is provided as
## supplement to the publication "Inferring Ethiopian Nile tilapia
## (Oreochromis niloticus) morphotypes with machine learning"
##
## For mcnemar see Github:
##
## https://gist.github.com/kylebgorman/c8b3fb31c1552ecbaafb
##
## We compute McNemar's test using the "mid-p" variant suggested by:
##    
## M.W. Fagerland, S. Lydersen, P. Laake. 2013. The McNemar test for 
## binary matched-pairs data: Mid-p and asymptotic are better than exact 
## conditional. BMC Medical Research Methodology 13: 91.
##    
## The code in evalres.py is available under a GPL v3.0 license and
## comes without any explicit or implicit warranty.
##
##
## (C) P. Sykacek 2020 <peter@sykacek.net> 


## load libraries for data handling. 
import numpy as np
import pandas as pd
import copy 
resdir="./Data/resdata/" ## location of results files

## we have patterns for the different input data 
## 1) procrustes landmarks
## 2) 14 top ranked GP-LVM dimensions
## 3) 14 selected GP-LVM dimensions which 
##    have no noticeable technical variation 
##    (to avoid clever Hans effects)
##
## For every input modality we resample 
## a) 10 times by reshuffling
## b) 10 times by resampling with replacement (bootstrap)
##
## This results in 60 result files which we get by HMC-MLP 
## based classification and 6 different file name patterns. 
## Every pattern occurs 10 times with indices 0-9. 
## Patterns for ARD files
allardfnam={
    "netprcshfl":"rshfl_prc_it{0}__allardres.csv",             # reshuffled procrustes mlp hmc
    "netprcboot":"rsmp_prc_it{0}__allardres.csv",              # bootstrapped procrustes mlp hmc
    "netgptopshfl":"rshfl_gpall_it{0}__allardres.csv",         # 14 top gplvm reshuffled mlp hmc
    "netgptopboot":"rsmp_gpall_it{0}__allardres.csv",          # 14 top gplvm bootstrapped mlp hmc
    "netgpselshfl":"rshfl_gpred_it{0}__allardres.csv",         # 14 selected gplvm reshuffled mlp hmc
    "netgpselboot":"rsmp_gpred_it{0}__allardres.csv",          # 14 selected gplvm bootstrapped mlp hmc
    "gpcgpselshfl":"rshfl_gpred_gpc_it{0}_allardres.csv",      # 14 selected gplvm reshuffled gpc
    #"gpcgpselboot":"rsmp_gpred_gpc_it{0}_allardres.csv",       # 14 selected gplvm bootstrapped gpc
    "gpcgptopshfl":"rshfl_gpall_gpc_it{0}_allardres.csv",      # 14 top gplvm reshuffled gpc
    #"gpcgptopboot":"rsmp_gpall_gpc_it{0}_allardres.csv",       # 14 top gplvm bootstrapped gpc
    "gpcprcshfl": "rshfl_prc_gpc_it{0}_allardres.csv",         # reshuffled procrustes gpc
    #"gpcprcboot": "rsmp_prc_gpc_it{0}_allardres.csv",          # bootstrapped procrustes gpc
    "cnnshfl":"",                                              # reshuffled cnn (convolutional deep no ARD)
    #"cnnboot":""                                               # bootstrapped cnn (convolutional deep no ARD)
}
## Patterns for probabilities true labels and predictions
allpredfnam={
    #"netprcshfl":"rshfl_prc_it{0}__allpreds.csv",      # reshuffled procrustes mlp hmc
    #"netprcboot":"rsmp_prc_it{0}__allpreds.csv",       # bootstrapped procrustes mlp hmc
    #"netgptopshfl":"rshfl_gpall_it{0}__allpreds.csv",  # 14 top gplvm reshuffled mlp hmc
    "netprcshfl":"rshfl_prc_it{0}__allpredres.csv",      # reshuffled procrustes mlp hmc
    "netprcboot":"rsmp_prc_it{0}__allpredres.csv",       # bootstrapped procrustes mlp hmc
	"netgptopshfl":"rshfl_gpall_it{0}__allpredres.csv",  # 14 top gplvm reshuffled mlp hmc

    "netgptopboot":"rsmp_gpall_it{0}__allpredres.csv",   # 14 top gplvm bootstrapped mlp hmc
    "netgpselshfl":"rshfl_gpred_it{0}__allpredres.csv",  # 14 selected gplvm reshuffled mlp hmc 
    "netgpselboot":"rsmp_gpred_it{0}__allpredres.csv",   # 14 selected gplvm bootstrapped mlp hmc
    "gpcgpselshfl":"rshfl_gpred_gpc_it{0}_allpreds.csv",           # 14 selected golvm reshuffled gpc
    #"gpcgpselboot":"rsmp_gpred_gpc_it{0}_allpreds.csv",            # 14 selected gplvm bootstrapped gpc
    "gpcgptopshfl":"rshfl_gpall_gpc_it{0}_allpreds.csv",           # 14 top golvm reshuffled gpc
    #"gpcgptopboot":"rsmp_gpall_gpc_it{0}_allpreds.csv",            # 14 top gplvm bootstrapped gpc
    "gpcprcshfl":"rshfl_prc_gpc_it{0}_allpreds.csv",               # reshuffled procrustes gpc
    #"gpcprcboot":"rsmp_prc_gpc_it{0}_allpreds.csv",                # bootstrapped procrustes gpc
    "cnnshfl":"rshfl_cnn_it{0}_allpreds.csv",                      # reshuffled cnn (convolutional deep learning)
    #"cnnboot":"rsmp_cnn_it{0}_allpreds.csv"                        # bootstrapped cnn (convolutional deep learning)
}
## legends for plots and tables
legs2features={
    "netprcshfl":"GPA",
    "netprcboot":"GPA",
    "netgptopshfl":"Top GP-LVM",
    "netgptopboot":"Top GP-LVM",
    "netgpselshfl":"Sel GP-LVM",
    "netgpselboot":"Sel GP-LVM",
    "gpcgpselshfl":"Sel GP-LVM",        
    #"gpcgpselboot":"Sel GP-LVM",  
    "gpcgptopshfl":"Top GP-LVM", 
    #"gpcgptopboot":"Top GP-LVM",  
    "gpcprcshfl":"GPA",     
    #"gpcprcboot":"GPA",      
    "cnnshfl":"Deep CNV",            
    #"cnnboot":"Deep CNV"              
}
## map keys to sampling methods
legs2sampling={
    "netprcshfl":"Reshuffled",
    "netprcboot":"Bootstrapped",
    "netgptopshfl":"Reshuffled",
    "netgptopboot":"Bootstrapped",
    "netgpselshfl":"Reshuffled",
    "netgpselboot":"Bootstrapped",
    "gpcgpselshfl":"Reshuffled",        
    #"gpcgpselboot":"Bootstrapped",  
    "gpcgptopshfl":"Reshuffled", 
    #"gpcgptopboot":"Bootstrapped",  
    "gpcprcshfl":"Reshuffled",     
    #"gpcprcboot":"Bootstrapped",      
    "cnnshfl":"Reshuffled",            
    #"cnnboot":"Bootstrapped"              
}
## map keys to classification method
legs2method={
    "netprcshfl":"HMC-MLP",
    "netprcboot":"HMC-MLP",
    "netgptopshfl":"HMC-MLP",
    "netgptopboot":"HMC-MLP",
    "netgpselshfl":"HMC-MLP",
    "netgpselboot":"HMC-MLP",
    "gpcgpselshfl":"GPC",        
    #"gpcgpselboot":"GPC",
    "gpcgptopshfl":"GPC",
    #"gpcgptopboot":"GPC",
    "gpcprcshfl":"GPC",
    #"gpcprcboot":"GPC",
    "cnnshfl":"CNN",
    #"cnnboot":"CNN"
}
infomap={"ardfnam":allardfnam, "predfnam":allpredfnam,
         "inputtype":legs2features, "resampling":legs2sampling,
         "classifier":legs2method}

def info2df(infomap):
    ## info2df recodes infomap as dataframe with dict keys as column
    ## "acronym" also used as index. Otherwise infomap keys are the
    ## column names of the resulting dataframe and the values in every
    ## row constitute a description of all settings which apply to the
    ## respective simulation.
    ##
    ## IN
    ##
    ## infomap: a dictionary which allows bluiding a description of
    ##          the experiment.
    ##
    ## OUT
    ##
    ## infodf:  experimental atrributes as dataframe with rows indexable by
    ##          the dict keys and columns representing algorithmic settings.
    ##
    ## (C) P. Sykacek 2020
    infodf={"acronym":[]}
    for infokey, valdict in infomap.items():
        ## generate an emty list entry in infodf
        infodf[infokey]=[]
        if len(infodf["acronym"])==0:
            ## we have to initialise infodf
            for valkey, value in valdict.items():
                infodf["acronym"].append(valkey)
                infodf[infokey].append(value)
        else:
            ## we have initialised acronyms and use their order to build the data entry
            for acronym in infodf["acronym"]:
                infodf[infokey].append(valdict[acronym])
    ## the result is a dict of lists with appropriately ordered entries
    infodf=pd.DataFrame(infodf)
    ## set acronym as index:
    infodf.set_index("acronym", drop=False, inplace=True)
    return infodf
    
def lbl2oneofc(targets, noclass=None):
    # converts labels to a 1-of-c target coding
    # IN
    # targets: zero based label vector with numerical lables
    # noclass: nr of classes (None if determined by targets)
    #
    # OUT
    # oneofc: one of c coded representation (
    #         column k of row n is 1 if targets[n] is k)
    # (C) P. Sykacek 2017 <peter@sykacek.net>

    if type(targets) != type(np.array([])):
        targets=np.array(targets)
    targets=targets-min(targets)
    
    if noclass is None:
        noclass=0
        
    nrsamples=targets.shape[0]
    noclass = max( np.max(targets)+1, noclass)  
    targs=np.zeros((nrsamples, noclass))
    for i in range(noclass):
        # take the row indices from nonzero 
        idone=np.array([list(np.nonzero(targets==i)[0])])
        # idone=idone+nrsamples*i;
        #print(i, idone)
        if np.prod(idone.shape) > 0:
            targs[idone, i]=np.ones((idone.shape[0], 1))
    return targs

## read HMC-MLP prediction result
def readhmc_classpreds(fname):
    ## loads the prediction result of a HMC-MLP simulation
    ## with the runardnet.sh classification toolchain
    ## 
    ## IN
    ## fname: file name of a csv file which contains the resul
    ##        The assumption is that the format is in line with
    ##        runardnet.sh output.
    ##
    ## OUT
    ## res:      dictionary with keys
    ## 'ttarg':  list with true targets
    ## 'ptorg':  list with predicted targets
    ## 'Post':   a [N x nr-of-class] maxtrix with posterior probabilites
    ## 'Pri':    a [N x nr-of-class] maxtrix with prior probabilites
    ##
    ## (C) P. Sykacek 2020 <peter@sykacek.net>
    data=pd.read_csv(fname, sep=',')
    res=dict()
    res['ttarg']=data['ttarg'].tolist()
    res['ptarg']=data['ptarg'].tolist()
    ## all columns from 2 to the last contain the probabilities
    Post=data.iloc[:,2:]#.to_numpy()
    ## in case Post has only one column, it represents Probabilities
    ## for class '1' and class '0' has probability 1-P.
    twoclass=Post.shape[1]==1
    nsamples=Post.shape[0]
    if twoclass:
        Post=np.concatenate((1-Post, Post), axis=1)
        Pri=np.mean(res['ttarg'])
        Pri=np.array([[1-Pri]*nsamples,[Pri]*nsamples]).T
    else:
        ## multi class problem requires getting the orior probabilities
        Pri=np.array([np.mean(lbl2oneofc(res['ttarg']), axis=0).tolist()]*nsamples)
    res["Post"]=Post
    res["Pri"]=Pri
    return res

def loadallres(basedir, infodf , idsperpatt):
    ## loadallres loads all randomised results for all provided patterns.
    ##
    ## IN
    ##
    ## basedir :    location of files
    ## infodf  :    a dataframe which maps experimental accronyms to file name patterns
    ## idsperpatt:  a list with entries to be used to convert a pattern to filenames.
    ##
    ## OUT
    ##
    ## resdict: a dict which maps the acronyms in infodf to lists of res dicts (see readhmc_classpreds)
    ##
    ## (C) P. Sykacek 2020
    resdict=dict()
    for key, row in infodf.iterrows():
        fnampatt=row["predfnam"]
        ## key is used in resdict to store all randomisations of a certain kind as list of res dicts
        resperkey=[]
        for cid in idsperpatt:
            resperkey.append(readhmc_classpreds(basedir+fnampatt.format(cid)))
        resdict[key]=resperkey
    return resdict

def res2cnftab(resdict, lab2leg={0:"Chamo", 1:"Hawassa", 2:"Koka",3:"Langano", 4:"Tana", 5:"Ziway"}):
    ## res2comftab converts a results dict resdict as returned by
    ## loadallres to confusion tables.
    ## IN
    ## resdict: a results dictionary as returned by loadallres.
    ##          keys are classification accronyms and values are
    ##          dictionaries with keys
    ## 'ttarg':  list with true targets
    ## 'ptarg':  list with predicted targets
    ## 'Post':   a [N x nr-of-class] maxtrix with posterior probabilites
    ## 'Pri':    a [N x nr-of-class] maxtrix with prior probabilites
    ##
    ## lab2leg is a dictionary which maps ttarg and ptarg values to lable strings.
    ##
    ## OUT
    ##
    ## conftabdf: a dataframe which codes all confusion tables.
    ##
    ## (C) P. Sykacek 2020 <peter@sykacek.net>

    confdict={'clssmthd':[],'truelab':[]}
    for key, val in lab2leg.items():
        confdict[val]=[]
    for clssmthd, allpreds in resdict.items():
        for preds in allpreds:
            ## iterate over all results (reshuffling and or bootstrap)
            ttarg= np.array(preds['ttarg'])
            ptarg= np.array(preds['ptarg'])
            unqtrg=list(set(ttarg))
            for tt in unqtrg:
                ## iterate over true targets
                confdict['clssmthd'].append(clssmthd)     ## store classification method
                confdict['truelab'].append(lab2leg[tt])   ## and true label
                for pt in unqtrg:
                    ## and now over predicted targets where we count the
                    ## number of classifications of ttarg==tt and
                    ## pt==prtarg
                    nohits=np.sum(np.logical_and(ttarg==tt, ptarg==pt))
                    ## we append that to confdict[lab2leg[pt]]
                    confdict[lab2leg[pt]].append(nohits)
    ## done and we can return confdict as pandas dataframe
    return pd.DataFrame(confdict)


# some definitions:
# mcnemar is from Github:
# https://gist.github.com/kylebgorman/c8b3fb31c1552ecbaafb
from scipy.stats import binom

def mcnemar(b, c):
    """
    Compute McNemar's test using the "mid-p" variant suggested by:
    
    M.W. Fagerland, S. Lydersen, P. Laake. 2013. The McNemar test for 
    binary matched-pairs data: Mid-p and asymptotic are better than exact 
    conditional. BMC Medical Research Methodology 13: 91.
    
    `b` is the number of observations correctly labeled by the first---but 
    not the second---system; `c` is the number of observations correctly 
    labeled by the second---but not the first---system.
    """

    n = b + c
    x = min(b, c)
    dist = binom(n, .5)
    p = 2. * dist.cdf(x)
    midp = p - dist.pmf(x)
    return midp

def lab2cnt(y_1, y_2, t):
    # lab2cnt converts two sets of predicted labels and known truth
    # to McNemars counts
    if type(y_1)!=type(np.array([])):
        y_1=np.array(y_1)
    if type(y_2)!=type(np.array([])):
        y_2=np.array(y_2)
    if type(t)!=type(np.array([])):
        t=np.array(t)
    na=np.sum(np.logical_and((y_1==t), (y_2 !=t)))
    nb=np.sum(np.logical_and((y_1!=t), (y_2 ==t)))
    return (na, nb)
    
## calculate mutual information
def mutinf(post, pri, logfunc=np.log2):
    ## IN
    ## post: a [nsample x nclass] matrix with posterior probabilities
    ## pri:  a [nsample x nclass] matrix with prior probabilities
    ## OUT
    ## minf: Sample estimate of mutual information 
    ##       \int_{x,y} P(x) p(y|x) (log(p(y|x))-log(p(y)))
    allz=post==0
    post[allz]=1
    lpo=logfunc(post)
    post[allz]=0 ## irrelevant but still done
    lpri=logfunc(pri)
    return(np.mean(np.sum(post*(lpo-lpri), axis=1)))

def resvsdefpred(resdict, maxsig=0.999, lab2leg={0:"Chamo", 1:"Hawassa", 2:"Koka",3:"Langano", 4:"Tana", 5:"Ziway"}):
    ## resvsdefpred compares all resulting predictions against default
    ## predictions which arrise when classifying from prior
    ## probabilites. Note that we do not have to evaluate MI as this
    ## is zero. We will however have to calculate Acc and Sig.
    allkeys=[]
    allacc=[]
    allsig=[]
    first=True
    ## first we calculate all generalisation acuracies and mutual informations
    for key in resdict.keys():
        results=resdict[key]
        keyaccs=[]
        keysig=[]
        for cres in results:
            if first:
                ## we obtain the default prediction whihc predicts the
                ## majority label.
                alltrue=np.array(cres["ttarg"])
                unqlab=list(set(list(alltrue)))
                unqlab.sort()
                allcnts=[]
                for lab in unqlab:
                    allcnts.append(np.sum(alltrue==lab))
                maxlabel=unqlab[allcnts.index(np.amax(allcnts))]
                for idx in range(len(unqlab)):
                    print("Lake: {0} sample nr.: {1}".format(lab2leg[unqlab[idx]], allcnts[idx]))
                print("Maxlabel: {0} Cnts: {1}".format(lab2leg[maxlabel], np.amax(allcnts)))
                defpreds=np.array([maxlabel]*len(alltrue))
                allkeys.append("Def")
                allacc.append(np.mean(defpreds==alltrue))
                allsig.append(maxsig)
                first=False
            ## we can now assess all predictions against the default
            ## label and calculate significances.
            keyaccs.append(np.mean(np.array(cres["ttarg"])==np.array(cres["ptarg"])))
            na, nb=lab2cnt(defpreds, cres["ptarg"], cres["ttarg"])
            keysig.append(np.minimum(mcnemar(na, nb), maxsig))
        ## store keys and average performance values
        allkeys.append(key)
        allacc.append(np.mean(keyaccs))
        allsig.append(np.mean(keysig))
    ## we may now convert everything to a dataframe.
    return pd.DataFrame({"key":allkeys, "Acc":allacc, "Sig":allsig})

def res2metric(resdict, maxsig=0.999):
    ## res2metric takes a resdict dictionary as generated by
    ## loadallres and produces a dataframe with column "key"
    ## containing resdict keys and columns "Acc" containing
    ## generalisation accuracies, "MI" containing mutual information
    ## and "Sig" containing Mc Nemar p-values against the least
    ## performing classification result among resdict keys.
    ##
    ## IN
    ##
    ## resduct: a dict which maps the acdronyms in infodf to lists of
    ##          res dicts (see readhmc_classpreds)
    ##
    ## maxsig:  maximal significance value (we truncate below)
    ##
    ## OUT
    ##
    ## metricdf: a dataframe with columns "key", "Acc", "MI" and "Sig".
    ##
    ## (C) P. Sykacek 2020
    allkeys=[]
    allacc=[]
    allmi=[]
    allna=[]
    allnb=[]
    allsig=[]
    avgenaccs=[]
    genacckeys=[]
    ## first we calculate all generalisation acuracies and mutual informations
    for key in resdict.keys():
        results=resdict[key]
        ## results is an ordered list of res dictionaries each
        ## representng one run under compatible conditions.
        cgenaccs=[]
        for cres in results:
            ## store the key as indication for the experiment
            allkeys.append(key)
            ## calculate and store generalisation accuracies for cres
            genacc=np.mean(np.array(cres["ttarg"])==np.array(cres["ptarg"]))
            allacc.append(genacc)
            cgenaccs.append(genacc)
            ## calculate and store mutual information
            allmi.append(mutinf(cres["Post"], cres["Pri"]))
        ## we can now calculate and store the average generalisation accuracies and the key.
        avgenaccs.append(np.mean(cgenaccs))
        genacckeys.append(key)
    ## we are now done with step 1 and can deduce the index of the
    ## smallest anverage generalisation accuracy which we use to
    ## determine the worst performing method which will be our base
    ## result which we compare against in te McNemar test.
    basekey=genacckeys[np.argmin(avgenaccs)]
    ## we iterate now again to calculate the McNemar p-values.
    baseres=resdict[basekey]
    for key in resdict.keys():
        ##print("{0} {1}".format(basekey, key))
        if key == basekey:
            ##print("basekey!")
            ## we are self vs. self and augment allsig with an
            ## appropriate number of p-values=1.0
            #print(len(resdict[key]))
            allsig=allsig+[maxsig]*len(resdict[key])
            allna=allna+[0.0]*len(resdict[key])
            allnb=allnb+[0.0]*len(resdict[key])
        else:
            ## we have to run len(resdict[key]) comparisons:
            for idx in range(len(resdict[key])):
                ## extract base and the respective competitor
                cbres=baseres[idx]
                ccres=resdict[key][idx]
                #print("diff: {0}".format(np.sum(np.array(ccres["ttarg"])!=np.array(cbres["ttarg"]))))
                ## obtain the two counts for McNemar:
                na, nb=lab2cnt(cbres["ptarg"], ccres["ptarg"], ccres["ttarg"])
                #print("na: {0} nb: {1}".format(na, nb))
                allna.append(na)
                allnb.append(nb)
                ## calculate and store the p-value of the appropriate
                ## binomial against a two sided alternative.
                pval=np.minimum(mcnemar(na, nb), maxsig)
                ##print("key:{0} p-val:{1}".format(key, pval))
                allsig.append(pval)
    ## we are done and may construct and return the dataframe metricdf
    return pd.DataFrame({"key":allkeys, "Acc":allacc, "MI":allmi, "Sig":allsig, "na":allna, "nb":allnb})

def info_sel2keys(infodf, sel):
    ## selects infodf["accronym"] values in rows which match with sel.
    ## sel is a dict with keys denoting infodf column names and values
    ## correspond to row selectors. The values are or seleced, while
    ## dict keys are and selected.
    ##
    ## IN
    ##
    ## infodf: description of experimental conditions
    ##
    ## sel:    a dict with keys representing infodf columns and values
    ##          vanlues in these columns
    ##
    ## OUT
    ##
    ## keys:   the values in the accronym column of the selection as list.
    ##
    ## (C) P. Sykacek 2020
    first = True
    nr_rows=len(infodf.index)
    for key, vals in sel.items():
        ## csel is a boolean row selector to infodf.
        csel=np.array([False]*nr_rows)
        ## force list type
        if type(vals) != type([]):
            vals=[vals]
        ## extract data from infodf
        ccol=infodf[key]#.to_numpy()
        for val in vals:
            csel=np.logical_or(csel, ccol==val)
        if first:
            first=False
            rwsel=csel
        else:
            rwsel=np.logical_and(rwsel, csel)
    ## use rwsel to extract and return the accronyms as list
    keys=infodf["acronym"]
    keys=keys.iloc[rwsel]
    return keys.tolist()

## test:
#sel1={'resampling':'Bootstrapped', 'classifier':'HMC-MLP', 'inputtype':'GPA'}
#print(info_sel2keys(infodf, sel1))

def resdict2metrics(resdict, infodf, selinfo={}):
    ## resdict2metrics converts a resdict structure to assessment metrics.
    ##
    ## Metrics include
    ##
    ## 1) generalisation accuracies per resdict key (a list with one entry
    ##    per resampling result)
    ##
    ## 2) Mutual information values per resdict key  (a list with one entry
    ##    per resampling result)
    ##
    ## 3) McNemars p-values of significant differences in classificarion
    ##    performance using the classifier with smallest average
    ##    generalisation accuracy as baseline method.
    ##
    ## To obtain meaningful results for 3), calculations are only done within
    ## a particular randomisation strategy. This is required as sample order
    ## is oly kept within one iteration of a particular sampling strategy but
    ## differs between reshufling and bootstrap. For that reason, we also
    ## provide unique links between resdict keys and the sampling method.
    ##
    ## IN
    ##
    ## resdict: a dict which maps the keys in allpatterns to lists of
    ##          res dicts (see readhmc_classpreds)
    ##
    ## infodf: a dataframe which maps the keys in resdict to a methods description.
    ##
    ## selinfo: a dict of dicts with outer keys denoting experimental
    ##          conditions and inner keys denoting infodf columns and
    ##          values which denote methodological restrictions we
    ##          wish to analyse against each other in the performance
    ##          evaluation. Every list entry is considered
    ##          individually and selects a group of methods which are
    ##          analysed in combination.
    ##
    ## OUT
    ##
    ## resdf: A data frame with columns "expgroup" containing selinfo
    ##          keys, "key" containing the resdict keys, all infodf
    ##          columns, "Acc" containing generalisation accuracies
    ##          "MI" containing mutual information and "Sig"
    ##          containing Mc Nemar p-values against the least
    ##          performing classification result in "expgroup" (the
    ##          correspinding p-value is 1.0).
    ##
    ## (C) P. Sykacek 2020

    ## we start by calculating the performance metrics 
    if selinfo != {}:
        ## we have to do iterative calculations
        first=True
        for key, val in selinfo.items():
            ## first extract the acronyms (keys) to resdict which we
            ## analyse together in a call of res2metric.
            curreskeys=info_sel2keys(infodf, val)
            crdict={crk:resdict[crk] for crk in curreskeys}
            crdf=res2metric(crdict)
            ## add the experiment group
            crdf["expgroup"]=[key]*len(crdf.index)
            if first:
                resdf=crdf
                first=False
            else:
                resdf=pd.concat([resdf, crdf], ignore_index=True)
    else:
        ## there is only one set of values which may all be compared
        resdf=res2metric(resdict)
        resdf["expgroup"]=["group_1"]*len(resdf.index)
    ## we have now got resdf and combine with infodf for adding the
    ## additional information. To do so we set resdf["key"] as index
    resdf.set_index("key", drop=True, inplace=True)
    ## and use an inner join
    resdf=pd.concat([resdf, infodf], axis=1, join='inner')
    return resdf
    

def logit(pvals, myeps=10**-100):
    ## logit transform of p-values to "unfold" the underlying metric
    ## 
    ## convert to numpy array
    if type(pvals) != type(np.array([])):
        if type(pvals) == type([]):
            pvals=np.array(pvals)
        else:
            pvals=np.array([pvals])
    ## make sure the value is > 0
    onemp=1-pvals
    pvals[pvals<myeps]=myeps
    onemp[onemp<myeps]=myeps
    ## return logit transformed p-values.
    return np.log(pvals)-np.log(onemp)

