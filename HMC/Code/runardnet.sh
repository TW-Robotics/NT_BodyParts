#!/bin/bash
# runardnet uses R. Neals mlp-hmc to fit a mlp with one hidden layer
# and standard settings to a problem the parameters are at large
# defaults which are taken from R. Neals DELVE ARD evaluation scripts.
#
# To aid reproducing our results, this code is provided as supplement
# to the publication "Inferring Ethiopian Nile tilapia (Oreochromis
# niloticus) morphotypes with machine learning"
# 
# The script depends on R. Neals HMC software
# (https://www.cs.toronto.edu/~radford/fbm.software.html) being
# installed and a respective path set.
# 
# Note: the python scripts called here depend on pandas and numpy
# being installed in the active python environment. To allow
# customising the python envirnonment which is used for executing the
# scripts prepdata.py and readmcres.py we prepend environment variable
# the VPYENVROOT in front of the python command which is used to
# execute the scripts.
#
# call as:
#
# runardnet.sh -h #hidden -a #ardlevel -f #folds -m #maxthreads
#              -n #hmciter -i datafnam -d simdir -l lognamebase
#              -z donormalise [Y/N] -s doreshuffle [Y/N]
#
# will run a number of folds cross testing on the data which are
# provided as datafnam. Specifications of all filds and intermediate
# results will be stored in <lognambase>0.log to
# <lognambase>#folds-1.log
#
# Hyperparameters are at large based on default settings. The only
# adjustments are:
#
# #hidden:    the number of hidden units and the
#
# #ardlevel: the ARD level which can be 0 (no ARD), 1 (ARD with per
#              subgroup weight decay hyperparameters) and 2 (ARD with
#              per parameter weight decay hyperparameters). The data
#              model is automatically adjusted to data.
#
# #folds:      determines the number of cross testing folds
#
# #maxthreads: controls the number of mlp-hmc processes we allow to
#              run in parallel (since HMC is sequential this is the
#              only way to speed things up)
#
# #hmciter:    number of iterations in Neals HMC code. Note that we
#              consider 25% thereof automatically as burnin which is
#              ignored for predictions and ARD assessment.
#
# datafnam:    is a file name of a neal hmc compatimble file. neals HMC
#              code expects files to have no header and the first D
#              columns determining the input data (regressors) while
#              the last column in the file to contain the target
#              variable. In case of "real" (continuous regression) the
#              target is a one dimensional depenent variable. In case
#              of "binary" the target is a column with classifications
#              in 0 and 1 codind and in case of "class" the last
#              column contains class 1-of-c lables in a 0..<no
#              classes>-1 coding.
#
# simdir:      directory for storing the simulation output
#
# lognambase: name specification for HMC log files, temporary HMC
#              shell scripts, predictions and ARD output.  All files
#              reside in simdir. The HMC log file for storing model
#              information and MC samples is called
#              simdir+lognamebase+"_net_<fold no>.log".
#              The HMC shell executable is called
#              simdir+lognamebase+"_<fold no>_runmc.sh".
#              The predictions output file is called
#              simdir+lognamebase+"_<fold no>_preds.txt".
#              The ARD output file name is called
#              simdir+lognamebase+"_<fold no>_inputard.txt".
#              The log file name for processing log which contains
#              information required for producing the ard assessment
#              and the prediction summary is called
#              simdir+lognamebase+"_resparse.txt".
#              The combined ARD result is stored in
#              simdir+lognamebase+"_inard.csv".
#              The combined prediction result is stored in
#              simdir+lognamebase+"_allpred.csv".
#
#              The latter 2 files are csv files with header
#              information and constructed after HMC simulations
#              completed by calling readmcres.py
#
# donormalise: [Y/N] controls whether data are normalised to have zero
#              mean and unit standard deviation.
# 
# doreshuffle: [Y/N] controls reshuffling
#
# This code is available under a GPL v3.0 license and comes without
# any explicit or implicit warranty.
#
# (C) P. Sykacek 2020 <peter@sykacek.net>


## define functions for processing
getprocscripts(){
    ## first argument to getprocscripts is the location of the scripts
    local scriptdir="$1"
    ## second argument to is getprocscripts the file pattern of the scripts
    local scriptpatt="$2"
    local res=''
    local newline=$'\n'
    for fnam in "$scriptdir/$scriptpatt"; do
	#echo $fnam
	res="$res$newline$fnam"
    done
    echo $res
}

# specify the parameters according to the above flags
# runardnet.sh -h #hidden -a #ardlevel -f #folds -m #maxthreads -n #hmciter -i datafnam -d simdir -l lognamebase -z donormalise [Y/N]
while getopts h:a:f:m:n:i:d:l:p:z:s: flag
do
    case "${flag}" in
        h) nohidden=${OPTARG};;
        a) ardlevel=${OPTARG};;
        f) nfolds=${OPTARG};;
	m) maxthreads=${OPTARG};;
	n) hmciter=${OPTARG};;
	i) infile=${OPTARG};;
	d) simdir=${OPTARG};;
	l) lognamebase=${OPTARG};;
	z) donormalise=${OPTARG};;
	s) doreshuffle=${OPTARG};;
    esac
done

echo "nohidden: -h $nohidden"
echo "ardlevel: -a $ardlevel"
echo "nfolds: -f $nfolds"
echo "maxthreads: -m $maxthreads"
echo "hmciter: -n $hmciter"
echo "infile: -i $infile"
echo "simdir: -d $simdir"
echo "lognamebase: -l $lognamebase"
echo "donormalise: -z $donormalise"
echo "doreshuffle: -s $doreshuffle"

## parameterisation is done and we have now got to prepare the hmc
## simulations this is done automatically by using prepdata.py

echo "calling: ./prepdata.py $infile $simdir $lognamebase $nohidden $ardlevel $hmciter $nfolds $donormalise $doreshuffle"

"$VPYENVROOT"python ./prepdata.py $infile $simdir $lognamebase $nohidden $ardlevel $hmciter $nfolds $donormalise $doreshuffle

## after prepdata.py we have for every fold a separate hmc script which
## needs to be executed to draw the samples from the posterior.
##
## pattern for scripts of all hmc posterior simulations to be run
runhmcpatt="*_runmc.sh"
## get the list of script files which sit in $simdir
allscripts=$(getprocscripts "$simdir" "$runhmcpatt") 

# time we wait between successive tests whether the next task should
# be launched.  time is in seconds
timewaiting=5

# timeinterleaved is used to delay the task launching. We do this to
# avoid that successive calls of the same pipeline call steps in the
# rather heterogeneous pipeline (in terms of requirements) in a
# synchronised fashion.
timeinterleaved=1

# number of currently running tasks.
tasksrunning=0

### loop which processes all shell mlp-hmc scripts in parallel
for cscript in $allscripts
do
    echo "cscript: $cscript"
    ## first we set the run permission on the script
    chmod a+x $cscript
    # We launch at maximum maxthread jobs in parallel
    echo "task running $tasksrunning"
    echo "max thread $maxthreads"
    if (("$tasksrunning" <= "$maxthreads"))
    then
	$cscript &   # we lauch the script with the hmc simulation in the background
	sleep $timeinterleaved # to desynchronise identical requirements we sleep now to let the launched task get a head start
    fi
    # the number of my running tasks is the number of my child
    # processes Note: `pgrep -P $$ | wc -w` gives one more
    # (presumably the current bash job as additional task)
    tasksrunning=`pgrep -P $$ | wc -w` 
    # waits until we have free capycity this "semaphore" polls the
    # variables. We should thus set timewaiting to a sufficiently
    # large number of seconds (e.g. 30)
    while(("$tasksrunning" > "$maxthreads"))
    do
	echo "running: " $tasksrunning
	sleep $timewaiting
	tasksrunning=`pgrep -P $$ | wc -w`
    done
done
## wait for completion of all jobs
wait
## clean up processing scripts
for cscript in $allscripts
do
    rm $cscript
done

## we can now call the processing of results readmcres logfilename
## outfname resmode (set resmode to ard for input relevance or pred fo
## predictions)
echo "$simdir$lognamebase""_resparse.txt"
logfilenam="$simdir$lognamebase""_resparse.txt"
echo "$simdir$lognamebase""_allardres.csv"
outard="$simdir$lognamebase""_allardres.csv"
echo "$simdir$lognamebase""_allpredres.csv"
outpred="$simdir$lognamebase""_allpredres.csv"
## call readmcres
echo "collecting predictions: ./readmcres.py $logfilenam $outpred pred"
"$VPYENVROOT"python ./readmcres.py $logfilenam $outpred pred
echo "true targets, predictions [and probabilities] found in file $outpred"
echo "collecting ARD information: ./readmcres.py $logfilenam $outard ard"
"$VPYENVROOT"python ./readmcres.py $logfilenam $outard ard
echo "input ARD found in file $outard"
