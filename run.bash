#TODO: 
# Installation scripts: done (2.1.2021)
# GP-LVM: done (3.1.2021)
# GPC: done (4/5.1.2021)
# Convert GPC output: done (5.1.2021)
# CNN + VIS: CNN done (6.1.2021) TODO integrate in this bash script (python ...) and vis
# Write a header to all files:
# Neals HMC
# Procrustes
# Vis scripts

# run.sh is used to control the data processing flow of the publication
# ''Assessing Ethiopian Nile Tilapia physiognomy with machine learning'' by
# Wilfried Wöber, Papius Tibihika, Paul Meulenbroek, Esayas Alemayehu, Lars Mehnen,
# Harald Meimberg and Peter Sykacek. 
# 
# To run the code, the requirements are installed into a python virtual
# environment. Afterwards, the processing pipeline described in the paper
# is run. Using the arguments, the processing steps can be activated.
#
# This code is available under a GPL v3.0 license and comes without
# any explicit or implicit warranty.
#
# (C) Wilfried Wöber 2020 <wilfried.woeber@technikum-wien.at>
#/bin/bash
#------------------------#
#--- Global functions ---#
#------------------------#
#-----------------------------------------------------------#
# Descr.: Checks if the Python virtual environment folders  #
#           exists. A flag indicates this status.           #
# Param.: -                                                 #
#-----------------------------------------------------------#
checkInstallation() {
    echo "Check system installation"
    isInstalled=0   #This flag is needed to indicate if installation is complete
    if [ -d "./Python/VE" ] && [ -d "./Python/VE_CNN" ]
    then
        isInstalled=1
    else
        echo "Installation not complete"
    fi
}
#-----------------------------------------------------------#
# Descr.: This function installs the python VE's if needed. #
# Param.: -                                                 #
#-----------------------------------------------------------#
installVE() {
    echo "Installing python virtual environment..."
    #-------------------------------------#
    #--- Checking for cuda 9.0 version ---#
    #-------------------------------------#
    if [ "$(nvcc --version | grep release | awk '{print $5}' | tr ',' ' ' | xargs)" != "9.0" ]
    then 
        echo -e "\033[0;31m YOUR CUDA VERSION IS NOT CORRECT - PLEASE USE CUDA 9.0\033[0m"
    fi
    #--------------------------#
    #--- Check installation ---#
    #--------------------------#
    if [ "$isInstalled" == "1" ]
    then
        echo -e "\033[0;32mNo need for installation - check logfile if problems occurs\033[0m"
        return
    fi
    #--- Install stuff ---#
    logfile_name="./Python/$(date '+%Y%m%d_%H%M_Install.log')"
    { 
        virtualenv --python=python3 --no-site-packages ./Python/VE
        virtualenv --no-site-packages ./Python/VE_CNN
        #--- Install ---#
        ./Python/VE_CNN/bin/pip install -r Python/requirements_CNN.txt
        ./Python/VE/bin/pip install -r Python/requirements.txt 
        ./Python/VE/bin/pip install GPy==1.9.5 
    } >> "$logfile_name"
}
#-----------------------------------------------------------#
# Descr.: Estimate the Bayesian GP-LVM.                     #
# Param.: -                                                 #
#-----------------------------------------------------------#
getGPLVM() {
    echo "Estimate GP-LVM features from data"
    #--------------------------#
    #--- Check installation ---#
    #--------------------------#
    if [ "$isInstalled" == "0" ]
    then
        echo -e "\033[0;31m Check your system installation or logfile - faulty installation found\033[0m"
        return
    fi
    #------------------------------#
    #--- Check previous results ---#
    #------------------------------#
    process=1   #Processing flag
    if [ -f "./GPLVM/optModel_bgp_features_train.csv" ] && [ -f "./GPLVM/optModel_bgp_lDim.csv" ] && [ -f "./GPLVM/optModel_bgp_model.npy" ] && \
       [ -f "./GPLVM/optModel_bgp_nrInd.csv" ] && [ -f "./GPLVM/optModel_bgp_loglik.csv" ]
    then
        echo -e "\033[0;31mFound previous results.\nPress 'r' to remove those files and calculate a new model.\nPress any other key to process with existing files \033[0m"
        read deleteResults
        #---------------------#
        #--- Delete or not ---#
        #---------------------#
        if [ "$deleteResults" == 'r' ]; then
           echo "Remove old files..." 
           rm ./GPLVM/*csv
           rm ./GPLVM/*.npy
        else 
            echo "Use previous results"
            process=0
        fi
    fi
    #-------------------------#
    #--- GP-LVM processing ---#
    #-------------------------#
    if [ "$process" == "1" ]
    then
        echo "Estimate features..."
        logfile_name="./$(date '+%Y%m%d_%H%M_GPLVM.log')"
        cd ./GPLVM/
        ../Python/VE/bin/python ./getGPLVM.py >> "$logfile_name" #Estimate GP-LVM features from data 
        cd ..
    fi
    #----------------------------#
    #--- Create training data ---#
    #----------------------------#
    paste -d, ./Data/targetID.csv ./Data/labels.csv ./GPLVM/optModel_bgp_features_train.csv > ./GPLVM/BGPLVM_DATA.csv
}
#-----------------------------------------------------------#
# Descr.: Do classification for GP-LVM featrues and         #
#           procrustes landmarks.                           #
#-----------------------------------------------------------#
doGPC() {
    echo "Classify data using GPC"
    cd ./GPC/
    #-----------------------------------#
    #--- Do GPC with GP-LVM features ---#
    #-----------------------------------#
    echo "Classify GP-LVM features"
    logfile_name="./GPLVM_$(date '+%Y%m%d_%H%M_GPLVM.log')"
    mkdir -p GPLVM #Create subfolder for GP-LVM classification results
    ../Python/VE/bin/python ./GPC.py ../GPLVM/BGPLVM_DATA.csv ../Data/unselectedFeatures.csv  > "$logfile_name"
    #--- Move reults ---#
    mv ./*.csv ./GPLVM/.
    mv ./*.log ./GPLVM/.
    #---------------------------------------#
    #--- Do GPC with procrustes features ---#
    #---------------------------------------#
    echo "Classify Procrustes features"
    logfile_name="./Procrustes_$(date '+%Y%m%d_%H%M_GPLVM.log')"
    mkdir -p Procrustes #Create subfolder for Procrustes classification results
    ../Python/VE/bin/python ./GPC.py ../Procrustes/PROCRUSTES_DATA.csv > "$logfile_name"
    #--- Move reults ---#
    mv ./*.csv ./Procrustes/.
    mv ./*.log ./Procrustes/.
    cd ..
}
#-----------------------------------------------------------#
# Descr.: Do CNN classification based on the previously     #
#           generated data files.                           #
#-----------------------------------------------------------#
doCNN() {
    echo "Do CNN classification"
    cd ./CNN/Classification 
    python CNN.py
    cd ../..
}
#-----------------------------------------------------------#
# Descr.:  This function prints the usage of this script.   #
# Param: -                                                  #
#-----------------------------------------------------------#
usage() { 
	echo "Usage: bash run.sh [-I] [-G] [-g] [-C] [-c] [-N] [-R]" 1>&2 
    printf "\t I... Install framework in VE\n\t G... Do GP-LVM\n\t g... Estimate GP-LVM features\n\t C... Apply GPC to data\n\t c... Apply HMC to data\n\t N... Apply CNN classification\n\t R... Remove all installed and estimated files\n"
    exit 1; 
}
#-----------------------#
#--- Main processing ---#
#-----------------------#
set -e      #Abort if any error occurs
#--- Check number of arguments ---#
if [ $# -lt 1 ]
then
    echo "Specify parameters"
    usage
fi
#--- Check parameters and perform given task ---#
checkInstallation   #Initially, we check if everything is installed
while getopts 'IGgCcNR' OPTION
do
    case "$OPTION" in
        I)
            installVE #Install the virtual environment
            ;;
        G)
            getGPLVM #Estimate Bayesian GP-LVM features
            ;;
        g)
            echo "Visualize GP-LVM"
            ;;
        C)
            doGPC #Do GP classification
            ;;
        c)
            echo "Classify data using R. Neals HMC"
            ;;
        N)
            doCNN #Do CNN classification
            ;;
        R)
            echo "Remove all installed and estimated files"
            ;;
        *)
            usage
            ;;
    esac
done
