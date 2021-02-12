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
# Descr.: This function installs R. Neals HMC package.      #
#-----------------------------------------------------------#
installNeal() {
    echo "Install R. Neals package"
    git clone https://gitlab.com/radfordneal/fbm.git
    cd ./fbm
    git checkout fbm.2020-01-24
    cd ..
    mv fbm ./HMC/. #Move folder in HMC folder
    #--- compile it ---#
    cd ./HMC/fbm
    ./make-all > ../hmc_install.log
    cd ../..
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
# Descr.: Get heatmaps from estimated GP-LVM                #
#-----------------------------------------------------------#
getGPLVM_Maps(){
    echo "Estimate GP-LVM features from data"
    logfile_name="./$(date '+%Y%m%d_%H%M_GPLVM_VIS.log')"
    cd ./GPLVM
    ../Python/VE/bin/python ./FeatureVariance.py >> "$logfile_name" #Estimate GP-LVM features from data 
    cd .. 
}
#-----------------------------------------------------------#
# Descr.: Do classification for GP-LVM featrues and         #
#           procrustes landmarks.                           #
#-----------------------------------------------------------#
doGPC() {
    echo "Classify data using GPC"
    cd ./GPC/
    #--------------------------------------------#
    #--- Do GPC with selected GP-LVM features ---#
    #--------------------------------------------#
    echo "Classify GP-LVM features"
    logfile_name="./GPLVM_$(date '+%Y%m%d_%H%M_GPLVM.log')"
    mkdir -p GPLVM #Create subfolder for GP-LVM classification results
    ../Python/VE/bin/python ./GPC.py ../GPLVM/BGPLVM_DATA.csv ../Data/unselectedFeatures.csv  > "$logfile_name"
    #--- Move reults ---#
    mv ./*.csv ./GPLVM/.
    mv ./*.log ./GPLVM/.
    #-------------------------------------------#
    #--- Do GPC with full 14 GP-LVM features ---#
    #-------------------------------------------#
    mkdir -p GPLVM_full #Create subfolder for GP-LVM classification results
    ../Python/VE/bin/python ./GPC.py ../GPLVM/BGPLVM_DATA.csv ../Data/FullFeatures.csv  > "$logfile_name"
    #--- Move reults ---#
    mv ./*.csv ./GPLVM_full/.
    mv ./*.log ./GPLVM_full/.
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
# Descr.: Do CNN modelling and visualization afterwards.    #
#-----------------------------------------------------------#
visCNN() {
    echo "Do Grad-CAM and LRP visualization"
    cd ./CNN/Visualization
    python CNN.py
    cd ../..
}
#-----------------------------------------------------------#
# Descr.: Do HMC classification.                            #
#-----------------------------------------------------------#
doHMC() {
    echo "Classify data using R. Neals HMC"
    cd ./HMC/Code
    source ./setvenv.sh
    #--- Now we run the code for each experiment ---#
    logfile_name="_$(date '+%Y%m%d_%H%M_HMC.log')"
    ../../Python/VE/bin/python hmc_mlp_4_tilapia.py "procrustes" &> "procrustes""$logfile_name" 
    ../../Python/VE/bin/python hmc_mlp_4_tilapia.py "gplvm_all" &> "GPAll""$logfile_name"
    ../../Python/VE/bin/python hmc_mlp_4_tilapia.py "gplvm_red" &> "GPReduced""$logfile_name"
    cd ../..
}
#------------------------------------------------------------#
# Descr: Calculates the procrustes based on the landmarks ---#
#------------------------------------------------------------#
doProcrustes() {
    #echo "Calculate Procrustes unsing R (default - uncomment if R is not installed)"
    logfile_name="./$(date '+%Y%m%d_%H%M_GPA.log')"
    cd ./Procrustes 
    #Rscript Procrustes.R &> $logfile_name
    echo "Calculate Procrustes unsing Python"
    ../Python/VE/bin/python Procrustes.py &>> $logfile_name
    cd ..
    
}
#-----------------------------------------------------------#
# Descr.:  This function prints the usage of this script.   #
#-----------------------------------------------------------#
usage() { 
	echo "Usage: bash run.sh [-I] [-G] [-g] [-P] [-C] [-c] [-N] [-n] [-R]" 1>&2 
    printf "\t I... Install framework in VE\n\t i... Install R. Neals package\n\t G... Do GP-LVM\n\t g... Estimate GP-LVM features\n\t P... Do GPA (Procrustes extraction)\n\t C... Apply GPC to data\n\t c... Apply HMC to data\n\t N... Apply CNN classification\n\t n... Estimate CNN visualization\n\t R... Remove all installed and estimated files\n"
    exit 1; 
}

#-----------------------------------------------------------#
# Descr.:  Clears up installation and data.                 #
#-----------------------------------------------------------#
removeData() {
    echo "Remove all installed and estimated files"
    #--------------------#
    #--- Installation ---#
    #--------------------#
    #--- CNN and GP-LVM ---#
    cd ./Python 
    rm -rf VE VE_CNN __pycacke__ *.log
    cd ..
    #--- Neals HMC ---#
    cd ./HMC 
    cd ./Code; rm -rf ./Data *.log
    cd ..
    rm -rf fbm *.log
    cd ..
    #-------------------#
    #--- GP-LVM data ---#
    #-------------------#
    cd ./GPLVM 
    rm -rf *.csv *.log *.npy *.pdf
    rm -rf ./Heatmaps
    cd ..
    #-------------------#
    #--- GPC results ---#
    #-------------------#
    cd ./GPC 
    rm -rf ./GPLVM
    rm -rf ./Procrustes 
    cd ..
    #-----------------#
    #--- CNN stuff ---#
    #-----------------#
    cd ./CNN/Classification 
    rm -rf ./test ./train
    rm -rf *.csv *.log
    cd ../..
    cd ./CNN/Visualization
    rm -rf ./test ./train ./AUG_* ./noAUG_*
    rm -rf *.csv *.log
    cd ../..
    #--------------------#
    #--- Vizualization --#
    #--------------------#
    cd ./Vizualizations/ResultVisualization
    rm -rf data
    rm -rf resdata
    rm *pdf
    rm *.csv
    rm *.log
    cd ../RelevanceHeatmap
    rm *.pdf
    rm *.log
    cd ../ResultVisualization
    rm -rf __pycache__
    rm -rf resdata
    rm -rf res_GPC_CNN
    rm -rf res_HMC
    rm *.pdf
    rm *.log
    cd ../..
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
#installNeal
checkInstallation   #Initially, we check if everything is installed
while getopts 'IiGgCcNnRP' OPTION
do
    case "$OPTION" in
        I)
            installVE #Install the virtual environment
            ;;
        i)
            installNeal #Install Neals Bayes inference package
            ;;
        G)
            getGPLVM #Estimate Bayesian GP-LVM features
            ;;
        g)
            getGPLVM_Maps #Estimate heatmaps from GP-LVM
            ;;
        C)
            doGPC #Do GP classification
            ;;
        c)
            doHMC #Do Neals HMC 
            ;;
        N)
            doCNN #Do CNN classification
            ;;
        n)
            visCNN #Do CNN visualization
            ;;
        R)
            removeData #Remove all estimated and installed files
            ;;
        P)
        doProcrustes #Estimate procrustes
            ;;
        *)
            usage
            ;;
    esac
done
