# NT_BodyParts
This repository is used for the PlosOne publication Assessing Ethiopian Nile tilapia physiognomy with machine learning by Wilfried Wöber, Manuel Curto, Papius Tibihika, Paul Meulenbroek, Esayas Alemayehu, Lars Mehnen, Harald Meimberg and Peter Sykacek

The data analysis is fully controlable with the run.bash script in the root directory.

Usage: bash run.sh [-I] [-G] [-g] [-C] [-c] [-N] [-n] [-R]"
   
    I... Install framework in VE
    
    i... Install R. Neals package
    
    G... Do GP-LVM
    
    g... Estimate GP-LVM features
    
    C... Apply GPC to data
    
    c... Apply HMC to data
    
    N... Apply CNN classification
    
    n... Estimate CNN visualization
    
    R... Remove all installed and estimated files

Currently, R. Neals is not implementen.

# OS information and must haves
This package was tested unter Ubuntu 16.04 with a Python2 and 3 isntallations as well as CUDA 9.0 drivers. The main processing hardware was a NVIDIA 1080Ti and a Intel i9-7900X CPU. If you do not have the appropriate hardware/software, you may have to adapt the scripts to be fully functional on your system.

# Software handling
Initially, the libraries must be installed. This is currently implemented using Python virtual environments. The run.bash script autonomously installs all the relevant software packages by calling *bash run.sh -I -i*. 

**Note** that we currently assuma a Cuda 9.0 installation. If you do not have this driver, the software will use the default CPU Tensorflow/Keras implementation.

# Handling of this package
Before you start using this package note, that the all scripts may take a long time (>10h on our system) to finish. If you do not want to estimate the GP-LVM by yourself, you can initially copy the DUMMYBGPLVM_DATA.csv file into the GPLVM folder. Afterwards, you can do the remaining functionalities such as CNN, GPC, etc. 

Initially you have to install all packages using the *-I* (for CNN and GPy) and *-i* flag (for HMC). 
Afterwards, the installation the GP-LVM must be applied using the *-G* flag. **You cannot use the CNN/HMC before the BGPLVM_DATA.csv file was created at the end of the GP-LVM.** Finally, you can use the *-C* flag for GPC, the *-N* flag for CNN classification and *-c* flag for HMC. 
**If you want to do CNN or MHC based classification without GP-LVM, you can use our DUMMYBGPLVM_DATA.csv file in the data folder.*

For visualization, the *-g* flag for GP-LVM and *-n* flag for the CNN can be used. The heatmaps are stored in the model folders.

**Note:** all functions in the run script generates logfiles. The script output (CNN, GPC, etc.) are stored in those logfiles.
**Note:** You can remove **anything** (installation as well as estimated files) with the *-R* flag.

# Usage of this software
You are free to use this software (see licence) or parts of this software. If you use parts of our package, please cite our paper.
