# NT_BodyParts
This repository is used for the PlosOne publication Assessing Ethiopian Nile tilapia physiognomy with machine learning by Wilfried Wöber, Manuel Curto, Papius Tibihika, Paul Meulenbroek, Esayas Alemayehu, Lars Mehnen, Harald Meimberg and Peter Sykacek

The data analysis is fully controlable with the run.bash script in the root directory.
Usage: bash run.sh [-I] [-G] [-C] [-c] [-N] [-R]
	 I... Install framework in VE
	 G... Do GP-LVM
	 C... Apply GPC to data
	 c... Apply HMC to data
	 N... Apply CNN classification
	 R... Remove all installed and estimated files

Currently, R. Neals is not implementen.

# Software handling
Initially, the libraries must be installed. This is currently implemented using Python virtual environments. The run.bash script autonomously installs all the relevant software packages by calling *bash run.sh -I*. **Note** that we currently assuma a Cuda 9.0 installation. If you do not have this driver, the software will use the default CPU Tensorflow/Keras implementation.

# Usage of this software
You are free to use this software or parts of this software. If you use parts of our package, please cite our paper.
