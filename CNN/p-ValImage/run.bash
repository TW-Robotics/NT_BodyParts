# coding: utf-8
# This script controls the p-value image generation.
# Note, that this script is analyses one iteration of
# the 10-fold cross validation.
#
# This code is available under a GPL v3.0 license and comes without
# any explicit or implicit warranty.
#
# (C) Wilfried Woeber 2020 <wilfried.woeber@technikum-wien.at>
#!/bin/bash
#---------------------------------------------#
#--- Do visualization for augmented images ---#
#---------------------------------------------#
#mkdir Augmented
#../../Python/VE/bin/python ./createVis.py ../Classification/Augmented/
#mv *.pdf Augmented/.
#mv *.png Augmented/.
#---------------------------------------#
#--- Do visualization for raw images ---#
#---------------------------------------#
mkdir notAugmented
../../Python/VE/bin/python ./createVis.py ../Classification/notAugmented/
mv *.pdf notAugmented/.
mv *.png notAugmented/.
