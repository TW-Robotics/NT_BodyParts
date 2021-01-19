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
mkdir Augmented
../../Python/VE/bin/python ./createVis.py ../Classification/Augmented/ LRP
mv *.pdf Augmented/.
mv *.png Augmented/.
mkdir Augmented_GRAD
../../Python/VE/bin/python ./createVis.py ../Classification/Augmented/ GRAD
mv *.pdf Augmented_GRAD/.
mv *.png Augmented_GRAD/.
#---------------------------------------#
#--- Do visualization for raw images ---#
#---------------------------------------#
mkdir notAugmented
../../Python/VE/bin/python ./createVis.py ../Classification/notAugmented/ LRP
mv *.pdf notAugmented/.
mv *.png notAugmented/.
mkdir notAugmented_GRAD
../../Python/VE/bin/python ./createVis.py ../Classification/notAugmented/ GRAD
mv *.pdf notAugmented_GRAD/.
mv *.png notAugmented_GRAD/.
