# This script runs all vizualization scripts for the presented paper.
# NOTE: The data is based on random initializations and statistical
# methodologies. The results will differ.
#
# This code is available under a GPL v3.0 license and comes without
# any explicit or implicit warranty.
#
# (C) Wilfried Wöber 2020 <wilfried.woeber@technikum-wien.at>
#!/bin/bash
#--- Create Pies ---#
echo "Create pie charts..."
cd ./PieVisualization 
../../Python/VE/bin/python 1_prepareData.py &> pieLog.log  #Prepare data from GPC and CNN
bash 1b_prepareHMC.bash &>> pieLog.log                  #Prepare data from HMC
../../Python/VE/bin/python convertHMCRelevance.py &>> pieLog.log   #Convert array format
../../Python/VE/bin/python 2_relevancePies.py &>> pieLog.log       #Create pie plots
cd ..
#--- Create relevance maps ---#
echo "Create relevance maps..."
cd RelevanceHeatmap 
../../Python/VE/bin/python RelevanceHeatmaps.py &> HeatmapLog.log
cd ..
#--- Create model comparisons ---#
echo "Create model comparison plots..."
cd ResultVisualization
../../Python/VE/bin/python 1_GPCtoHMC.py &> visLog.log
../../Python/VE/bin/python 2_GraphGeneration.py &>> visLog.log
