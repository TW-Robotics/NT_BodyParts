import sys                  #System stuff
import numpy as np          #You should know that
import matplotlib.pyplot as plt #We aim to plot something...
import os                   #For bash commands
def plotHeatmaps(relevance,Classifier,printpostfix):
    invSelection	= "../../Data/unselectedFeatures.csv"
    Selection           = np.loadtxt(invSelection,   delimiter=",",dtype=int)   #selection of features
    #relevance       = "../PieVisualization/GPCData/ranking_BGPLVM_noResampling_manual.csv"#  "../../../20200923_Figures_Addon/Pies_GPC/ranking_BGPLVM_noResampling_manual.csv"
    RelevanceRanking    = np.loadtxt(relevance,   delimiter=" ",dtype=float)    #Estimated relevance rank
    #---------------------#
    #--- Load heatmaps ---#
    #---------------------#
    varData_filtered = np.zeros((len(Selection),(224*224))) #Create memory for variance images
    for i in range(0,len(Selection)):
        varImg = np.loadtxt("../../GPLVM/Heatmaps/"+str(Selection[i])+".csv",   delimiter=",",dtype=float)
        varData_filtered[i,:]=varImg.flatten()
    for RANK in range(0,RelevanceRanking.shape[0]): #Looper over all ranks
        rel_rank        = RelevanceRanking[RANK,:]/np.sum(RelevanceRanking[RANK,:])     #Get relevance weights
        rel_img         = np.dot(rel_rank.reshape((1,14)),varData_filtered)             #Get segmentation based on empirical probability
        #------------------------#
        #--- Do visualization ---#
        #------------------------#
        plt.figure()
        MEAN_Fish_VAR=np.reshape(rel_img, (224,224))
        plt.imshow(MEAN_Fish_VAR, cmap='jet')
        plt.axis('off')
        plt.text(5,30, Classifier+" Rank "+str(RANK+1),fontsize=40,color='red')
        plt.savefig("./"+str(RANK)+printpostfix+".pdf")
#----------------------#
#--- Create for GPC ---#
#----------------------#
plotHeatmaps("../PieVisualization/GPCData/ranking_BGPLVM_noResampling_manual.csv", "GPC", "_GPC_")
#--------------------#
#--- Same for HMC ---#
#--------------------#
plotHeatmaps("../PieVisualization/HMCData/ranking_BGPLVM_NoReplace.csv", "HMC", "_HMC_")
os.system("ls | grep pdf | while read line; do pdfcrop --margins '0 0 0 0' --clip $line $line; done")
