import seaborn as sbn #Plot fancy stuff
import pandas as pd #Load CSV file
import matplotlib.pyplot as plt #Plot stuff
import numpy as np #Load csv files
#--- Load data ---#
#-----------------#
ProcData = pd.read_csv("./Python_fullData.csv",header=0,sep=' ')
LMName = np.genfromtxt("./Python_LMName.csv", delimiter=' ',dtype=str)
center_np = np.genfromtxt("./Python_Centers.csv", delimiter=' ')
center_plotOrder = np.array((1,2,1,3,4,5,13,6,7,11,12,10,8,9,8,10,14,1))-1 #Taken from R script
LM_X_DEV= -0.045    #Settings from R script
LM_Y_DEV = -0.03    #Settings from R script
#------------#
#--- Plot ---#
#------------#
#--- Do scatter plot ---#
sbn.scatterplot(data=ProcData, x="X", y="Y", hue="Label", palette="deep",edgecolor="white",linewidth=0.1,s=15)
plt.xlabel("X Coordinate")  #Add custom label in X
plt.ylabel("Inverted Y Coordinate") #Add custom label in Y
#--- Add center lines and names ---#
plt.plot(center_np[center_plotOrder,0],center_np[center_plotOrder,1],c='k')
#--- Add names ---#
#--- Move some manual ---#
center_np[4,0]=center_np[4,0]-0.005
center_np[4,1]=center_np[4,1]+0.01
center_np[8,0]=center_np[8,0]-0.02
center_np[10,0]=center_np[10,0]-0.02
center_np[12,1]=center_np[12,1]+0.01
center_np[12,0]=center_np[12,0]-0.01
#--- Plot them ---#
for i in range(0,center_np.shape[0]):
    plt.text(   center_np[i,0]+LM_X_DEV,
                center_np[i,1]+LM_Y_DEV,LMName[i])
#plt.show()
plt.savefig("Procrustes.pdf")
