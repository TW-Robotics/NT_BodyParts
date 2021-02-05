# This script implements Procrustes extraction based on given landmarks.
# Note, that the publication is based on the shapes R packages, which is 
# also included in this package.
#    
# The code in evalres.py is available under a GPL v3.0 license and
# comes without any explicit or implicit warranty.
#
#
# (C) P. Sykacek 2020 <peter@sykacek.net> 
import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
def genCcetering(configX):
    ## generate the centering matrix of a configuration matrix X
    ##
    ## IN
    ##
    ## configX: [k x m] dimensional matrix of k coordinates of m -
    ##          dimensional landmark s (points) which characterise one
    ##          sample instance.
    ##
    ## OUT
    ##
    ## centrC: [k x k] dimensional centering matrix. Note that CX is
    ##          the centered version of X such that the centroid of CX
    ##          is zero.
    ##
    ## (C) P. Sykacek

    if type(configX) != type(np.array([])):
        configX=np.array(configX)
    k=configX.shape[0]
    centrC=np.eye(k)-np.ones((k,k))/k
    return centrC

def config2centroid(configX):
    ## calculate the centroid of a configuration (mean accross k landmarks)
    ##
    ## IN
    ##
    ## configX: [k x m] dimensional matrix of k coordinates of m -
    ##          dimensional landmarks (points) which characterise one
    ##          sample instance.
    ##
    ## OUT
    ##
    ## centrX:  [m x ] dimensional vector of centroid values.
    ##
    ## (C) P. Sykacek

    if type(configX) != type(np.array([])):
        configX=np.array(configX)
    return np.mean(configX, axis=0)

def ctrdsz(configX):
    ## calculate the centroid size of a configuration
    ##
    ## IN
    ##
    ## configX: [k x m] dimensional matrix of k coordinates of m -
    ##          dimensional landmark s (points) which characterise one
    ##          sample instance.
    ##
    ## OUT
    ##
    ## ctsz: scalar centroid size of configuration. (square root of
    ##          sum of squares of configuration coordinates minus
    ##          centroid of configuration)
    ##
    ## (C) P. Sykacek
    
    if type(configX) != type(np.array([])):
        configX=np.array(configX)
    C=genCcetering(configX)
    CX=np.dot(C, configX)
    return np.sqrt(np.trace(np.dot(np.transpose(CX), CX)))

def ctrcfX(configX):
    ## center a configuration
    ##
    ## IN
    ##
    ## configX: [k x m] dimensional matrix of k coordinates of m -
    ##          dimensional landmark s (points) which characterise one
    ##          sample instance.
    ##
    ## OUT
    ##
    ## ctrdX:   [k x m] dimensional matrix with centered configuration
    ##
    ## (C) P. Sykacek
    if type(configX) != type(np.array([])):
        configX=np.array(configX)
    C=genCcetering(configX)
    return np.dot(C, configX)

def sclcfX(configX):
    ## scale a configuration
    ##
    ## IN
    ##
    ## configX: [k x m] dimensional matrix of k coordinates of m -
    ##          dimensional landmark s (points) which characterise one
    ##          sample instance.
    ##
    ## OUT
    ##
    ## scldX:   [k x m] dimensional matrix with scaled configuration
    ##
    ## (C) P. Sykacek
    if type(configX) != type(np.array([])):
        configX=np.array(configX)
    scl=ctrdsz(configX)
    return configX/scl


def getrot(tsX1, tsX2):
    ## get an optimal rotation matrix Gamma which minimizes
    ## ctrdsz(tsX2-tsX1 * Gamma). The optimal Gamma is given by U*V^T
    ## where V*L*U^T = SVD(tsX2^T * tsX1)
    ##
    ## IN
    ##
    ## tsX1: scaled and translated source configuration
    ##       which is to be optimally rotated into tsX2.
    ## tsX2: scaled and translated target configuration
    ##
    ## OUT
    ##
    ## Gamma: optimal rotation matrix.
    ##
    ## (C) P. Sykacek
    if type(tsX1) != type(np.array([])):
        tsX1=np.array(tsX1)
    if type(tsX2) != type(np.array([])):
        tsX2=np.array(tsX2)
    U, L, VT=np.linalg.svd(np.dot(np.transpose(tsX2), tsX1))
    ## Gamma -> the transpose is a result of differnces in formulation
    ## in Dryden & Mardia and the syntax in numpy.linalg.svd.
    return np.transpose(np.dot(U, VT)) 

def dorot(tsX1, tsX2):
    ## rotates tsX1 optimally into tsX2.
    ##
    ## IN
    ##
    ## tsX1: scaled and translated source configuration
    ##       which is to be optimally rotated into tsX2.
    ## tsX2: scaled and translated target configuration
    ##
    ## OUT
    ##
    ## tsrX1: optimally rotated tsX1.
    ##
    ## (C) P. Sykacek
    if type(tsX1) != type(np.array([])):
        tsX1=np.array(tsX1)
    if type(tsX2) != type(np.array([])):
        tsX2=np.array(tsX2)
    return np.dot(tsX1, getrot(tsX1, tsX2))

def doGPA(allconfigs, maxeps=10**-7, maxit=500):
    ## apply a generalized procrustes analysis
    ##
    ## IN
    ##
    ## allconfigs: a 3 dim numpy tensor with all configurations to be
    ##             analysed.  this is a [k x m x n] dimensional tensor
    ##             of k coordinates of m - dimensional landmarks
    ##             provided for n samples.
    ##
    ## maxeps,
    ## maxit:      flow control of GPA.
    ##
    ## OUT
    ##
    ## alltsrconfigs: a 3 dim numpy tensor with all translated, scaled and optimally rotated
    ##             configurations (the GPA result).
    ##
    ## (C) P. Sykacek
    if type(allconfigs) != type(np.array([])):
        allconfigs=np.array(allconfigs)
    ## make sure we operate on a copy to leave the original data invariant.
    allconfigs=copy.deepcopy(allconfigs)
    ## first we apply the translation and rescaling for all n configurations.
    for n in range(allconfigs.shape[2]):
        ## centering 
        tX=ctrcfX(allconfigs[:,:,n])
        ## rescaling
        tsX=sclcfX(tX)
        ## and update allconfigs[:,:,n]
        allconfigs[:,:,n]=tsX
    ## get average shape as rotation target
    mncfg=np.mean(allconfigs, axis=2)
    mncfg=ctrcfX(mncfg)
    mncfg=sclcfX(mncfg)
    doiter=True
    cit=0
    while doiter:
        ## we rotate allconfigs[:,:,n] into mncfg (the mean shape)
        for n in range(allconfigs.shape[2]):
            allconfigs[:,:,n]=dorot(allconfigs[:,:,n], mncfg)
        ## mew average shape
        mncfg=np.mean(allconfigs, axis=2)
        mncfg=ctrcfX(mncfg)
        mncfg=sclcfX(mncfg)
        cdist=np.max([ctrdsz(mncfg-allconfigs[:,:,n]) for n in range(allconfigs.shape[2])])
        doiter=cit<maxit and cdist>maxeps
        cit=cit+1
        #print(cdist)
    return allconfigs
#-----------------------#
#--- Main processing ---#
#-----------------------#
landmarks_x = np.loadtxt("../Data/Landmarks/rawLandmarks_X.csv",delimiter=",")
landmarks_y = np.loadtxt("../Data/Landmarks/rawLandmarks_Y.csv",delimiter=",")
metadata = pd.read_csv("../Data/Landmarks/rawLandmarks_MetaData.csv", names=['specimenNames', 'target'])
landmarks = np.zeros((14,2,209))
landmarks[:,0,:]=landmarks_x
landmarks[:,1,:]=landmarks_y
procrustes = doGPA(landmarks)
#--------------------------#
#--- Convert to XY data ---#
#--------------------------#
data = []       #DF for procrustes data
for i in range(0,procrustes.shape[2]):  #Loop over all specimens
    label = metadata['target'][i]       #Get current label
    ID = metadata['specimenNames'][i]   #Get current specimen ID name
    DF_data = {'name':ID,'target':label}
    targetOrder = ['name','target']
    for l in range(0,procrustes.shape[0]):  #Looper over nr of landmarks
        DF_data.update({'X'+str(l):procrustes[l,0,i]})   #X coordinate
        DF_data.update({'Y'+str(l):procrustes[l,1,i]})   #Y coordinate
        targetOrder.append('X'+str(l))
        targetOrder.append('Y'+str(l))
    if(len(data)==0):
        data = pd.DataFrame(data=DF_data,index=[0])
    else:
        data=data.append(DF_data,ignore_index=True)
data=data[targetOrder]  #Get right order
data.to_csv('PROCRUSTES_DATApy.csv',header=False, index=False)
#--------------------#
#--- Plot results ---#
#--------------------#
cmap = plt.get_cmap("jet")
plt_colors = cmap(metadata['target']/6)
coordinates = np.array(data[data.columns[2:]])
for i in range(0,procrustes.shape[2]):  #Loop over all specimens
    for n in range(0,coordinates.shape[1],2):   #Loop over all landmarks
        plt.plot(coordinates[i,n],-coordinates[i,n+1],'.',c=plt_colors[i,:])    #NOTE: negative Y axis
PC = (0,1,0,2,3,4,12,5,6,10,11,9,7,8,7,9,13,0)  #Plot center IDs of landmarks
coordinates_X=coordinates[:,range(0,28,2)]      #X values of procrustes coordinates
coordinates_Y=coordinates[:,range(1,28,2)]      #Y values of procrustes coordinates
for i in range(0,len(PC)-1):
    plt.plot(   (np.mean(coordinates_X[:,PC[i]]),(np.mean(coordinates_X[:,PC[i+1]]))), 
                (-np.mean(coordinates_Y[:,PC[i]]),-(np.mean(coordinates_Y[:,PC[i+1]]))),
                '-',c='black')
plt.xlabel("X Coordinate")
plt.ylabel("Inverted Y Coordinate")
import matplotlib.patches as mpatches
Chamo_patch =   mpatches.Patch(color=cmap(0/6), label='Chamo')
Hawassa_patch = mpatches.Patch(color=cmap(1/6), label='Hawassa')
Koka_patch =    mpatches.Patch(color=cmap(2/6), label='Koka')
Lan_patch =     mpatches.Patch(color=cmap(3/6), label='Langano')
Tana_patch =    mpatches.Patch(color=cmap(4/6), label='Tana')
Ziway_patch =   mpatches.Patch(color=cmap(5/6), label='Ziway')
plt.legend(handles=[Chamo_patch,Hawassa_patch,Koka_patch,Lan_patch,Tana_patch,Ziway_patch],loc='upper right')
#plt.show()
plt.savefig("RawProcrustespy.pdf", bbox_inches='tight')
