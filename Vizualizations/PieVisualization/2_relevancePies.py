# This script uses the previously calculated relevance values
# and plots the piecharts.
#
# This code is available under a GPL v3.0 license and comes without
# any explicit or implicit warranty.
#
# (C) Wilfried Woeber 2020 <wilfried.woeber@technikum-wien.at>
import pandas as pd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import os
#--------------------------#
#--- Global definitions ---#
#--------------------------#
LM_name = ["UTP","EYE","AOD","POD","DIC","VOC","PIA","BPF","PEO","VEO","AOA","AOP","HCF","EMO"]
#-----------------#
#--- Functions ---#
#-----------------#
## combine lengthscales in L to an isotropic lengthscale matrix
## lambda.  Note that lambda is charactrised by one constant l
## which is the scalar multiplied to the identity matrix to give
## lambda. The approximation minimizes the sum of squares between
## the quadratic norms: sum_n (x_n^T L x_n - l x_n^T I x_n)^2
##
## IN
##
## X: [N x D] times matrix with N D dimensional input vectors.
##
## L: [D x D] dimensional lengthscale matrix. Matrix L would
##    normally be diagonal but that is not a precondition of the
##    algorithm.
##
## OUT
##
## l: scaler which when multiplied with a [D x D] identity matrix
##    approximates L.
##
## (C) P. Sykacek 2020 <peter@sykacek.net>
def combscale(design, LS):
    LS = np.array(LS)           #Convert to array
    LS = np.transpose(LS)       #Transpose flat vector to standing vector
    converted_LS = np.zeros((int(LS.shape[0]/2),LS.shape[1]))  #Prepare memory - we combine X and Y, thus /2
    for D in range(0,LS.shape[1]):
        looper = 0                      #Index looper
        for feature in range(0,28,2):   #Loop over X/Y combinations
            LS_local = np.diag((LS[(feature):(feature+2),D]))   #Get current LS pair
            sqrL = np.sum(np.dot(design[:,(feature):(feature+2)],LS_local)*design[:,(feature):(feature+2)],axis=1)
            sqrI = np.sum(design[:,(feature):(feature+2)]*design[:,(feature):(feature+2)],axis=1)
            converted_LS[looper,D] = np.sum(sqrL*sqrI)/np.sum(sqrI**2)
            looper = looper + 1         #Increment looping counter
        #End feature loop
    #End dimension loop
    return(converted_LS)    #Return combined LS values
# This function converts the HMC procrustes landmarks
# as discussed in the paper
def combscale_HMC(relevance):
    relevance = np.array(relevance)           #Convert to array
    relevance = np.transpose(relevance)         #Convert vector
    converted_relevance = np.zeros((int(relevance.shape[0]/2),relevance.shape[1]))  #Prepare memory - we combine X and Y, thus /2
    looper=0
    for feature in range(0,28,2):   #Loop over X/Y combinations
        converted_relevance[looper,0]=np.sqrt(relevance[feature,0]**2+relevance[1+feature,0]**2)
        looper=looper+1
    converted_relevance = np.transpose(converted_relevance) #Convert back to flat vector
    return(converted_relevance) 
# This function calculates the relevance of each position. 
# IN
# ranking:       Ranking matrix perexperiment
# OUT
# plot pie data
def pieData_getPositionRanking(ranking, top_N=3):
    pie_data = np.zeros((top_N, ranking.shape[0]))   #Init pie data to plot
    for rank in range(0,top_N):
        for feature in range(0,ranking.shape[0]):
            pie_data[rank,feature]=np.sum(ranking[feature,:]==rank)
        #End feature loop
    #end rank loop
    return(pie_data)
#-----------------------#
#--- Main processing ---#
#-----------------------#
nr_classes = 6          #Number of populations
k_fold = range(0,10)    #Range of ks
iterations = 10         #Number of iterations
top_N=5                 #Number of top features to plot
selection   = np.genfromtxt("../../Data/unselectedFeatures.csv", delimiter=' ',dtype=int)               #Load manual selection
GP_names    = ["F_"+str(k) for k in selection]  #Create names for GP-LVM features
Procrustes_PD  = pd.read_csv("../../Procrustes/PROCRUSTES_DATA.csv",header=None, usecols=range(2,30))   #Get procrustes data just the data
Procrustes =np.array(Procrustes_PD)         #Convert to an array
shuffleData = np.genfromtxt("../../Data/NoReplace_20205809_085859.csv",dtype=int)                       #The shuffled data we need that for Procrustes conversion
#--- Create initial data ---#
data_1 = {'name': "BGPLVM_noResampling_manual", 'folder':"../../GPC/GPLVM/",                'isCNN': False, 'nrFeatures':14, 'isProc':False, 'isHMC': False}
data_2 = {'name': "BGPLVM_noResampling_top14",  'folder':"../../GPC/GPLVM_full/",           'isCNN': False, 'nrFeatures':14, 'isProc':False, 'isHMC': False}
data_3 = {'name': "Procrustes_noResampling",    'folder':"../../GPC/Procrustes/",           'isCNN': False, 'nrFeatures':14, 'isProc':True,  'isHMC': False} #Actually 28 features...
data_4 = {'name': "BGPLVM_noResampling_manual", 'folder':"./resdata/BGPLVM_NoReplace",      'isCNN': False, 'nrFeatures':14, 'isProc':False, 'isHMC': True}
data_5 = {'name': "BGPLVM_noResampling_top14",  'folder':"./resdata/BGPLVM_NoReplace_14",   'isCNN': False, 'nrFeatures':14, 'isProc':False, 'isHMC': True}
data_6 = {'name': "Procrustes_noResampling",    'folder':"./resdata/Procrustes_NoReplace",  'isCNN': False, 'nrFeatures':14, 'isProc':True,  'isHMC': True} #Actually 28 features...
#data_4 = {'name': "CNN_Aug_noResampling",       'folder':"../../CNN/Classification/Augmented/", 'isCNN': True, 'nrFeatures':0, 'isProc':False}
DF = pd.DataFrame(data=data_1,index=[0])
DF = DF.append(data_2,ignore_index=True)
DF = DF.append(data_3,ignore_index=True)
DF = DF.append(data_4,ignore_index=True)
DF = DF.append(data_5,ignore_index=True)
DF = DF.append(data_6,ignore_index=True)
#DF = pd.DataFrame(data=data_3,index=[0])

#DF = DF.append(data_4,ignore_index=True) #The final information matrix to process do not show CNN relevance
#-----------------------------#
#--- Do the mainprocessing ---#
#-----------------------------#
for i in range(0,len(DF)):  #Loop over all defined classifiers
    data = DF.loc[i,]       #Get classifier info
    isProcrustes = data[3]  #Procrustes indicator
    name = data[4]          #Name of classifier
    path = data[0]          #Path to classifier data
    isCNN= bool(data[1])    #See if classifier was an CNN
    nrFeatures=data[5]      #Number of features of classifier
    isHMC= bool(data[2])    #See if classifier was HMC
    #--- Process the iterations ---#
    relevance_memory = np.zeros((iterations,nrFeatures))    #Memory for calculated relevance values
    for n in range(0,iterations):
        #--- Convert if HMC format ---#
        if(isHMC):
            DF_relevance = pd.read_csv(path+"/"+str(n)+"_Relevance.csv")
            #--- Cut out appropriate ARD vals ----#
            if(isProcrustes):
                DF_relevance = np.array(DF_relevance)[:,0:28]
            else:
                DF_relevance = np.array(DF_relevance)[:,0:14]
        else:
            DF_relevance = pd.read_csv("./data/data_"+name+"/"+str(n)+"_Relevance.csv")
        #--- Process Procrustes ---#
        if(isProcrustes):
            print("Proc Procrustes")
            if(isHMC):
                relevance_memory[n,:]=combscale_HMC(DF_relevance)
            else:
                designProcrustes = Procrustes[shuffleData[:,n],:]   #Get design matrix for the inherent LS values
                DF_LS = 1./DF_relevance                             #Convert back to lengthscales
                comb_LS = combscale(designProcrustes,DF_LS)         #Combine procrustes locations
                relevance_memory[n,:]=np.transpose(1/comb_LS)       #Store combined procrustes
        else:
            print("Proc GPLVM")
            relevance_memory[n,:]=np.array(DF_relevance)    #Store loaded relevance value
    #End iteration processing
    #--- Get ranking of each iteration ---#
    if(isHMC):
       name='HMC_'+name
    ranking_per_experiment = np.zeros((relevance_memory.shape[1],relevance_memory.shape[0]))    #Init ranking matrix with approproate size
    for n in range(0,ranking_per_experiment.shape[1]):
        print("Proc row:%d"%n)
        ranking_per_experiment[:,n]=np.flip(np.argsort(relevance_memory[n,:]))    #Decreasing sort
    ranking_per_experiment=ranking_per_experiment.astype(int)   #Convert to int (indices are ints)
    ranking_pie_position = pieData_getPositionRanking(ranking_per_experiment,top_N)   #Get pie data
    np.savetxt("ranking_"+name+".csv", ranking_pie_position, delimiter=',')     #Store ranking results
    #np.savetxt("relevance_"+name+".csv", relevance_memory, delimiter=',')     #Store ranking results
    #---------------------#
    #--- Plot the pies ---#
    #---------------------#
    cmap = plt.get_cmap("jet")
    plt_colors = [cmap(n/14) for n in range(0,14)]
    # Now, we process each possible rank of the features 
    for k in range(0,top_N):    #Loop over top features
        plot_pie_data = pd.DataFrame()  #Init empty pandas data frame
        for m in np.nonzero(ranking_pie_position[k,:]>0)[0]:   #Loop over non zero elements
            if(isProcrustes):
                data_local = {'name':LM_name[m], 'col':plt_colors[m], 'match':ranking_pie_position[k,m]}
            else:
                data_local = {'name':GP_names[m], 'col':plt_colors[m], 'match':ranking_pie_position[k,m]}
            plot_pie_data = plot_pie_data.append(data_local, ignore_index=True) #Extend data frame
        #--- Plot the pies ---#
        texts, autotexts =plt.pie(plot_pie_data['match'],
                labels=plot_pie_data['name'], 
                colors=plot_pie_data['col'],
                startangle=0,labeldistance=0.5
                )
        plt.setp(autotexts, size=20, weight="bold")
        plt.title('Rank '+str(k+1),fontsize=20)
        plt.savefig("./"+name+"_R"+str(k)+"_pie.pdf")
        plt.close()
        #--- Make pdf beautiful ---#
        os.system("pdfcrop --margins '0 0 0 0' --clip ./"+name+"_R"+str(k)+"_pie.pdf ./"+name+"_R"+str(k)+"_pie.pdf")
    #---------------------#
    #--- Combine plots ---#
    #---------------------#
    os.system("pdfjam "+ name+"_R0_pie.pdf "+ name+"_R1_pie.pdf "+ name+"_R2_pie.pdf "+ name+"_R3_pie.pdf "+ name+"_R4_pie.pdf --nup 5x1 --landscape --outfile "+name+"_pie.pdf")
    os.system("pdfcrop --margins '0 0 0 0' --clip "+name+"_pie.pdf "+name+"_pie.pdf")
    os.system("rm *_R*_pie.pdf") #Remove old results
