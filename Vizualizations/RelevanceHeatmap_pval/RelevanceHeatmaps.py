import sys                  #System stuff
import numpy as np          #You should know that
import matplotlib.pyplot as plt #We aim to plot something...
import os                   #For bash commands
import cv2
#--- global infos ---#
kernel_width = 5 	#Width and height of Gaussian kernel
kernel_sigma = 0 	#Default, estimated from width
k=10000			#Number of iterations
alpha = 0.001		#Alpha threshold value
#-----------------#
#--- functions ---#
#-----------------#
def mySmoother(img):
    img = img.copy()    #Get a deep copy of the image
    img = cv2.GaussianBlur(img,(kernel_width,kernel_width),kernel_sigma)#Smooth image
    return(img)
def randomizeImage(img):
    img = img.copy()
    img_vector = img.reshape((img.shape[0]*img.shape[1]),1)
    index = np.random.choice(range(0,np.prod(img.shape)),np.prod(img.shape),replace=False)
    random_image = np.zeros((np.prod(img.shape),1))
    random_image[index]=img_vector
    random_image=random_image.reshape(img.shape)
    return(random_image)
def normalize(img):
    img=(img-np.min(img))/(np.max(img)-np.min(img))
    return(img)
def getPimage(img,k):
    memory = np.zeros((224,224,k))
    for looper in range(0,k):
        memory[:,:,looper]=mySmoother(randomizeImage(img))
    print("Check random images")
    counter = np.zeros((224,224))
    smoothed_LRP = mySmoother(img)
    for R in range(0,224):
        for C in range(0,224):
            random_vector = memory[R,C,:]
            counter[R,C] = np.sum(
                                    random_vector > smoothed_LRP[R,C]
                                    )
    #--- normalize image ---#
    counter_norm = counter/float(k)
    return(counter_norm)
def addBorder(img):
    #--- add black border ---#
    for i in range(0,int(kernel_width/2)):
        cv2.rectangle(  img,   #Destination=source
                    (0+i,0+i),          #Top left
                    (img.shape[0]-1-i,img.shape[1]-1-i),#Bottom right
                    (0),            #Color
                    1) #Thickness of lines
    return(img)



def plotHeatmaps(relevance,Classifier,printpostfix):
    invSelection	= "../../Data/unselectedFeatures.csv"
    Selection           = np.loadtxt(invSelection,   delimiter=",",dtype=int)   #selection of features
    #relevance       = "../PieVisualization/GPCData/ranking_BGPLVM_noResampling_manual.csv"#  "../../../20200923_Figures_Addon/Pies_GPC/ranking_BGPLVM_noResampling_manual.csv"
    RelevanceRanking    = np.loadtxt(relevance,   delimiter=",",dtype=float)    #Estimated relevance rank
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
        plt.close()
        plt.figure()
        MEAN_Fish_VAR=np.reshape(rel_img, (224,224))
        plt.imshow(MEAN_Fish_VAR, cmap='jet')
        plt.axis('off')
        plt.text(5,30, Classifier+" Rank "+str(RANK+1),fontsize=40,color='red')
        plt.savefig("./"+str(RANK)+printpostfix+".pdf")
        #--- Get p img ---#
        plt.close()
        img=MEAN_Fish_VAR
        pImg = getPimage(MEAN_Fish_VAR,k)   #Get the p value image
        mask = (pImg<alpha).astype(np.int8)
        res = cv2.bitwise_and(img,img,mask = mask)
        plt.imshow(addBorder(res),cmap='jet')
        plt.axis('off')
        plt.text(5,30, Classifier+" Rank "+str(RANK+1)+ " MSK",fontsize=30,color='red')
        plt.savefig("./"+str(RANK)+printpostfix+"P.pdf")

#----------------------#
#--- Create for GPC ---#
#----------------------#
plotHeatmaps("../PieVisualization/ranking_BGPLVM_noResampling_manual.csv", "GPC", "_GPC_")
##--------------------#
##--- Same for HMC ---#
##--------------------#
plotHeatmaps("../PieVisualization/ranking_HMC_BGPLVM_noResampling_manual.csv", "HMC", "_HMC_")
os.system("ls | grep pdf | while read line; do pdfcrop --margins '0 0 0 0' --clip $line $line; done")
#--------------------------#
#--- Create final image ---#
#--------------------------#
os.system("pdfjam 1_GPC_.pdf 1_GPC_P.pdf --nup 1x2 --landscape --outfile 1.pdf")
os.system("pdfjam 0_GPC_.pdf 0_GPC_P.pdf --nup 1x2 --landscape --outfile 0.pdf")
os.system("pdfcrop --margins '0 0 0 0' --clip 0.pdf 0.pdf")
os.system("pdfcrop --margins '0 0 0 0' --clip 1.pdf 1.pdf")

os.system("pdfjam 0_HMC_.pdf 0_HMC_P.pdf --nup 1x2 --landscape --outfile 0_HMC.pdf")
os.system("pdfjam 1_HMC_.pdf 1_HMC_P.pdf --nup 1x2 --landscape --outfile 1_HMC.pdf")
os.system("pdfcrop --margins '0 0 0 0' --clip 0_HMC.pdf 0_HMC.pdf")
os.system("pdfcrop --margins '0 0 0 0' --clip 1_HMC.pdf 1_HMC.pdf")

os.system("pdfjam 0.pdf 1.pdf --nup 2x1 --landscape --outfile 1_full.pdf")
os.system("pdfcrop --margins '0 0 0 0' --clip 1_full.pdf 1_full.pdf")

os.system("pdfjam 0_HMC.pdf 1_HMC.pdf --nup 2x1 --landscape --outfile 1_full_HMC.pdf")
os.system("pdfcrop --margins '0 0 0 0' --clip 1_full_HMC.pdf 1_full_HMC.pdf")

os.system("pdfjam 1_full.pdf 1_full_HMC.pdf --nup 1x2 --landscape --outfile fin.pdf")
os.system("pdfcrop --margins '0 0 0 0' --clip fin.pdf fin.pdf")
