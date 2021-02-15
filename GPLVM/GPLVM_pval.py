# GPLVM_pval.py estiamtes the p-val images for GP-LVM. 
#
# This code is available under a GPL v3.0 license and comes without
# any explicit or implicit warranty.
#
# (C) Wilfried Wöber 2020 <wilfried.woeber@technikum-wien.at>
import sys                  #System stuff
sys.path.append('../Python/')  #Add path to project library
from bGPLVM import bGPLVM   #GPy wrapper
import numpy as np          #You should know that
import matplotlib.pyplot as plt #We aim to plot something...
image_dim = (224,224)
import os                   #For bash stuff
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
#-------------------------------#
#--- Create summarized plots ---#
#-------------------------------#
plot_top_n=((3*2),4)
selection = np.loadtxt("../Data/unselectedFeatures.csv", dtype=int)[0:12]
f, arr = plt.subplots(1,4)  #Create 'grid' for plot
x_looper=0
y_looper=0
for i in range(0,len(selection)):
    img=np.loadtxt('./Heatmaps/'+str(selection[i])+'.csv',delimiter=',')
    #pImg = getPimage(img,k)   #Get the p value image
    arr[x_looper].imshow(img,cmap='jet')
    arr[x_looper].text(5,30, "F"+str(selection[i]),fontsize=20,color='red')
    arr[x_looper].axis('off')
    #plt.subplots_adjust(wspace=0, hspace=0, left=0, right=0.1, bottom=0, top=1)
    #arr[y_looper+1,x_looper].imshow((1-pImg),cmap='gray')
    #arr[y_looper+1,x_looper].axis('off')
    x_looper=x_looper+1
    if(x_looper >= plot_top_n[1]):
        x_looper=0
        y_looper=y_looper+1
        plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
        plt.savefig("./Heatmaps/row_"+str(y_looper)+".pdf",bbox_inches = 'tight',pad_inches = 0)
        os.system("pdfcrop --margins '0 0 0 0' --clip ./Heatmaps/row_"+str(y_looper)+".pdf ./Heatmaps/row_"+str(y_looper)+".pdf")
        f, arr = plt.subplots(1,4)  #Create 'grid' for plot
x_looper=0
y_looper=0
for i in range(0,len(selection)):
    img=np.loadtxt('./Heatmaps/'+str(selection[i])+'.csv',delimiter=',')
    pImg = getPimage(img,k)   #Get the p value image
    mask = (pImg<alpha).astype(np.int8)
    res = cv2.bitwise_and(img,img,mask = mask)
    arr[x_looper].imshow(addBorder(res),cmap='jet')
    arr[x_looper].text(5,30, "F"+str(selection[i])+" MSK",fontsize=20,color='red')
    arr[x_looper].axis('off')
    x_looper=x_looper+1
    if(x_looper >= plot_top_n[1]):
        x_looper=0
        y_looper=y_looper+1
        plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
        plt.savefig("./Heatmaps/Prow_"+str(y_looper)+".pdf",bbox_inches = 'tight',pad_inches = 0)
        os.system("pdfcrop --margins '0 0 0 0' --clip ./Heatmaps/Prow_"+str(y_looper)+".pdf ./Heatmaps/Prow_"+str(y_looper)+".pdf")
        f, arr = plt.subplots(1,4)  #Create 'grid' for plot
os.system("cd Heatmaps; pdfjam row_1.pdf Prow_1.pdf row_2.pdf Prow_2.pdf row_3.pdf Prow_3.pdf --nup 1x6 --landscape --outfile pImg.pdf")
os.system("cd Heatmaps; pdfcrop --margins '0 0 0 0' --clip pImg.pdf pImg.pdf")
os.system("cd Heatmaps; rm row*.pdf; rm Prow*.pdf")
#----------------------------#
#--- Technical background ---#
#----------------------------#
selection = np.setdiff1d(np.array((range(0,11))), selection)[0:4]   #According to paper, the technical background 
f, arr = plt.subplots(1,4)  #Create 'grid' for plot
x_looper=0
y_looper=0
for i in range(0,len(selection)):
    img=np.loadtxt('./Heatmaps/'+str(selection[i])+'.csv',delimiter=',')
    arr[x_looper].imshow(img,cmap='jet')
    arr[x_looper].text(5,30, "F"+str(selection[i]),fontsize=20,color='red')
    arr[x_looper].axis('off')
    x_looper=x_looper+1
plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
plt.savefig("./Heatmaps/row_"+str(y_looper)+".pdf",bbox_inches = 'tight',pad_inches = 0)
os.system("pdfcrop --margins '0 0 0 0' --clip ./Heatmaps/row_"+str(y_looper)+".pdf ./Heatmaps/row_"+str(y_looper)+".pdf")
f, arr = plt.subplots(1,4)  #Create 'grid' for plot
x_looper=0
for i in range(0,len(selection)):
    img=np.loadtxt('./Heatmaps/'+str(selection[i])+'.csv',delimiter=',')
    pImg = getPimage(img,k)   #Get the p value image
    mask = (pImg<alpha).astype(np.int8)
    res = cv2.bitwise_and(img,img,mask = mask)
    arr[x_looper].imshow(addBorder(res),cmap='jet')
    arr[x_looper].text(5,30, "F"+str(selection[i])+" MSK",fontsize=20,color='red')
    arr[x_looper].axis('off')
    x_looper=x_looper+1
plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
#plt.show()
plt.savefig("./Heatmaps/Prow_"+str(y_looper)+".pdf",bbox_inches = 'tight',pad_inches = 0)
os.system("pdfcrop --margins '0 0 0 0' --clip ./Heatmaps/Prow_"+str(y_looper)+".pdf ./Heatmaps/Prow_"+str(y_looper)+".pdf")
os.system("cd Heatmaps; pdfjam row_0.pdf Prow_0.pdf --nup 1x2 --landscape --outfile TB_pImg.pdf")
os.system("cd Heatmaps; pdfcrop --margins '0 0 0 0' --clip TB_pImg.pdf TB_pImg.pdf")
os.system("cd Heatmaps; rm row*.pdf; rm Prow*.pdf")
