# coding: utf-8
# This code generates th p-value images. For a given alpha value,
# these images are thresholded and used as masks for the grayscale
# Nile tilapia images.
#
# This code is available under a GPL v3.0 license and comes without
# any explicit or implicit warranty.
#
# (C) Wilfried Woeber 2020 <wilfried.woeber@technikum-wien.at>
import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
import os
#--- global infos ---#
kernel_width = 5 	#Width and height of Gaussian kernel
kernel_sigma = 0 	#Default, estimated from width
k=10000				#Number of iterations
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
#--------------------------#
#--- Do main processing ---#
#--------------------------#
if __name__ == "__main__":
    if(len(sys.argv)==1):
        sys.argv=('','../Classification/Augmented/','GRAD')
    path_data= sys.argv[1]	    #Path to CNN data
    if(sys.argv[2] == "GRAD"):
        doGRAD=True     #Check if Grad-CAM should be used
    elif(sys.argv[2] == "LRP"):
        doGRAD=False
    else:
        print("ERROR: choose appropriate visualization")
        sys.exit(-1)
    pImg_k=1000                 #Number of samples for p img creation
    alpha = 0.001               #Alpha value for test
    #---------------------------#
    #--- Print system config ---#
    #---------------------------#
    print("Processing folger %s"%path_data)
    print("Use kernel width/height %d" % kernel_width)
    print("Use std sigma value")
    print("Use %d sampling iterations" % pImg_k)
    print("Using alpha value %f" %alpha)
    print("Doing Grad-CAM %d" %doGRAD)
    #------------------------------#
    #--- Process augmented data ---#
    #------------------------------#
    class_names=("Chamo","Hawassa","Koka","Lan","Tana","Ziway")
    Iteration="0"
    for k in range(0,10):
        print("Process %d"%k)
        test_folder=path_data+"test_"+Iteration+"_"+str(k)
        test_files = [f for f in os.listdir(test_folder) if 
                os.path.isfile(os.path.join(test_folder, f))]   #Get all files in the test folder
        img_files_all = [f for f in test_files if (os.path.splitext(f)[1]=='.npy')] #Get the npy images
        img_files = [f.split('__')[0] for f in img_files_all if len(f.split('LRP'))>1]     #Get specimen names
        #--- Now filter in best specimen names and all other specimens ---#
        img_files_bestSpecimen = [f for f in img_files if len(f.split('.'))!=1]
        img_files_allSpecimen  = [f for f in img_files if len(f.split('.'))==1]
        #--- Process all images ---#
        for IMAGE in img_files_allSpecimen:
            print("  Process %s"%IMAGE)
            img_name=IMAGE
            #---------------------------------------#
            #--- Get all images from data folder ---#
            #---------------------------------------#
            img_LRP = np.load(test_folder+"/"+img_name+"__LRP_10.npy")
            img_GRD = np.load(test_folder+"/"+img_name+"__grad.npy")
            img_GRD_raw = img_GRD.copy()
            #--- Conversion from heatmap to grayscale image ---#
            img_GRD = cv2.cvtColor(img_GRD, cv2.COLOR_RGB2LUV) 
            img_GRD = cv2.cvtColor(img_GRD, cv2.COLOR_BGR2GRAY) 
            img     = np.load(test_folder+"/"+img_name+"_.npy")
            img_RGB = img.copy()
            img     = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)     #Convert to grayscale
            #------------------------------------#
            #--- Do the p-value visualization ---#
            #------------------------------------#
            if(doGRAD):
                pImg=getPimage(img_GRD,pImg_k)  #Get p-value image
            else:
                pImg=getPimage(img_LRP,pImg_k)  #Get p-value image
            #------------------------#
            #--- Do visualization ---#
            #------------------------#
            f, arr = plt.subplots(1,3)
            arr[0].imshow(img, cmap='gray');arr[0].axis('off')
            if(doGRAD):
                arr[1].imshow(img_GRD_raw);arr[1].axis('off')
            else:
                arr[1].imshow(img_LRP);arr[1].axis('off')
            #arr[2].imshow(pImg<alpha, cmap='gray');arr[2].axis('off')
            img_RGB[:,:,0]=img_RGB[:,:,2]+addBorder(np.multiply(pImg<alpha,img))
            arr[2].imshow(img_RGB);arr[2].axis('off')
            #arr[3].imshow(np.multiply(pImg<alpha,img), cmap='gray');arr[3].axis('off')
            plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
            #plt.show()
            plt.savefig(IMAGE+".pdf",bbox_inches = 'tight',pad_inches = 0)
            plt.savefig(IMAGE+".png",bbox_inches = 'tight',pad_inches = 0)
            plt.close()
