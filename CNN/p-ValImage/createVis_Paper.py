# coding: utf-8
# This script generates the images used in the publication.
# Note, that the results of the CNN are strongly based on random 
# initialization. Thus, you have to adapt the scripts if you want
# similar images generated
#
# This code is available under a GPL v3.0 license and comes without
# any explicit or implicit warranty.
#
# (C) Wilfried Woeber 2020 <wilfried.woeber@technikum-wien.at>
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import createVis as VIZ
import subprocess
import cv2
#----------------------#
#--- Main variables ---#
#----------------------#
populations = ('Chamo','Hawassa','Koka','Lan','Tana','Ziway')
enumerations = ('a', 'b', 'c', 'd', 'e', 'f')
#---------------------------------------#
#--- Specimen spurious visualization ---#
#---------------------------------------#
## We not turn our discussion to extract chosen 
## samples from the CNN result. We manually selected
## sample, where we found spurious CNN decisions.
alpha=0.001
pImg_k=10000
iteration = "0"     #We just analyse the very first iteration
names = ('Ziway', 'Ziway', 'Tana', 'Langano', 'Koka', 'Koka')   #Names of spurious specimens
ID    = ('38',    '35',    '24',   '25',      '27',   '03')     #Inherent ID
for i in range(0,len(names)):
    #------------------#
    #--- Load image ---#
    #------------------#
    cmd="cd ../Classification/Augmented/; find ./ | grep "+names[i]+ID[i]+"_.npy | grep test_0"
    process = subprocess.Popen(['bash', '-c', cmd], stdout=subprocess.PIPE) #Create bash command
    out, err = process.communicate()            #Perform bash command
    out_str = str(out)                          #Convert to string
    path_img = out_str[2:out_str.find('\\')-4]    #Get path from data folder to image file
    #---------------------#
    #--- Create images ---#
    #---------------------#
    img     = np.load("../Classification/Augmented/"+path_img+'.npy')
    img_RGB = img.copy()
    img_RGB2 = img.copy()
    img     = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_LRP = np.load("../Classification/Augmented/"+path_img+'_LRP_10.npy')
    img_GRD = np.load("../Classification/Augmented/"+path_img+'_grad.npy')
    img_GRD_raw = img_GRD.copy()
    img_GRD_raw = img_GRD.copy()
    #--- Conversion from heatmap to grayscale image ---#
    img_GRD = cv2.cvtColor(img_GRD, cv2.COLOR_RGB2LUV) 
    img_GRD = cv2.cvtColor(img_GRD, cv2.COLOR_BGR2GRAY) 
    #--- Get p-val images ---#
    pImg_GRD=VIZ.getPimage(img_GRD,pImg_k)  #Get p-value image
    pImg_LRP=VIZ.getPimage(img_LRP,pImg_k)  #Get p-value image
    #-----------------------------#
    #--- Create visualizations ---#
    #-----------------------------#
    #LRP
    f, arr = plt.subplots(1,3)  #All 3 in a row
    arr[0].imshow(img, cmap='gray');arr[0].axis('off')
    #arr[0].text(5,30, names[i]+ID[i],fontsize=20,color='red')
    arr[0].text(5,30, names[i],fontsize=20,color='red')
    arr[1].imshow(img_LRP);arr[1].axis('off')
    img_RGB[:,:,0]=img_RGB[:,:,2]+VIZ.addBorder(np.multiply(pImg_LRP<alpha,img))
    arr[2].imshow(img_RGB);arr[2].axis('off')
    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
    #plt.show()
    plt.savefig("_LRP_"+str(i)+".pdf",bbox_inches = 'tight',pad_inches = 0)
    os.system("pdfcrop --margins '0 0 0 0' --clip _LRP_"+str(i)+".pdf _LRP_"+str(i)+".pdf")
    plt.close()
    #Grad-CAM
    f, arr = plt.subplots(1,3)  #All 3 in a row
    arr[0].imshow(img, cmap='gray');arr[0].axis('off')
    #arr[0].text(5,30, names[i]+ID[i],fontsize=20,color='red')
    arr[0].text(5,30, names[i],fontsize=20,color='red')
    arr[1].imshow(img_GRD_raw);arr[1].axis('off')
    img_RGB2[:,:,0]=img_RGB2[:,:,2]+VIZ.addBorder(np.multiply(pImg_GRD<alpha,img))
    arr[2].imshow(img_RGB2);arr[2].axis('off')
    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
    #plt.show()
    plt.savefig("_GRD_"+str(i)+".pdf",bbox_inches = 'tight',pad_inches = 0)
    os.system("pdfcrop --margins '0 0 0 0' --clip _GRD_"+str(i)+".pdf _GRD_"+str(i)+".pdf")
    plt.close()
#---------------------#
#--- Combine plots ---#
#---------------------#
os.system("pdfjam _LRP_0.pdf _LRP_1.pdf _LRP_2.pdf _LRP_3.pdf _LRP_4.pdf _LRP_5.pdf --nup 1x6 --landscape --outfile LRP_anomal.pdf")
os.system("pdfcrop --margins '0 0 0 0' --clip LRP_anomal.pdf LRP_anomal.pdf")
os.system("pdfjam _GRD_0.pdf _GRD_1.pdf _GRD_2.pdf _GRD_3.pdf _GRD_4.pdf _GRD_5.pdf --nup 1x6 --landscape --outfile GRD_anomal.pdf")
os.system("pdfcrop --margins '0 0 0 0' --clip GRD_anomal.pdf GRD_anomal.pdf")
os.system("rm _LRP_*; rm _GRD_*")
os.system("mkdir anomaly; mv LRP_anomal.pdf anomaly/.; mv GRD_anomal.pdf anomaly/.")
#------------------------------------#
#--- Create plots for best models ---#
#------------------------------------#
for MODE in range(0,3):
    #MODE = 2
    iterations = range(0,10)
    k_folds = range(0,10)
    for ITERATION in iterations:
        for K in k_folds:
            looper=0
            for CLASS in populations:
                plt.close()     #Close existing plots
                img     = np.load(  "../Classification/Augmented/test_"+
                                    str(ITERATION)+"_"+str(K)+"/"+CLASS+'.npy')
                img_LRP  = np.load(  "../Classification/Augmented/test_"+
                                    str(ITERATION)+"_"+str(K)+"/"+CLASS+'_LRP_10.npy')
                img_GRD  = np.load(  "../Classification/Augmented/test_"+
                                    str(ITERATION)+"_"+str(K)+"/"+CLASS+'_grad.npy')
                img_RGB = img.copy()
                img_RGB2 = img.copy()
                img     = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img_GRD_raw = img_GRD.copy()
                img_GRD_raw = img_GRD.copy()
                #--- Conversion from heatmap to grayscale image ---#
                img_GRD = cv2.cvtColor(img_GRD, cv2.COLOR_RGB2LUV) 
                img_GRD = cv2.cvtColor(img_GRD, cv2.COLOR_BGR2GRAY) 
                #--- Get p-val images ---#
                pImg_GRD=VIZ.getPimage(img_GRD,pImg_k)  #Get p-value image
                pImg_LRP=VIZ.getPimage(img_LRP,pImg_k)  #Get p-value image
                #---------------------#
                #--- Create images ---#
                #---------------------#
                if(CLASS =="Lan"):
                    CLASS="Langano"
                #LRP
                f, arr = plt.subplots(1,3)  #All 3 in a row
                if(MODE == 0):
                    arr[0].imshow(img, cmap='gray');arr[0].axis('off')
                    arr[0].text(5,30, CLASS ,fontsize=20,color='red')
                elif(MODE == 1):
                    arr[0].imshow(img, cmap='gray')
                    arr[0].spines['right'].set_visible(False)
                    arr[0].spines['left'].set_visible(False)
                    arr[0].spines['top'].set_visible(False)
                    arr[0].spines['bottom'].set_visible(False)
                    arr[0].set_xticks([], [])
                    arr[0].set_yticks([], [])
                    arr[0].set_xlabel(enumerations[looper]+')',fontsize=18)
                elif(MODE == 2):
                    arr[0].imshow(img, cmap='gray');arr[0].axis('off')
                if(MODE == 2):
                    arr[1].imshow(img_LRP)
                    arr[1].spines['right'].set_visible(False)
                    arr[1].spines['left'].set_visible(False)
                    arr[1].spines['top'].set_visible(False)
                    arr[1].spines['bottom'].set_visible(False)
                    arr[1].set_xticks([], [])
                    arr[1].set_yticks([], [])
                    arr[1].set_xlabel('('+enumerations[looper]+')',fontsize=18)
                else:
                    arr[1].imshow(img_LRP);arr[1].axis('off')
                img_RGB[:,:,0]=img_RGB[:,:,2]+VIZ.addBorder(np.multiply(pImg_LRP<alpha,img))
                arr[2].imshow(img_RGB);arr[2].axis('off')
                plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
                plt.savefig(str(ITERATION)+"_"+str(K)+"MODE"+
                            str(MODE)+"_"+CLASS+"_LRP.pdf",bbox_inches = 'tight',pad_inches = 0)
                os.system("pdfcrop --margins '0 0 0 0' --clip "+
                        str(ITERATION)+"_"+str(K)+"MODE"+str(MODE)+"_"+CLASS+"_LRP.pdf "+
                        str(ITERATION)+"_"+str(K)+"MODE"+str(MODE)+"_"+CLASS+"_LRP.pdf ")
                #Grad-CAM
                f, arr = plt.subplots(1,3)  #All 3 in a row
                arr[0].imshow(img, cmap='gray');arr[0].axis('off')
                arr[0].text(5,30, CLASS,fontsize=20,color='red')
                arr[1].imshow(img_GRD_raw);arr[1].axis('off')
                img_RGB2[:,:,0]=img_RGB2[:,:,2]+VIZ.addBorder(np.multiply(pImg_GRD<alpha,img))
                arr[2].imshow(img_RGB2);arr[2].axis('off')
                plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
                plt.savefig(str(ITERATION)+"_"+str(K)+"MODE"+
                            str(MODE)+"_"+CLASS+"_GRAD.pdf",bbox_inches = 'tight',pad_inches = 0)
                os.system("pdfcrop --margins '0 0 0 0' --clip "+
                        str(ITERATION)+"_"+str(K)+"MODE"+str(MODE)+"_"+CLASS+"_GRAD.pdf "+
                        str(ITERATION)+"_"+str(K)+"MODE"+str(MODE)+"_"+CLASS+"_GRAD.pdf ")
                #--- Prepare next loop ---#
                looper=looper+1
            #--------------------#
            #--- Fuse figures ---#
            #--------------------#
            os.system("pdfjam "+ 
                       str(ITERATION)+"_"+str(K)+"MODE"+str(MODE)+"_"+populations[0]+"_GRAD.pdf "+
                       str(ITERATION)+"_"+str(K)+"MODE"+str(MODE)+"_"+populations[1]+"_GRAD.pdf "+
                       str(ITERATION)+"_"+str(K)+"MODE"+str(MODE)+"_"+populations[2]+"_GRAD.pdf "+
                       str(ITERATION)+"_"+str(K)+"MODE"+str(MODE)+"_"+names[3]+"_GRAD.pdf "+
                       str(ITERATION)+"_"+str(K)+"MODE"+str(MODE)+"_"+populations[4]+"_GRAD.pdf "+
                       str(ITERATION)+"_"+str(K)+"MODE"+str(MODE)+"_"+populations[5]+"_GRAD.pdf "+
                        "--nup 1x6 --landscape --outfile "+ 
                        str(ITERATION)+"_"+str(K)+"MODE"+str(MODE)+"_GRAD.pdf ")
            os.system("pdfcrop --margins '0 0 0 0' --clip "+
                        str(ITERATION)+"_"+str(K)+"MODE"+str(MODE)+"_GRAD.pdf "+
                        str(ITERATION)+"_"+str(K)+"MODE"+str(MODE)+"_GRAD.pdf")
            os.system("pdfjam "+ 
                       str(ITERATION)+"_"+str(K)+"MODE"+str(MODE)+"_"+populations[0]+"_LRP.pdf "+
                       str(ITERATION)+"_"+str(K)+"MODE"+str(MODE)+"_"+populations[1]+"_LRP.pdf "+
                       str(ITERATION)+"_"+str(K)+"MODE"+str(MODE)+"_"+populations[2]+"_LRP.pdf "+
                       str(ITERATION)+"_"+str(K)+"MODE"+str(MODE)+"_"+names[3]+"_LRP.pdf "+
                       str(ITERATION)+"_"+str(K)+"MODE"+str(MODE)+"_"+populations[4]+"_LRP.pdf "+
                       str(ITERATION)+"_"+str(K)+"MODE"+str(MODE)+"_"+populations[5]+"_LRP.pdf "+
                        "--nup 1x6 --landscape --outfile "+ 
                        str(ITERATION)+"_"+str(K)+"MODE"+str(MODE)+"_LRP.pdf ")
            os.system("pdfcrop --margins '0 0 0 0' --clip "+
                        str(ITERATION)+"_"+str(K)+"MODE"+str(MODE)+"_LRP.pdf "+
                        str(ITERATION)+"_"+str(K)+"MODE"+str(MODE)+"_LRP.pdf")
            #--- Clear iteration ---#
            for pop in populations:
                os.system("rm *"+pop+"*.pdf")
            #End k loop
            break
        #end ITERATION loop
        break
    os.system("mkdir bestModels")
    os.system("mv *MODE*pdf bestModels/.")
