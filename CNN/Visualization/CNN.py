# Similar to the CNN.py script in the Classification folder,
# this script generates the data used for the CNN model.
# Different to the classification scripts, we are here interested 
# in generating visualization maps using Grad-CAM[1] and LRP[2].
# Initially and similar to the classification scripts, the images
# are generated. However instead of storing the classification results
# (e.g.: CSV files), we store here the visualization maps.
# 
# [1] R. R. Selvaraju, A. Das, R. Vedantam, M. Cogswell, D. Parikh, and D. Batra. 
#           Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based 
#           Localization. 
#           In 2017 IEEE International Conference on Computer Vision (ICCV), pages618–626, 2017.
# [2] S. Bach, A. Binder, G. Montavon, F. Klauschen, K. R. Müller, and W. Samek.
#           On Pixel-Wise Explanations for Non-Linear Classifier Decisions by Layer-Wise 
#           Relevance Propagation. PLoS ONE, 10(7):e0130140, 2015.
#
# This code is available under a GPL v3.0 license and comes without
# any explicit or implicit warranty.
#
# (C) Wilfried Woeber 2020 <wilfried.woeber@technikum-wien.at>
#-------------------------#
#--- Run main programm ---#
#-------------------------#
import numpy as np                      #Numpy ;)
import pandas as pd                     #We need that for novel data loading
import os                               #Folder handling for training data creation
import shutil                           #Recursivly folder delete, copy files
#import io
import sys
#-------------------------------#
#--- Some global definitions ---#
#-------------------------------#
n_fold=10                               #k for k-fold cross validation
path_images = "../../Data/images/"      #Path to images
#-----------------------------#
#--- Some global functions ---#
#-----------------------------#
#Removing existing folders
def rm_dirs():
    #-----------------------#
    #--- RM train folder ---#
    #-----------------------#
    if(os.path.isdir("./train")):
        print("rm train folder...")
        shutil.rmtree(os.path.abspath(os.getcwd())+"/train")
    if(os.path.isdir("./test")):
        print("rm test folder...")
        shutil.rmtree(os.path.abspath(os.getcwd())+"/test")
#Create class folders
def create_class(file_path,CLASS):
    try:
        os.mkdir(file_path)
    except OSError:
        print ("Cannot create folder %s" % file_path)
    else:
        print ("Created folder %s" % file_path)
    for CLASS in class_names:
        try:
            os.mkdir(file_path+CLASS)
        except OSError:
            print ("Cannot create train folder for %s" % file_path+CLASS)
        else:
            print ("Created train folder for %s " % file_path+CLASS)
#Move images to folders
def moveImages(IDs,folder,labels):
    #print("Move %d images" % IDs.shape[0])
    looper=0    #Looper for labels
    for ID in IDs:
        #print("Move %s to %s" % (ID,folder))
        #--- Check if its copy ---#
        prefix=""
        while(True):
            copy_path = folder+class_names[labels[looper]]+"/"+prefix+ID+".jpg"
            if(os.path.isfile(copy_path)):
                prefix=prefix+"copy_"
            else:
                #shutil.copy2(path_images+ID+".jpg", folder+class_names[labels[looper]]+"/"+ID+".jpg")   #Copy file to folder
                shutil.copy2(path_images+ID+".jpg", copy_path)   #Copy file to folder
                looper=looper+1
                break
#Convert CNN result in our format again
def convertData(CNNDataMatrix, ID_test):
    data_fin = np.ones((CNNDataMatrix.shape[0],1))*(-1)
    #--- Loop over all test IDs ---#
    looper=0
    for testID in ID_test:
        ID=-1
        #--- Find right row and store indices ---#
        #print("Check for %s" % testID)
        for i in range(0,CNNDataMatrix.shape[0]):   #Loop over all CNN test cases
            #print("Check %s" % CNNDataMatrix[i,0])
            matchedID = CNNDataMatrix[i,0].find(testID)
            if(matchedID >= 0):
                ID=i
                break
        #print("Match @ %d" % ID)
        data_fin[looper,:] = ID
        looper=looper+1
    return data_fin.astype(int) #Return indices for correction
#--------------------------------------#
#--- Load files from data generator ---#
#--------------------------------------#
class_names = np.genfromtxt("../../Data/classes.csv", dtype=str)                #Names of fishes
data_file = pd.read_csv('../../GPLVM/BGPLVM_DATA.csv', header=None).as_matrix() #Read data including sample ID
# The format is for each row: sampleID (string), label (int), data vector (double)
samples_iteration  = np.genfromtxt("../../Data/NoReplace_20205809_085859.csv", 
                                    delimiter=' ',dtype=int)                    #Load data from R script
#-------------------#
#--- Create data ---#
#-------------------#
sample_ID = data_file[:,0]                      #The sample ID
labels_raw = data_file[:,1]                     #Stored labels
data_raw = data_file[:,1:data_file.shape[1]]    #The label followed by data (BGPLVM features)
#------------------#
#--- Do looping ---#
#------------------#
for ITERATION in range (0,samples_iteration.shape[1]):
    #-----------------------#
    #--- Get specimen ID ---#
    #-----------------------#
    IDs = sample_ID[samples_iteration[:,ITERATION]]     #Get IDs for image filename generation
    labels = labels_raw[samples_iteration[:,ITERATION]] #Get randomized labels
    #-------------------------------#
    #--- k-fold cross validation ---#
    #-------------------------------#
    fold_iterator = np.ceil(float(IDs.shape[0])/float(n_fold))     #We get the index iteration number here
    data_memory = np.array(())
    data_memory_test = np.array(())
    test=np.array(range(0,209))
    for k in range(0,n_fold):
        #-------------------------------------#
        #--- Create train and test folders ---#
        #-------------------------------------#
        rm_dirs()                               #Remove existing folder
        create_class("./train/", class_names)   #Create train folders
        create_class("./test/", class_names)    #Create test folders
        #sys.stdout = sys.__stdout__
        #--- get sample names ---#
        index_start = int(k*fold_iterator)          #Get start point
        index_end = int(index_start+fold_iterator)  #Increment using theoretical sampling
        print("Indices %d - %d " % (index_start , index_end))
        if(index_end > (IDs.shape[0]-1)):           #Check if everything is fine
            index_end = IDs.shape[0]                #In case of my bad math
            print("Indices %d - %d " % (index_start , index_end))
        #--- Build sampel IDs ---#
        ID_train    = np.delete(IDs,range(index_start,index_end),0)
        ID_test     = IDs[index_start:index_end]
        labels_train= np.delete(labels,range(index_start,index_end),0)
        labels_test = labels[index_start:index_end]
        #--- Move images ---#
        moveImages(ID_train,"./train/", labels_train)  #Move to train folder
        moveImages(ID_test,"./test/", labels_test)     #Move to test folder
        #---------------#
        #--- Run CNN ---#
        #---------------#
        os.system("../../Python/VE_CNN/bin/python train_model.py 1 > logfile_AUG_"+str(ITERATION)+"_"+str(k)+"_.log")   #Run with data augmentation
        #os.system("../../Python/VE_CNN/bin/python train_model.py 1")   #Run with data augmentation
        rename_string="AUG_"+str(ITERATION)+"_"+str(k)     #String for map renaming
        os.system("mkdir "+rename_string)
        os.system("mv *.pdf "+rename_string+"/.")

        os.system("../../Python/VE_CNN/bin/python train_model.py 0 > logfile_nonAUG_"+str(ITERATION)+"_"+str(k)+"_.log")   #Run without data augmentation
        rename_string="noAUG_"+str(ITERATION)+"_"+str(k)     #String for map renaming
        os.system("mkdir "+rename_string)
        os.system("mv *.pdf "+rename_string+"/.")
