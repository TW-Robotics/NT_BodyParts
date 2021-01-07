# The GPC_nfCV.py script implements a GPy based GPC with n-fold
# cross-validation. The class is called with a design matrix,
# a label vector and the number of folds. The run method creates
# the data structures for each fold including the GPy GPC data
# structures. The GPC is created and evaluated. The results of 
# each fold is stored in a CSV file. The summarized result is 
# returned.
#
# This code is available under a GPL v3.0 license and comes without
# any explicit or implicit warranty.
#
# (C) Wilfried Wöber 2020 <wilfried.woeber@technikum-wien.at>

import numpy as np                                  	#Numpy
from sklearn.preprocessing import StandardScaler    	#Scale input data
from sklearn.metrics import confusion_matrix        	#Confusion matrix
from sklearn.metrics import accuracy_score		#Classification accuracy
import GPy		#GPy for GPC
#-----------------#
#--- The class ---#
#-----------------#
class GPC_nfCV:
    def __init__(self, X,Y,n_fold):
        self.X = X              #Features
        self.Y = Y              #Labels (binary)
        self.n_fold = n_fold    #How many folds
        if(np.min(self.Y) < 0):
            raise Exception("Check target (<0)") 
    def run(self, experimentPrefix=""):
        #-----------------------------#
        #--- Print system settings ---#
        #-----------------------------#
        print("|-----------------------------------|")
        print("|--- Run n-fold cross validation ---|")
        print("|-----------------------------------|")
        print("n = %d " % self.n_fold)
        print("X [%d x %d]" % (self.X.shape[0], self.X.shape[1]))
        print("Y [%d x 1]" %  self.Y.shape[0])
        print("Number of classes: %d " % (np.max(self.Y)+1))
        #-----------------------------#
        #--- Variables for running ---#
        #-----------------------------#
        fold_iterator = np.ceil(float(self.X.shape[0])/float(self.n_fold))     #We get the index iteration number here
        #-----------------------------#
        #--- Memory for iterations ---#
        #-----------------------------#
        GPC_best_LS = 0     #Best lengthscale values
        GPC_best_LIN = 0    #Best linear values
        GPC_best_ACC = -1   #Best accuracy
        GPC_memory_ACC = np.zeros((self.n_fold,1))   #Memory to store n fold cross validation accuracy
        #-----------------------------------------#
        #--- Prepare matrices to store results ---#
        #-----------------------------------------#
        SUM_PRED_Y      = np.zeros((self.X.shape[0],1))*(-1) #Memory for predicted labels
        SUM_PRED_Y_PROB = np.zeros((self.X.shape[0],6))*(-1) #Memory for predicted class probability
        #-----------------------------------#
        #--- Run n fold cross validation ---#
        #-----------------------------------#
        for i in range(0,self.n_fold):
            print("Fold %d:"%i)
            index_start = int(i*fold_iterator)          #Get start point
            index_end = int(index_start+fold_iterator)  #Increment using theoretical sampling
            if(index_end > (self.X.shape[0]-1)):        #Check if everything is fine
                index_end = self.X.shape[0]-1           #In case of my bad math
            #print("Index:", index_start, " - ", index_end)
            #----------------------#
            #--- For each class ---#
            #----------------------#
            class_p = np.zeros((1+index_end-index_start,6))     #Memory for class probabilities
            ls_memory = np.zeros((6,self.X.shape[1]))
            for k in np.array((0,1,2,3,4,5)):
                print("|- Class %d"% k)
                y_train_bin = np.zeros(self.Y.shape[0]).reshape((-1,1))
                for j in range(0,self.Y.shape[0]):
                    if(self.Y[j] == k):
                        y_train_bin[j]=1
                #---------------------#
                #--- Cut out index ---#
                #---------------------#
                X_train = np.delete(self.X,range(index_start,index_end+1),0)
                X_test = self.X[index_start:index_end+1,:]
                Y_train = np.delete(y_train_bin,range(index_start,index_end+1)).reshape((-1,1))
                Y_test = y_train_bin[index_start:index_end+1].reshape((-1,1))
                #print("Train size: ", X_train.shape)
                #print("Test size: ", Y_train.shape)
                GPC_C=GPC(X_train,Y_train,X_test,Y_test)
                GPC_C.fit()
                class_p[:,k]=GPC_C.prediction_label
                ls_memory[k,:]=GPC_C.GPC_LS
            #--- Normalize probability values ---#
            for R in range(0,class_p.shape[0]): #For ech row in the probability matrix
                class_p[R,:] = class_p[R,:]/np.sum(class_p[R,:])    #Scale to 1
            #--- Gett CM ---#
            y_true_pred = np.zeros(Y_test.shape[0]).reshape((-1,1)) #Memory for label with highest GPC prob
            y_true = self.Y[index_start:index_end+1]        #Real labels from randomly selected test set of folf
            for k in range(0,y_true_pred.shape[0]):         #Get maximum prob as a label
                y_true_pred[k] = np.argmax(class_p[k,:])
            #print(confusion_matrix(y_true_pred, y_true))
            print("|- Accuracy: %f" % accuracy_score(y_true_pred, y_true))
            #------------------#
            #--- Store data ---#
            #------------------#
            GPC_memory_ACC[i]=accuracy_score(y_true_pred, y_true)
            #---------------------#
            #--- Store results ---#
            #---------------------#
            np.savetxt(experimentPrefix+"_foldNr_"+str(i)+"_CM.csv",confusion_matrix(y_true_pred, y_true))
            np.savetxt(experimentPrefix+"_foldNr_"+str(i)+"_LS.csv",ls_memory)
            #------------------------------------#
            #--- store results in data matrix ---#
            #------------------------------------#
            SUM_PRED_Y[index_start:index_end+1] = y_true_pred
            SUM_PRED_Y_PROB[index_start:index_end+1,:] = class_p
        #--- End n fold cross validation ---#
        print("|-----------------------------|")
        print("|--- End cross validationa ---|")
        print("|-----------------------------|")
        print("Mean accuracy: %f" % np.mean(GPC_memory_ACC))
        np.savetxt(experimentPrefix+"_AccMemory.csv",GPC_memory_ACC)
        return([np.mean(GPC_memory_ACC),SUM_PRED_Y, SUM_PRED_Y_PROB ])
class GPC:
    def __init__(self,X_train,Y_train,X_test,Y_test):
        self.X_train=X_train    #Get design data
        self.Y_train=Y_train    #Get BINARY labels
        self.X_test=X_test      #Get design data
        self.Y_test=Y_test      #Get BINARY labels
    def fit(self):
        while(True):
            #--- Create GPC model ---#
            kernel = GPy.kern.RBF(self.X_train.shape[1],ARD=1) +  GPy.kern.Bias(self.X_train.shape[1])  #Define kernel
            m = GPy.models.GPClassification(self.X_train, self.Y_train, kernel=kernel)      #Define GPC
            #--- Optimize ---#
            ret = m.optimize(optimizer='lbfgs')
            if(ret.status!="Errorb'ABNORMAL_TERMINATION_IN_LNSRCH'"):
                break
        #--- Get labels ---#
        self.prediction_label = m.predict(self.X_test)[0].reshape((-1,1))[:,0]
        self.GPC_LS = m.kern['sum.rbf.lengthscale'][:] 
        #self.GPC_LIN = m.predict(X_testSplit)[0].reshape((-1,1))[:,0]
