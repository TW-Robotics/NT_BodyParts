#This file converts the GPC results to PS format
rm(list = ls())    #Remove old variables
library(readr)
#------------------------#
#--- Global functions ---#
#------------------------#
mkdir <- function(classifier){
  system(paste("mkdir data/data_",classifier[1],sep = ""))
}
getdir <- function(classifier){
  return(paste("data/data_",classifier[1],sep = ""))
}
isCNN <- function(classifier){
  return(as.logical(classifier[3]))
}
#Converts loaded data to PS format
convertFormat <- function(trainM){
  #sample.names <- trainM[,1] #Getnames of samples
  true.class <- as.numeric(trainM[,1])   #Get true class
  pred.class <- as.numeric(trainM[,2])   #Get predicted class
  pred.prob.raw <- trainM[,3:(3+number.classes-1)] #Get prediction probabilities
  pred.prob.numeric <- apply(pred.prob.raw,1,as.numeric) #Convert to numbers
  pred.prob <- t(apply(pred.prob.numeric, 2, function(x){return(x/sum(x))})) # Normalize probabilities
  #according to https://github.com/scikit-learn/scikit-learn/blob/0fb307bf3/sklearn/multiclass.py#L346
  return(list(
              #sample.names = sample.names, 
              true.label = true.class, 
              predicted.label = pred.class, 
              class.prob = pred.prob))
}
#Creates file to write
createFileData <- function(trainM){
  data.matrix.PS <- cbind(trainM$class.prob, trainM$predicted.label, trainM$true.label)
  colnames(data.matrix.PS) <- c(unlist(lapply(seq(0,number.classes-1),function(x){paste("probs",x,sep = "")})), "ptarg","ttarg")
  return(data.matrix.PS)
}
#Convert Relevance vector to PS format
convertARD <- function(RelVector){
  dim(RelVector) <- c(1,length(RelVector))
  colnames(RelVector) <- unlist(lapply(seq(1,length(RelVector)),function(x){return(paste("ard_",x,sep = ""))}))
  return(RelVector)
}
# The target format is: probs0, ..., probsn,ptarg,ttarg
number.classes <- 6 #Number og classes for GPC
k.fold <- seq(0,9)  #sequence of k's
iterations <- 10    #How many times repeated
isNoCNN <- F        #Flag indicates if classifier is CNN or not
#BGPLVM.replace.manual        <- c("BGPLVM_resampling_manual",       "../GPC/BGPLVM_replace/",       isNoCNN)
BGPLVM.noReplace.manual      <- c("BGPLVM_noResampling_manual",     "/home/mluser/git/NT_BodyParts/GPC/GPLVM/",     isNoCNN)
#BGPLVM.replace.top14         <- c("BGPLVM_resampling_top14",        "../GPC/BGPLVM_replace_14/",    isNoCNN)
BGPLVM.noReplace.top14       <- c("BGPLVM_noResampling_top14",      "/home/mluser/git/NT_BodyParts/GPC/GPLVM_full/",  isNoCNN)
#Procrustes.replace.manual    <- c("Procrustes_resampling",          "../GPC/Procrustes_replace/",   isNoCNN)
Procrustes.noReplace.manual  <- c("Procrustes_noResampling",        "/home/mluser/git/NT_BodyParts/GPC/Procrustes/", isNoCNN)
#CNN.replace.augmented        <- c("CNN_Aug_resampling",             "../CNN_augmentation/CNN_replace/",   !isNoCNN)
CNN.noReplace.augmented      <- c("CNN_Aug_noResampling",           "/home/mluser/git/NT_BodyParts/CNN/Classification/Augmented/", !isNoCNN)
#--- summarize classifier ---#
processor <- rbind(#BGPLVM.replace.manual,
                   BGPLVM.noReplace.manual,
                   #BGPLVM.replace.top14,
                   BGPLVM.noReplace.top14,
                   #Procrustes.replace.manual,
                   Procrustes.noReplace.manual,
                   #CNN.replace.augmented,
                   CNN.noReplace.augmented
                   )
system("rm -rf data/data*") #RM previous classifier results
#---------------------#
#--- Do processing ---#
#---------------------#
for(i in seq(1,nrow(processor))){ #Process all classifiers
  mkdir(processor[i,]) #Create directory for current classifier
  #---------------------------------#
  #--- Get classification result ---#
  #---------------------------------#
  REL.memory <- c()
  for(ITERATION in seq(0,iterations-1)){
    #if(!isCNN(processor[i,])){
    #  classifier.result <- suppressMessages(as.matrix(read_csv(paste(processor[i,2],ITERATION,"__dataMatrix.csv",sep = ""), col_names = FALSE)))
    #}else{
      classifier.result <- suppressMessages(as.matrix(read_csv(paste(processor[i,2],ITERATION,"_dataMatrix.csv",sep = ""))))
      #stop("Not implemented yet")
    #}
    #--- Get format ---#
    classifier.result.format <- convertFormat(classifier.result) #Convert to numeric and a list
    classifier.data_for_file <- createFileData(classifier.result.format) #Convert to PS format
    #--- Store ---#
    write.table(classifier.data_for_file, paste(getdir(processor[i,]),"/",ITERATION,"_classifierResult.csv",sep = ""),row.names = F,quote = F,sep = ',')
    #----------------------------------------------------------#
    #--- Get ls values and prepare for relevance processing ---#
    #----------------------------------------------------------#
    if(!isCNN(processor[i,])){
      REL.memory.local <- c() #Memory for LS values
      REL.memory.per_Population <- c()
      cat("Get GPC relevance for ", processor[i,1],"\n")
      for(k in k.fold){
        LS.k <- suppressMessages(as.matrix(read_table2(paste(processor[i,2],ITERATION,"_foldNr_",k,"_LS.csv",sep = ""), col_names = FALSE)))
        REL.k <- 1/LS.k
        REL.memory.local <- rbind(REL.memory.local,colMeans(REL.k))
        if(is.null(dim(REL.memory.per_Population))){
          REL.memory.per_Population <- REL.k
        }else{
          REL.memory.per_Population <- REL.memory.per_Population+REL.k
        }
      }
      REL.memory.per_Population <- REL.memory.per_Population / length(k.fold) #Normalize per-population relevance
      REL.memory <- rbind(REL.memory,colMeans(REL.memory.local)) # Just for later plotting
      write.table(convertARD(colMeans(REL.memory.local)), paste(getdir(processor[i,]),"/",ITERATION,"_Relevance.csv",sep = ""), row.names = F,quote = F,sep = ",")
      write.table(REL.memory.per_Population, paste(getdir(processor[i,]),"/",ITERATION,"_MeanRelevance_perPopulation.csv",sep = ""), row.names = F,quote = F,sep = ",")
    }#End if isCNN
  }#End Iteration loop
  #-------------------------------------#
  #--- Store results of REL analysis ---#
  #-------------------------------------#
  if(!isCNN(processor[i,])){
    pdf(paste(getdir(processor[i,]),"/Relevance.pdf",sep = ""))
    matplot(t(REL.memory),
            type='l', lty=1, col=seq(1,10), 
            xlab = "Features", ylab="Relevance",main=paste("Rel. Analysis for",processor[i,1]), 
            cex.main=0.75)
    abline(v = seq(0,40,1),h = seq(0,2,0.1),col='gray')
    dev.off()
    write.table(REL.memory, paste(getdir(processor[i,]),"/RelevanceSUM.csv",sep = ""),col.names = F, row.names = F,quote = F)
  }
}#End classifier loop