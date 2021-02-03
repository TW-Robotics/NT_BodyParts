#This file converts the GPC results to PS format
rm(list = ls())    #Remove old variables
library(readr)
library(ggplot2)
#----------------------#
#--- Landmark names ---#
#----------------------#
landmark.name <- c(
  "UTP",
  "EYE",
  "AOD",
  "POD",
  "DIC",
  "VOC",
  "PIA",
  "BPF",
  "PEO",
  "VEO",
  "AOA",
  "AOP",
  "HCF",
  "EMO"
)
#------------------------#
#--- Global functions ---#
#------------------------#
#-------------------------------------------------#
# Function comblscale(...)                        #
# Descr: This function estimates a diagonal cov.  #
#       insteat of the LS values combing from the #
#       GPC for X and Y coordinates. We estimate  #
#       here a 'common' LS val for both Procrustes#
#       coordinates.                              #
# Param:  design data                             #
#         LS values                               #
# Return: common LS values                        #
#-------------------------------------------------#
comblscale <- function(design, LS){
  converted.LS <- matrix(0,nrow(LS)/2,ncol(LS)) #Same number of classes but half features
  for(D in seq(1,ncol(LS))){
    looper <- 1
    for(features in seq(1,28,2)){
      #--- Core from Peter Sykacek's Python script ---#
      LS.loc <- diag(LS[c(features:(features+1)),D]) #Get current LS pair as diag matrix
      product.L <- (design[,c(features:(features+1))]%*%LS.loc)*design[,c(features:(features+1))]
      prodcut.L.rowSum <- rowSums(product.L)
      product.l <- (design[,c(features:(features+1))])*design[,c(features:(features+1))]
      product.l.rowSum <- rowSums(product.l)
      l <- sum(prodcut.L.rowSum) / sum(product.l.rowSum^2)
      #--- Store new lengthscale as relevance ---#
      converted.LS[looper,D] <- l
      looper <- looper +1
    }
  }
  return(converted.LS)
}
#-------------------------------------------------#
# Function: returnPieData(...)                    #
# Descr: This function calculates the relevance of#
#       each position. In contrast to             #
#       pieData.getFeatureRanking, this function  #
#       focuses on the ranken, not the features.  #
# Param: ranking      Ranking matrix perexperiment#
# Return: plot pie data                           #
#-------------------------------------------------#
#ranking \in [#feature x nr tries]
pieData.getPositionRanking <- function(ranking,top.N=3){
  pie.data <- matrix(0,nrow = top.N, ncol = nrow(ranking))
  #--- We initially loop over the top N ranks ---#
  for(rank in seq(1,top.N)){
    #--- Check each feature -> how many times were this feature on rank position rank? ---#
    for(feature in seq(1,nrow(ranking))){
      pie.data[rank,feature] <- length(which(ranking[feature,] == rank) )
    }
  }
  return(pie.data)
}
#-----------------------#
#--- Main processing ---#
#-----------------------#
tries <- seq(0,9)  #Iterations
classes <- 6 #Number of classes
top.N <- 5
#-------------------------------------#
#--- Define classifiers to analyse ---#
#-------------------------------------#
is.ProcrustesLandmark <- T
with.replace <- T
Selection <- suppressMessages(as.matrix(read_csv("/home/mluser/git/NT_BodyParts/Data/unselectedFeatures.csv", col_names = FALSE)))
#--- Get procrustes design data for LS conversion ---#
procrustes.design.data.raw <- suppressMessages(as.matrix(read_csv("/home/mluser/git/NT_BodyParts/Procrustes/PROCRUSTES_DATA.csv", col_names = FALSE)))
procrustes.design.data.str <- procrustes.design.data.raw[,3:30] #Get Procrustes LM
procrustes.design.data <- t(apply(procrustes.design.data.str, 1, as.numeric))
shuffel.data <- suppressMessages(as.matrix(read_table2("/home/mluser/git/NT_BodyParts/Data/NoReplace_20205809_085859.csv", col_names = FALSE)))
BGPLVM.noReplace.manual      <- c("BGPLVM_noResampling_manual",     "/home/mluser/git/NT_BodyParts/GPC/GPLVM/",     14,!is.ProcrustesLandmark, !with.replace)
BGPLVM.noReplace.top14       <- c("BGPLVM_noResampling_top14",      "/home/mluser/git/NT_BodyParts/GPC/GPLVM_full/",  14,!is.ProcrustesLandmark, !with.replace)
Procrustes.noReplace.manual  <- c("Procrustes_noResampling",        "/home/mluser/git/NT_BodyParts/GPC/Procrustes/", 14, is.ProcrustesLandmark, !with.replace)
#--- Create list to process ---#
processor <- rbind( BGPLVM.noReplace.manual,
                    BGPLVM.noReplace.top14,
                    Procrustes.noReplace.manual
                    )
#-------------------------#
#--- To pie generation ---#
#-------------------------#
ggplot.piePlot <- T
for(i in seq(1,nrow(processor))){
  memory.relevance <- array(0,dim = c(processor[i,3],length(tries)))
  for(n in tries){
    rel.local <- suppressMessages(as.matrix(read_csv(paste("./data/data_",processor[i,1],"/",n,"_Relevance.csv",sep = ""))))
    #------------------------------------#
    #--- Combine LS Procrustes values ---#
    #------------------------------------#
    if(as.logical(processor[i,4])){
      cat("Convert Procrustes LS vals\n")
      LS.local <- 1/rel.local #Re-estimate LS values
      #--- re-create design data ---#
      if(as.logical(processor[i,5])){
        #--- With replacement ---#
        index <- replace.data[,n+1]
      }else{
        #--- Without replacement ---#
        index <- shuffel.data[,n+1]
      }
      design.Procrustes <- procrustes.design.data[index,]
      ls.combined <- comblscale(design.Procrustes, t(LS.local))
      rel.local <- 1/ls.combined
      memory.relevance[,n+1] <- rel.local
    }else{
      #------------------------------------------------------------#
      #--- GP-LVM relevance data can be used without conversion ---#
      #------------------------------------------------------------#
      memory.relevance[,n+1] <- rel.local
    }
  }
  memory.relevance <- t(memory.relevance)
  ranking.per.exp <- apply(memory.relevance,1,function(x){return(order(x,decreasing = T))}) #Rank data
  ranking.pie.data.position <- pieData.getPositionRanking(ranking.per.exp,top.N = top.N) #Get pie plot info
  write.table(ranking.pie.data.position, paste("ranking_",processor[i,1],".csv",sep = ""), row.names = F,col.names = F)
  #---------------------#
  #--- Plot pie char ---#
  #---------------------#
  #if(grepl("manual",processor[i,1])){
  #  #--- Adapt colors to data ---#
  #  pie.rank.cols <- rainbow(max(Selection)) #One color for each feature
  #  pie.rank.cols <- pie.rank.cols[Selection]
  #}else{
  #  #--- Typical colors ---#
    pie.rank.cols <- rainbow(nrow(ranking.per.exp)*2) #One color for each feature
  #}

  #--- Prepare data ---#
  for(k in seq(1,top.N)){
    #--------------------#
    #--- Prepare plot ---#
    #--------------------#
    pdf(paste(processor[i,1],"_R",k,"_pie.pdf",sep = ""))
    #if(ggplot.piePlot){
    #  par(family='serif') 
    #  par(mfrow = c(top.N,1), 
    #      mai = c(0.1, 0.3, 0.1, 0.3),
    #      oma=c(0,0,3,0))
    #}
    #--------------------#
    #--- Get pie data ---#
    #--------------------#
    index.notZero <- which(ranking.pie.data.position[k,] > 0) #See which feature hat relevance on this rank
    data.pie <- c()
    label.pie <- c()
    color.pie <- c()
    #--- Create pie data for each feature with relevance ---#
    for(m in index.notZero){
      data.pie <- c(data.pie, ranking.pie.data.position[k,m])
      #--- Depending on features, other info is displayed ---#
      if(as.logical(processor[i,4])){   #Check if Procrustes features
        label.pie <- c(label.pie, landmark.name[m])  
      }else{
        if(grepl("manual",processor[i,1])){
          #--- Manually selected ---#
          label.pie <- c(label.pie, paste("F",Selection[m],sep = ""))
        }else{
          #--- Raw features -> no selection ---#
          label.pie <- c(label.pie, paste("F",m-1,sep = ""))
        }
      }
      color.pie <- c(color.pie, pie.rank.cols[m])
    }
    #--------------------------#
    #--- ggplot needs more! ---#
    #--------------------------#
    if(ggplot.piePlot){
      for(m in seq(1,length(ranking.pie.data.position[k,]))[-index.notZero]){
        data.pie <- c(data.pie, 0)
        #--- Depending on features, other info is displayed ---#
        #if(as.logical(processor[i,4])){   #Check if Procrustes features
          label.pie <- c(label.pie, "")#landmark.name[m])  
        #}else{
        #  if(grepl("manual",processor[i,1])){
        #    #--- Manually selected ---#
        #    label.pie <- c(label.pie, paste("F",Selection[m],sep = ""))
        #  }else{
        #    #--- Raw features -> no selection ---#
        #    label.pie <- c(label.pie, paste("F",m-1,sep = ""))
        #  }
        #}
        color.pie <- c(color.pie, pie.rank.cols[m])
      }
    }#End add ggplot data
    #--------------------#
    #--- Reorder data ---#
    #--------------------#
    index <- order(data.pie)
    data.pie <- data.pie[index]
    label.pie <- label.pie[index]
    color.pie <- color.pie[index]
    #--------------------#
    #--- Do pie plots ---#
    #--------------------#
    if(!ggplot.piePlot){
      #png(paste(processor[i,1],"_Rank",k,"_pie_2.png",sep = ""))
      pie(data.pie,labels = label.pie, 
          main = paste("Rank ",k,sep = ""),
          col=color.pie, cex.main=2.5, cex=2.5)  
    #dev.off()
    }else{
    #--- Reorder stuff ---#
      cat(data.pie,"\n")
      if(as.logical(processor[i,4])){
        #--- Procrustes cols ---#
        list.col <- c("UTP" = pie.rank.cols[1],
                      "EYE" = pie.rank.cols[2],
                      "AOD" = pie.rank.cols[3],
                      "POD" = pie.rank.cols[4],
                      "PEO" = pie.rank.cols[5],
                      "VEO" = pie.rank.cols[6],
                      "AOA" = pie.rank.cols[7],
                      "AOP" = pie.rank.cols[8],
                      "HCF" = pie.rank.cols[9],
                      "EMO" = pie.rank.cols[10],
                      "BPF" = pie.rank.cols[11],
                      "DIC" = pie.rank.cols[12],
                      "VOC" = pie.rank.cols[13],
                      "PIA" = pie.rank.cols[14])
      }else{
        list.col <- c()
        list.names <- c()
        for(looper in seq(0,14)){
          list.col <- c(list.col, pie.rank.cols[looper+1])
          list.names <- c(list.names, paste("F",looper,sep = ""))
        }
        names(list.col) <- list.names
      }
      DF.pie.data <- data.frame(value = (data.pie), group = as.factor(label.pie)) #Has to be factors
      pie.plot2 <- ggplot(DF.pie.data, aes(x = "", y = value, fill = group)) +
        geom_bar(width = 1, stat = "identity", color = "white") +
        coord_polar("y", start = 0)+
        #scale_fill_manual(values = (color.pie)) +
        scale_fill_manual(values = list.col)+
        theme_void()+
        geom_text(aes(label = (group)), 
                      size=12, 
                      position = position_stack(vjust = 0.5))+
        theme(legend.position = "none")+
        ggtitle(paste("Rank ",k,sep = ""))+  theme(plot.title = element_text(hjust = 0.5,size = 40))
        plot(pie.plot2)
    }
    dev.off()
    #-----------------------#
    #--- Store relevance ---#
    #-----------------------#
    
  }
    system(paste("pdfjam ",  
                 processor[i,1],"_R",1,"_pie.pdf ",
                 processor[i,1],"_R",2,"_pie.pdf ",
                 processor[i,1],"_R",3,"_pie.pdf ",
                 processor[i,1],"_R",4,"_pie.pdf ",
                 processor[i,1],"_R",5,"_pie.pdf ",
                 "--nup 5x1 --landscape --outfile ",processor[i,1],"_pie.pdf",sep = ""))
  system(paste("pdfcrop --margins '0 0 0 0' --clip ",processor[i,1],"_pie.pdf"," ",processor[i,1],"_pie.pdf",sep = ""))
  system("rm *_R*_pie.pdf") #Remove old results
  #break
}
system("mkdir GPCData")
system("mv *ranking*csv GPCData/.")