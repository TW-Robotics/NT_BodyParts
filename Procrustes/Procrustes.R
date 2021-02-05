# This script is the original implementation used in the publication
# for GPA estiamtion. 
#    
# The code in evalres.py is available under a GPL v3.0 license and
# comes without any explicit or implicit warranty.
##
# (C) W. Wöber 2020 <peter@sykacek.net> 
rm(list = ls())                                   #Remove old variables
library(readr)
library(shapes)
#---------------------#
#--- Get main info ---#
#---------------------#
class.names  <- suppressMessages(as.matrix(read_csv("../Data/classes.csv", col_names = FALSE))) #Class names
sample.IDs   <- suppressMessages(as.matrix(read_csv("../Data/targetID.csv", col_names = FALSE))) #Get sample ID 
#-----------------------------------------#
#--- Load data in same order as BGPLVM ---#
#-----------------------------------------#
landmark.dim <- 2 #Number of dimensions of landmarks
nr.landmarks <- 14 #Number of defined landmarks
procrustes.input.data <- array(0,dim=c(nr.landmarks,landmark.dim,length(sample.IDs)))
target <- suppressMessages(as.matrix(read_csv("../Data/Landmarks/rawLandmarks_MetaData.csv", col_names = FALSE)))
procrustes.input.data.X <- suppressMessages(as.matrix(read_csv("../Data/Landmarks/rawLandmarks_X.csv", col_names = FALSE)))
procrustes.input.data.Y <- suppressMessages(as.matrix(read_csv("../Data/Landmarks/rawLandmarks_Y.csv", col_names = FALSE)))
procrustes.input.data[,1,] <- procrustes.input.data.X
procrustes.input.data[,2,] <- procrustes.input.data.Y
#-----------------------------#
#--- Plot raw image points ---#
#-----------------------------#
min.val.x <- min(procrustes.input.data[,1,])
max.val.x <- max(procrustes.input.data[,1,])
min.val.y <- min(procrustes.input.data[,2,])
max.val.y <- max(procrustes.input.data[,2,])
pdf("./LandmarkPosition.pdf")
par(family='serif')
plot(NA, 
     xlim=c(min.val.x,max.val.x), ylim = c(min.val.y,max.val.y),
     xlab="X Coordinate",ylab = "Y Coordinate", main = "Raw landmarks"
)
abline(v = seq(-5000,5000,1000),h = seq(-5000,5000,500), col='gray')
for(n in seq(1,dim(procrustes.input.data)[1])){
  for(i in seq(1,dim(procrustes.input.data)[3])){
    points(procrustes.input.data[n,1,i],procrustes.input.data[n,2,i],pch=n, col=n, cex=0.5)
  }
}
dev.off()
#------------------------------#
#--- Do procrustes analysis ---#
#------------------------------#
eigen.analysis = T #(De)activate scaling of Procrustes
procrustes.output <- procGPA(procrustes.input.data, pcaoutput = T,eigen2d = eigen.analysis) #Do Procrustes analysis
#------------------#
#--- Store data ---#
#------------------#
#--- Prepare data ---#
data.GPC <- array(0,dim=c(length(sample.IDs),(nr.landmarks*2)+1+1))  #14*2 = 28 + label + ID = 29
for(i in seq(1,length(sample.IDs))){
  #We create a data vector, where the first element is the sample ID, the second the label and the remaining elements
  #are the X and Y procrustes coordinates
  data.GPC.local <- sample.IDs[i] #Place sample ID
  data.GPC.local <- c(data.GPC.local, target[i,2]) #Place label
  #--- Add remaining X and Y coordinates ---#
  for(k in seq(1,nr.landmarks)){
    data.GPC.local <- c(data.GPC.local, c(procrustes.output$rotated[k,1,i],procrustes.output$rotated[k,2,i]))
  }
  data.GPC[i,] <- data.GPC.local #Store data in GPC matrix
}
write.table(data.GPC, "./PROCRUSTES_DATA.csv",row.names = F,col.names = F, quote = F, sep = ",")
#--------------------#
#--- Draw results ---#
#--------------------#
#--- invert Y axis ---#
warning("Invert Y axis for plotting")
procrustes.output$mshape[,2] <- -procrustes.output$mshape[,2]
procrustes.output$rotated[,2,] <- -procrustes.output$rotated[,2,]
#--- Do processing ---#
min.val.x <- min(procrustes.output$rotated[,1,])
max.val.x <- max(procrustes.output$rotated[,1,])
min.val.y <- min(procrustes.output$rotated[,2,])
max.val.y <- max(procrustes.output$rotated[,2,])
#--- Plot procrustes ---#
pdf("./RawProcrustes.pdf")
par(family='serif')
plot(NA, 
     xlim=c(min.val.x,max.val.x), ylim = c(min.val.y,max.val.y),
     xlab="X Coordinate",ylab = "Inverted Y Coordinate"#, main = "Procrustes Analysis"
     )
if(!eigen.analysis){
  abline(v=seq(-2000,2000,by = 500),h = seq(-2000,2000,by = 250),col='gray')  
}else{
  abline(v=seq(-2,2,by = 0.1),h = seq(-2,2,by = 0.05),col='gray')
}

for(i in seq(1,length(sample.IDs))){
  for(k in seq(1,nr.landmarks)){
    points(procrustes.output$rotated[k,1,i],procrustes.output$rotated[k,2,i],
           pch='.', cex=2, col=target[i,2]
           )
  }
}
for(i in seq(1,nrow(procrustes.output$mshape))){
  if(!eigen.analysis){
    text(x = procrustes.output$mshape[i,1]-100,y = procrustes.output$mshape[i,2]-100, toString(i))
  }else{
    text(x = procrustes.output$mshape[i,1]-0.025,y = procrustes.output$mshape[i,2]-0.025, toString(i))
  }
}
#--- draw lines between centers ---#
center.abfolge <- c(1,2,1,3,4,5,13,6,7,11,12,10,8,9,8,10,14,1)
for(i in seq(2,length(center.abfolge))){
  lines(c(procrustes.output$mshape[center.abfolge[i-1],1],procrustes.output$mshape[center.abfolge[i],1]), 
        c(procrustes.output$mshape[center.abfolge[i-1],2],procrustes.output$mshape[center.abfolge[i],2]))
}
#--- legend ---#
legend("topright",legend=class.names, col = seq(1,6),cex=0.75,bg='white', lty=1)
dev.off()