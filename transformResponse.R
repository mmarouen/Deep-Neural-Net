#Utility function
#transform response vector according to binary & multinom classification
transformResponse<-function(resp,tt="Regression"){
  classes=NULL
  if(tt=="Regression"){respMat=as.matrix(resp)}
  if(tt=="Classification"){
    K=length(unique(resp))
    if(K==2){respMat=as.matrix(resp)}
    if(K>2){
      respMat=matrix(0,nrow=length(resp),ncol=K)
      classes=sort(unique(resp))
      for (i in 1:length(resp)){respMat[i,which(resp[i]==classes)]=1}
    }
  }
  return(list(respMat=respMat,CL=classes,response=resp))
}
