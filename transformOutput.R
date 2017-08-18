#transform NN output vector/matrix into factor
transformOutput<-function(ZZ,tt="Regression",activation="sigmoid",classes){
  if(tt=="Regression"){
    yhat=ZZ[[length(ZZ)]]
    yhatMat=yhat
  }
  if(tt=="Classification"){
    yhatMat=ZZ[[length(ZZ)]]
    K=ncol(as.matrix(yhatMat))
    if(K==1){
      yhat=rep(0,length=nrow(yhatMat))
      yhat[yhatMat>=0.5]=1
      yhat=as.factor(yhat)
    }
    if(K>2){
      yhat=apply(yhatMat,1,function(x) classes[which.max(x)])
      yhat=as.factor(yhat)
    }
  }
  return(list(yhat=yhat,yhatMat=yhatMat))
}
