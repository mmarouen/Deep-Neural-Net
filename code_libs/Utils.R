#make  predictions from a generated model and preprocessed input data
predictNN<-function(model,#model input
                    X #input test matrix
                   ){
  L=model$argV
  out1=forwardPropagate(model$W,model$b,X,L$outF,L$active,L$bnVars,model$popStats)
  yhatTest=transformOutput(out1$Y,L$tt,model$CL)$yhat
  return(yhatTest)
}

#softmax output function
softmax<-function(X){
  eps=1e-15
  Eps=1-eps
  M=max(X)
  product=apply(X,2,function(x) exp(-M-log(rowSums(exp(X-M-x)))))
  product=pmax(product,eps)
  product=pmin(product,Eps)
  return(product)
}

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
