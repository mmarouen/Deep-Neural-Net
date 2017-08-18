#performs forward propagation algorithm
#### Parameters
#weight=list containg network weights
#X=input data
#tt="Regression" or "Classification"
#outF=output function
#activation=activation function
#### Return
#Z=updated neurons outputs by layer

forwardPropagate<-function(weight,X,tt="Regression",outF="Identity",activation="sigmoid"){
  L=length(weight)+1
  output=list()
  output[[1]]=as.matrix(cbind(1,X))
  for(i in 2:L){
    if(i<L){
      mat=as.matrix(weight[[i-1]])
      if (activation=="sigmoid"){
        output[[i]]=apply(mat,2,function(x) 1/(1+exp(-output[[i-1]]%*%x)))
      }
      if (activation=="tanh"){
        output[[i]]=apply(mat,2,function(x) 1.7159*tanh((2/3)*output[[i-1]]%*%x))
      }
      if(activation=="linear"){
        output[[i]]=output[[i-1]]%*%mat
      }
      output[[i]]=cbind(1,output[[i]])
    }
    if(i==L){
      mat0=output[[L-1]]%*%weight[[L-1]]
      if((outF=="Identity")&(tt=="Regression")){output[[L]]=mat0}
      if((outF=="Sigmoid")&(tt=="Regression")){output[[L]]=1/(1+exp(-mat0))}
      if((outF=="Tanh")&(tt=="Regression")){output[[L]]=tanh(mat0)}
      if((outF=="Softmax")&(tt=="Regression")){output[[L]]=softmax(mat0)}
      if((tt=="Classification") & (outF=="Softmax")){
        K=ncol(mat0)
        if(K==1){output[[L]]=1/(1+exp(-mat0))}
        if(K>2){output[[L]]=softmax(mat0)}
      }
    }
  }
  return(list(Z=output))
}
