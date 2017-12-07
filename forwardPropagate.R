#performs forward propagation algorithm
#### Return
#output=updated neurons outputs by layer
#Z=scalar product within each neuron

forwardPropagate<-function(weight, #network weights
                           biases, #biases
                           X, #input matrix
                           outF="Identity", #output layer activation
                           activation="sigmoid", #hidden layers activation
                           bnVars, #batch normalization variables
                           popStats=NULL #containing population statistics (mean, variance) by layer
                          ){
  
  #load FP parameters
  L=length(weight)
  BN=bnVars$BN
  gammas=bnVars$gammas
  betas=bnVars$betas
  #init FP response
  Y=list()
  Y[[1]]=as.matrix(X)
  Z_hat=list()
  Z_hat[[1]]="0"
  sigma2=list()
  sigma2[[1]]="0"
  mu=list()
  mu[[1]]="0"
  for(i in 2:L){
    if(!BN) Z=t(t(Y[[i-1]]%*%as.matrix(weight[[i]]))+biases[[i]])
    if(BN){ #batch normalization layer
      Z=Y[[i-1]]%*%as.matrix(weight[[i]])
      mu[[i]]=colMeans(Z)
      mu_Z=mu[[i]]
      if(!is.null(popStats)) mu_Z=popStats$mu[[i]]
      demean=t(t(Z)-mu_Z)
      sigma2[[i]]=colMeans(demean^2)+1e-6
      sigma2_Z=sigma2[[i]]
      if(!is.null(popStats)) sigma2_Z=popStats$sigma2[[i]]
      Z_hat[[i]]=t(t(demean)/sqrt(sigma2_Z))
      Z=t(t(Z_hat[[i]])*gammas[[i]]+betas[[i]])
    }
    if(i<L){
      if (activation=="sigmoid") Y[[i]]=1/(1+exp(-Z))
      if (activation=="tanh") Y[[i]]=1.7159*tanh((2/3)*Z)
      if(activation=="linear") Y[[i]]=Z
      if(activation=="ReLU"){
        Y[[i]]=as.matrix(Z)
        Y[[i]][Y[[i]]<0]=0
      }
    }
    if(i==L){
      if(outF=="Identity") Y[[L]]=Z
      if(outF=="Tanh") Y[[L]]=tanh(Z)
      if(outF=="Softmax"){
        K=ncol(Z)
        if(K==1) Y[[L]]=1/(1+exp(-Z))
        if(K>2) Y[[L]]=softmax(Z)
      }
    }
  }
  popStats=list(mu=mu,sigma2=sigma2)
  return(list(Y=Y,Z_hat=Z_hat,popStats=popStats))
}
