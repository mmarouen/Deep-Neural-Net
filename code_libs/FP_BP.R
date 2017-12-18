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

#performs backpropagation algorithm
#### Output
#W=updated weights list
#D=gradient
#tr=trace weights update (if trW)
#gradupdate=gradient values update (if trW)

backPropagate<-function(Y, #network activations
                        weight, #network current weights
                        biases, #network current biases 
                        resp, #response vector
                        ll="RSS", #loss function
                        outF="Identity", #activation function for final layer
                        activation="sigmoid", #hidden units activation
                        rr, #learning rate
                        wD, #weight decay 
                        trW, #binary switch to trace weights
                        optimization='GD', #opt. alg. Options="GD","Momentum", "RMSProp","Adam"
                        beta1=0.9,beta2=0.99,# vanishing coefficients for Momentum & rmsprop
                        momentum=NULL,momentum_b=NULL,#momentum matrices
                        rmsprop=NULL,rmsprop_b=NULL, #rmsprop matrices
                        tol, #tolerence value
                        t, #epoch number
                        BNVars #batch normalization variables
                       ){
  #load parameters
  N=nrow(Y[[1]]) #number of observations
  K=ncol(resp) 
  L=length(weight)#total layers count
  BN=BNVars$BN
  Z_hat=BNVars$Z_hat
  sigma2=BNVars$sigma2
  gammas=BNVars$gammas
  betas=BNVars$betas
  #init lists
  delta=list()#delta[[1]]=NULL
  dltaHat=list()#dltaHat[[1]]=NULL
  gradRaw=list()
  gradRaw_b=list()
  grdRaw_gammas=list()
  grdRaw_betas=list()
  tmp=list()#new weights
  tmp_b=list()
  tmp_g=list()
  tmp_beta=list()
  tmp[[1]]=0
  tmp_b[[1]]=0
  tmp_g[[1]]=0
  tmp_beta[[1]]=0
  traceW=list(length=L-1)
  gradUpdate=list()
  
  for(i in L:2){#weights in layer i are one index below: Z[[i]]=f(Z[[i-1]]%*%W[[i-1]])
    if(i==L){#output layer
      mat0=Y[[L]]
      deltaL=mat0-resp
      if((ll=="RSS" & outF=="Identity")|(ll=="CrossEntropy" & outF=="Softmax")) dltaHat[[i]]=deltaL
      if(ll=="RSS" & outF=="Sigmoid") dltaHat[[i]]=mat0*(1-mat0)*deltaL
      if(ll=="RSS" & outF=="Tanh") dltaHat[[i]]=deltaL*(1-mat0^2)
      if(ll=="RSS" & outF=="Softmax"){
        mat1=matrix(0,ncol=ncol(mat0),nrow = nrow(mat0))
        mat2=deltaL
        for(p in 1:K){
          for(j in 1:K){
            mat1[,p]=mat1[,p]+mat0[,j]*mat2[,j]*(delt(p,j)-mat0[,p])
          }
        }
        dltaHat[[i]]=mat1
      }
    }
    if(i<=(L-1)){#BP to hidden layers
      mat1=Y[[i]]
      mat2=delta[[i+1]]%*%t(weight[[i+1]])
      if(activation=="tanh") dltaHat[[i]]=1.7159*(2/3)*(1-(mat1/1.7159)^2)*mat2
      if(activation=="sigmoid") dltaHat[[i]]=mat1*(1-mat1)*mat2
      if(activation=="linear") dltaHat[[i]]=mat2
      if(activation=="ReLU"){
        mat1[mat1>0]=1
        mat1[mat1==0]=0
        dltaHat[[i]]=mat1*mat2
      }
    }
    if(!BN) delta[[i]]=dltaHat[[i]]
    if(BN){
      Z=as.matrix(Z_hat[[i]])
      D=as.matrix(dltaHat[[i]])
      dg=colMeans(Z*D)
      db=colMeans(D)
      m1=t(t(Z)*dg)
      m1=D-m1
      m1=t(t(m1)-db)
      mult=gammas[[i]]/sqrt(sigma2[[i]])
      grdRaw_gammas[[i]]=dg
      grdRaw_betas[[i]]=db
      delta[[i]]=t(t(m1)*mult)
    }
    
    #gradient update
    gradRaw[[i]]=(1/N)*t(Y[[i-1]])%*%delta[[i]]
    gradRaw_b[[i]]=colMeans(delta[[i]])
    if(wD[[1]]){#regularization
      lambda=wD[[2]]
      gradRaw[[i]]=gradRaw[[i]]+lambda*weight[[i]]
      }
    gradMat=matrix(0,ncol=ncol(gradRaw[[i]]))
    gradMat_b=rep(0,length(gradRaw_b[[i]]))
    gradMat_g=0
    gradMat_betas=0
    if(BN){
      gradMat_g=rep(0,length(grdRaw_gammas[[i]]))
      gradMat_betas=rep(0,length(grdRaw_betas[[i]]))
    }
    if(optimization=='GD'){
      gradMat=gradRaw[[i]]
      gradMat_b=gradRaw_b[[i]]
      if(BN){
        gradMat_g=grdRaw_gammas[[i]]
        gradMat_betas=grdRaw_betas[[i]]
      }
    } 
    if(optimization%in%c('Momentum','Adam')){
      momentum[[i]]=beta1*momentum[[i]]+(1-beta1)*gradRaw[[i]]
      momentum_b[[i]]=beta1*momentum_b[[i]]+(1-beta1)*gradRaw_b[[i]]
      gradMat=momentum[[i]]
      gradMat_b=momentum_b[[i]]
    }
    if(optimization%in%c('RMSProp','Adam')){
      rmsprop[[i]]=beta2*rmsprop[[i]]+(1-beta2)*(gradRaw[[i]]^2)
      rmsprop_b[[i]]=beta2*rmsprop_b[[i]]+(1-beta2)*(gradRaw_b[[i]]^2)
      gradMat=gradRaw[[i]]/(sqrt(rmsprop[[i]]+tol))
      gradMat_b=gradRaw_b[[i]]/(sqrt(rmsprop_b[[i]]+tol))
    }
    mom_cor=matrix(0,ncol=ncol(gradRaw[[i]]))
    rms_cor=matrix(0,ncol=ncol(gradRaw[[i]]))
    mom_cor_b=rep(0,length(gradRaw_b[[i]]))
    rms_cor_b=rep(0,length(gradRaw_b[[i]]))
    if(optimization=='Adam'){
      mom_cor=momentum[[i]]/(1-beta1^t)
      rms_cor=rmsprop[[i]]/(1-beta2^t)
      mom_cor_b=momentum_b[[i]]/(1-beta1^t)
      rms_cor_b=rmsprop_b[[i]]/(1-beta2^t)
      gradMat=mom_cor/(sqrt(rms_cor)+tol)
      gradMat_b=mom_cor_b/(sqrt(rms_cor_b)+tol)
    }
    tmp[[i]]=weight[[i]]-rr*gradMat
    tmp_b[[i]]=biases[[i]]-rr*gradMat_b
    if(BN){
      tmp_g[[i]]=gammas[[i]]-rr*gradMat_g
      tmp_beta[[i]]=betas[[i]]-rr*gradMat_betas
    }
    if(trW){
      weightAbsMean=mean(abs(tmp[[i]]))
      biaisAbsMean=mean(abs(tmp_b[[i]]))
      weightAbsUpdateMean=mean(abs(tmp[[i]]))
      biaisAbsUpdateMean=mean(abs(tmp_b[[i]]-biases[[i]]))
      traceW[[i]]=as.matrix(c(weightAbsUpdateMean/weightAbsMean,biaisAbsUpdateMean/biaisAbsMean))
      gradUpdate[[i]]=rr*gradMat
    }
  }
  return(list(W=tmp,biases=tmp_b,D=delta,Dhat=dltaHat,b=tmp_beta,g=tmp_g,
              tr=traceW,gradupdate=gradUpdate,
              momentum=momentum,momentum_b=momentum_b,rmsprop=rmsprop,rmsprop_b=rmsprop_b))
}
