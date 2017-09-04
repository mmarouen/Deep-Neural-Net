#performs backpropagation algorithm
#### Output
#W=updated weights list
#D=gradient
#tr=trace weights update (if trW)
#gradupdate=gradient values update (if trW)

backPropagate<-function(output, #neuron output
                        Z, #scalar product output
                        weight,#neurons
                        resp,#response vector
                        ll="RSS", #loss function
                        outF="Identity", #final layer activation
                        activation="sigmoid",#hidden layers activation
                        rr,#learning rate
                        wD,#weights decay
                        trW,weightsVector,
                        optimization='GD',#optimization alg
                        beta1=0.9,beta2=0.99,momentum=NULL,rmsprop=NULL, #optimization parameters
                        tol,#numerical tolerance
                        t #iteration count
                       ){
  N=nrow(output[[1]])
  K=ncol(resp)
  L=length(weight)+1#total layers count
  delta=list()#delta[[1]]=NULL
  gradRaw=list()
  tmp=list(length=L-1)#new weights
  traceW=list(length=L-1)
  gradUpdate=list()
  for(i in L:2){#weights in layer i are one index below: Z[[i]]=f(Z[[i-1]]%*%W[[i-1]])
    if(i==L){#output layer
      mat0=output[[L]]
      deltaL=mat0-resp
      if((ll=="RSS" & outF=="Identity")|(ll=="CrossEntropy" & outF=="Softmax")){
        delta[[i]]=deltaL
      }
      if(ll=="RSS" & outF=="Sigmoid"){
        delta[[i]]=mat0*(1-mat0)*deltaL
      }
      if(ll=="RSS" & outF=="Tanh"){
        delta[[i]]=deltaL*(1-mat0^2)
      }
      if(ll=="RSS" & outF=="Softmax"){
        mat1=matrix(0,ncol=ncol(mat0),nrow = nrow(mat0))
        mat2=delta[[i]]
        for(p in 1:K){
          for(j in 1:K){
            mat1[,p]=mat1[,p]+mat0[,j]*mat2[,j]*(delt(p,j)-mat0[,p])
          }
        }
        delta[[i]]=mat1
      }
      #delta[[i]]=delta[[i]]%*%weightsVector
    }
    if(i<=(L-1)){#BP to hidden layers
      mat1=output[[i]][,-1]
      if(is.null(dim(mat1))){mat1=matrix(mat1,ncol=1,nrow=length(mat1))}
      if(is.null(dim(weight[[i]][2:nrow(weight[[i]]),]))){
        vec=matrix(weight[[i]][2:nrow(weight[[i]]),],
                   ncol = nrow(weight[[i]])-1,nrow = ncol(delta[[i+1]]))
        mat2=delta[[i+1]]%*%vec
      }
      if(!is.null(dim(weight[[i]][2:nrow(weight[[i]]),]))){
        mat2=delta[[i+1]]%*%t(weight[[i]][2:nrow(weight[[i]]),])
      }
      if(activation=="tanh"){
        delta[[i]]=1.7159*(2/3)*(1-(mat1/1.7159)^2)*mat2

      }
      if(activation=="sigmoid"){
        delta[[i]]=mat1*(1-mat1)*mat2
      }
      if(activation=="linear"){
        delta[[i]]=mat2
      }
      if(activation=="ReLU"){
        mat1[mat1>0]=1
        mat1[mat1==0]=0
        delta[[i]]=mat1*mat2
      }
    }
    #gradient update
    gradRaw[[i]]=(1/N)*t(output[[i-1]])%*%delta[[i]]
    if(wD[[1]]){#regularization
      lambda=wD[[2]]
      mat0=weight[[i-1]][2:nrow(weight[[i-1]]),]
      if(ncol(tmp[[i-1]])==1){mat0=c(0,mat0)}
      if(ncol(tmp[[i-1]])>1){mat0=rbind(0,mat0)}
      gradRaw[[i]]=gradRaw[[i]]+lambda*mat0}
    gradMat=matrix(0,ncol=ncol(gradRaw[[i]]))
    
    if(optimization=='GD') gradMat=gradRaw[[i]]
    if(optimization%in%c('Momentum','Adam')){
      momentum[[i-1]]=beta1*momentum[[i-1]]+(1-beta1)*gradRaw[[i]]
      gradMat=momentum[[i-1]]
    }
    if(optimization%in%c('RMSProp','Adam')){
      rmsprop[[i-1]]=beta2*rmsprop[[i-1]]+(1-beta2)*(gradRaw[[i]]^2)
      gradMat=gradRaw[[i]]/(sqrt(rmsprop[[i-1]])+tol)
    }
    momentum_cor=matrix(0,ncol=ncol(gradRaw[[i]]))
    rmsprop_cor=matrix(0,ncol=ncol(gradRaw[[i]]))
    if(optimization=='Adam'){
      momentum_cor=momentum[[i-1]]/(1-beta1^t)
      rmsprop_cor=rmsprop[[i-1]]/(1-beta2^t)
      gradMat=momentum_cor/(sqrt(rmsprop_cor)+tol)
    }
    tmp[[i-1]]=weight[[i-1]]-rr*gradMat
    if(trW){
      weightAbsMean=mean(abs(tmp[[i-1]][2:nrow(tmp[[i-1]]),]))
      biaisAbsMean=mean(abs(tmp[[i-1]][1,]))
      weightAbsUpdateMean=mean(abs(tmp[[i-1]][2:nrow(tmp[[i-1]]),]-weight[[i-1]][2:nrow(weight[[i-1]]),]))
      biaisAbsUpdateMean=mean(abs(tmp[[i-1]][1,]-weight[[i-1]][1,]))
      traceW[[i-1]]=as.matrix(c(weightAbsUpdateMean/weightAbsMean,biaisAbsUpdateMean/biaisAbsMean))
      gradUpdate[[i-1]]=weight[[i-1]]-tmp[[i-1]]
    }
  }
  return(list(W=tmp,D=delta,tr=traceW,gradupdate=gradUpdate,momentum=momentum,rmsprop=rmsprop))
}
