source("code_libs/FP_BP.R")
source("code_libs/Utils.R")
source("code_libs/debuggers.R")

#Optimization function
OptimizeCost<-function(weight,biases,resp,X,respT=NULL,XT=NULL,
                       tt="Regression",ll="RSS",outF="Identity",active="sigmoid",rr,minib,
                       wD,gradientCheck=FALSE,traceobj,trW,Epochs,Tolerance,
                       optimization,beta1,beta2,
                       momentum,momentum_b,
                       #momentum_g,momentum_betas,
                       rmsprop,rmsprop_b,
                       #rmsprop_g,rmsprop_betas,
                       bnVars){
  r=1 #epochs count
  t=0 #iterations count
  error=NULL
  resp0=as.matrix(resp$respMat)
  response=as.matrix(resp$response)
  CL=resp$CL
  resp1=resp0
  lossCurve=c()
  lossCurve2=c()
  score=c()
  score2=c()
  weightsRatio=list()
  weightTune=list()
  
  if(traceobj & !is.null(XT) & !is.null(respT)){
    XT=as.matrix(XT)
    respT0=transformResponse(respT,tt)$respMat}
  if(trW) for(i in 1:length(weight)) weightsRatio[[i]]=0
  N=nrow(X)
  m=ifelse(!is.null(minib),2^minib,N) # minibatch size
  NB=ifelse(m!=N,ifelse(N%%m==0,N%/%m,N%/%m+1),1)
  BN=bnVars$BN
  gradLocs=seq(500,1500,by=100)
  error=matrix(NA,ncol=length(gradLocs),nrow=4)
  if(BN) error=matrix(NA,ncol=length(gradLocs),nrow=5)
  repeat{
    resp2=matrix(0,ncol = dim(resp0)[2],nrow=dim(resp0)[1])
    # if(!is.null(minib)){
    #   shuffle=sample(N,N,replace=FALSE)
    #   X=X[shuffle,]
    #   resp0=as.matrix(resp0[shuffle,])
    #   resp1=as.matrix(resp1[shuffle,])
    #   response=as.matrix(response[shuffle,])
    # }
    
    for(b in 1:NB){
      t=t+1
      #load mini-batch data
      b_size=ifelse(NB==1,N,ifelse(b==NB,(N-(b-1)*m),m))
      idx=((b-1)*m+1):((b-1)*m+b_size)
      Xb=as.matrix(X[idx,])
      yb=as.matrix(resp0[idx,])
      ### forward propagation
      FP=forwardPropagate(weight,biases,Xb,outF,active,bnVars)
      Y=FP$Y
      Zhat=FP$Z_hat
      bnVars_t=bnVars
      bnVars_t$Z_hat=Zhat
      bnVars_t$sigma2=FP$popStats$sigma2
      ### back propagation
      BP=backPropagate(Y,weight,biases,yb,ll,outF,active,rr,wD,trW,optimization,
                       beta1,beta2,
                       momentum,momentum_b,
                       #momentum_g,momentum_betas,
                       rmsprop,rmsprop_b,
                       #rmsprop_g,rmsprop_betas,
                       Tolerance,t,bnVars_t)
      
      ### update network response
      rsp2=transformOutput(Y,tt,CL)
      resp2[idx,]=as.matrix(rsp2$yhatMat)
      
      ### Gradient check
      if(gradientCheck & (r %in% gradLocs)){
        check=gradChecker2(weight,biases,BP$D,BP$Dhat,Y,Zhat,yb,tt,ll,outF,active,wD,Tolerance,bnVars)
        error[,which(gradLocs==r)]=check
      }
      
      ### weights update
      weight=BP$W
      biases=BP$biases
      bnVars$gammas=BP$g
      bnVars$betas=BP$b
      
      ### gradient update
      momentum=BP$momentum
      rmsprop=BP$rmsprop
      momentum_b=BP$momentum_b
      rmsprop_b=BP$rmsprop_b
      #momentum_g=BP$momentum_g
      #rmsprop_g=BP$rmsprop_g
      #momentum_betas=BP$momentum_betas
      #rmsprop_betas=BP$rmsprop_betas
    }
    
    ### Compute Cost
    if(traceobj){
      FP_obj=forwardPropagate(weight,biases,X,outF,active,bnVars)
      Y_obj=FP_obj$Y
      obj=objective(Y_obj[[length(Y_obj)]],resp0,weight,ll,wD)
      lossCurve=c(lossCurve,obj)
      respTr=as.matrix(transformOutput(Y_obj,tt,CL)$yhat)
      if(tt=="Classification"){score=c(score,mean(respTr==as.vector(response)))}
      if(tt=="Regression"){score=c(score,1-obj)}
      if(!is.null(XT) & !is.null(respT)){
        outT=forwardPropagate(weight,biases,XT,outF,active,bnVars)
        YT=outT$Y
        obj2=objective(YT[[length(YT)]],respT0,weight,ll,wD)
        lossCurve2=c(lossCurve2,obj2)
        yhatTest=transformOutput(YT,tt,CL)$yhat
        if(tt=="Classification"){score2=c(score2,mean(respT==yhatTest))}
        if(tt=="Regression"){score2=c(score2,1-obj2)}
      }
    }
    #trace weights evolution
    if(trW){
      for(i in 1:length(weight)){
        weightsRatio[[i]]=cbind(weightsRatio[[i]],BP$tr[[i]])
      }
    }
    
    ### Convergence test
    diff=mean(abs(resp2-resp1))
    # if((r==Epochs)|(diff<Tolerance)){break}
    if(r==Epochs){break}
    r=r+1
    resp1=resp2
  }
  
  if(trW){
    for(i in 1:length(weight)) weightsRatio[[i]]=weightsRatio[[i]][,-1]
    weightTune=list(weightsRatio,BP$gradupdate)
  }
  
  #final response
  FP_last=forwardPropagate(weight,biases,X,outF,active,bnVars)
  popStats=FP_last$popStats
  trans=transformOutput(FP_last$Y,tt,CL)
  return(list(yhat=trans$yhat,Z=Y,W=weight,b=biases,D=BP$D,reps=r,yhatMat=trans$yhatMat,
              popStats=popStats,grad=error,lossTrain=lossCurve,lossTest=lossCurve2,scoreTrain=score,
              scoreTest=score2,wTune=weightTune))
}

