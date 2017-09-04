#Neural Network Optimization method
#### Output
#yhat=prediction
#Z=estimated neuron output in final iteration
#W=weights in final iteration
#D=gradient value
#reps response variable
#yhatMat= response matrix if needed
#grad=error values if gradient check is activated
#lossTrain=loss Curve for training matrix if debugging
#lossTest=loss Curve for validation matrix if debugging
#scoreTrain=score Curve for training matrix if debugging
#scoreTest=score Curve for validation matrix if debugging
#wTune=weightTune

OptimizeCost<-function(weight,resp,X,respT=NULL,XT=NULL, #respectively: neurons, response, input data, test response, test input
                       tt="Regression", #operation type: "Regression" or "Classification"
                       ll="RSS", #loss function: "RSS" or "CrossEntropy"
                       outF="Identity", #final layer activation function
                       active="sigmoid", #hidden layers activation
                       rr, #learning rate
                       minib, #minibatch size,input value wil be converted to exponent of 2
                       # so if minib=5 ==> mini-batch size= 2^5=32
                       wD, #weight decay
                       gradientCheck=FALSE,traceobj,trW,Epochs,Tolerance,weightsVec, #debugging params
                       optimization,#optimization agorithm
                       beta1,#momentum parameter
                       beta2,#RMSprop parameter
                       momentum,#momentum grad
                       rmsprop #RMSprop grad
                      ){
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
  if(trW){for(i in 1:length(weight)){weightsRatio[[i]]=0}}
  N=nrow(X)
  m=ifelse(!is.null(minib),2^minib,N) # minibatch size
  NB=ifelse(m!=N,ifelse(N%%m==0,N%/%m,N%/%m+1),1)
  error=rep(0,length=10)
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
      FP=forwardPropagate(weight,Xb,outF,active)
      output=FP$output
      Z=FP$Z
      ### back propagation
      BP=backPropagate(output,Z,weight,yb,ll,outF,active,rr,wD,trW,weightsVec,optimization,
                       beta1,beta2,momentum,rmsprop,Tolerance,t)
      
      ### update network response
      rsp2=transformOutput(output,tt,active,CL)
      resp2[idx,]=as.matrix(rsp2$yhatMat)
      
      ### Gradient check loop
      if(gradientCheck & (r %in% c(500,700,1000,2000,3000,5000,6000,7000,8000,9000))){
        check=gradChecker2(weight,BP$D,output,yb,tt,ll,outF,active,wD,Tolerance,weightsVector = weightsVec)
        error[which(c(500,700,1000,2000,3000,5000,6000,7000,8000,9000)==r)]=check
      }
      ### weights update
      weight=BP$W
      ### gradient update
      momentum=BP$momentum
      rmsprop=BP$rmsprop
    }

    ### Compute Cost
    if(traceobj){
      FP_obj=forwardPropagate(weight,X,outF,active)
      output_obj=FP_obj$output
      obj=objective(output_obj[[length(output_obj)]],resp0,weight,ll,wD,weightsVec)
      lossCurve=c(lossCurve,obj)
      respTr=as.matrix(transformOutput(output_obj,tt,active,CL)$yhat)
      if(tt=="Classification"){score=c(score,mean(respTr==as.vector(response)))}
      if(tt=="Regression"){score=c(score,1-obj)}
      if(!is.null(XT) & !is.null(respT)){
        outT=forwardPropagate(weight,XT,outF,active)$Z
        obj2=objective(outT[[length(outT)]],respT0,weight,ll,wD,weightsVec)
        lossCurve2=c(lossCurve2,obj2)
        yhatTest=transformOutput(outT,tt,active,CL)$yhat
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
    if((r==Epochs)|(diff<Tolerance)){break}
    r=r+1
    resp1=resp2
  }
  
  if(trW){
    for(i in 1:length(weight)){
      weightsRatio[[i]]=weightsRatio[[i]][,-1]
    }
    weightTune=list(weightsRatio,BP$gradupdate)
  }
  
  #final response
  FP_last=forwardPropagate(weight,X,outF,active)
  trans=transformOutput(FP_last$output,tt,active,CL)
  return(list(yhat=trans$yhat,Z=output,W=weight,D=BP$D,reps=r,yhatMat=trans$yhatMat,grad=error,lossTrain=lossCurve,lossTest=lossCurve2,scoreTrain=score,scoreTest=score2,wTune=weightTune))
}
