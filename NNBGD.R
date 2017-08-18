#Neural Network Batch Gradient Descent
#### Parameters: same used in other functions
#### Return
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

NNBGD<-function(weight,resp,X,respT=NULL,XT=NULL,tt="Regression",ll="RSS",outF="Identity",
                active="sigmoid",rr,wD,gradientCheck=FALSE,traceobj,trW,Epochs,Tolerance,weightsVec){
  r=1
  error=NULL
  resp0=resp$respMat
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
  error=rep(0,length=8)
  repeat{
    FP=forwardPropagate(weight,X,tt,outF,active)
    output=FP$Z
    BP=backPropagate(output,weight,resp0,tt,ll,outF,active,rr,wD,trW,weightsVec)
    ###gradient check
    if(gradientCheck & (r %in% c(50,500,700,1000,2000,3000,5000,6000,7000,8000,9000))){
      check=gradChecker2(weight,BP$D,output,resp0,tt,ll,outF,active,wD,weightsVector = weightsVec)
      error[which(c(50,500,700,1000,2000,3000,5000,6000,7000,8000,9000)==r)]=check
    }
    weight=BP$W
    rsp2=transformOutput(output,tt,active,CL)
    if(traceobj){
      obj=objective(output[[length(output)]],resp0,weight,tt,ll,outF,wD,weightsVec)
      lossCurve=c(lossCurve,obj)
      respTr=rsp2$yhat
      if(tt=="Classification"){score=c(score,mean(respTr==resp$response))}
      if(tt=="Regression"){score=c(score,1-obj)}
      if(!is.null(XT) & !is.null(respT)){
        outT=forwardPropagate(weight,XT,tt,outF,active)$Z
        obj2=objective(outT[[length(outT)]],respT0,weight,tt,ll,outF,wD,weightsVec)
        lossCurve2=c(lossCurve2,obj2)
        yhatTest=transformOutput(outT,tt,active,CL)$yhat
        if(tt=="Classification"){score2=c(score2,mean(respT==yhatTest))}
        if(tt=="Regression"){score2=c(score2,1-obj2)}
        
      }
    }
    if(trW){
      for(i in 1:length(weight)){
        weightsRatio[[i]]=cbind(weightsRatio[[i]],BP$tr[[i]])
      }
    }
    resp2=rsp2$yhatMat
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
  trans=transformOutput(output,tt,active,CL)
  return(list(yhat=trans$yhat,Z=output,W=weight,D=BP$D,reps=r,yhatMat=trans$yhatMat,grad=error,lossTrain=lossCurve,
  lossTest=lossCurve2,scoreTrain=score,scoreTest=score2,wTune=weightTune))
}
