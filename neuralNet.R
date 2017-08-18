#Neural network method
#Train a neural network backpropagation using conj. grad. descent
#type ("Regression","Classification")
#loss function used: "RSS", "deviance" and the penalized version
#neurons architecture is referred via a vector DW:
#####takes a vector of length=#hidden layers, value[i]=#neurons layer i
#default is c(6): 1 hidden layer, 6 neurons 
neuralNet<-function(Input,response,InputTest=NULL,respTest=NULL,DW=c(6),type="Regression",
                    loss="RSS",family="MLP",epochs=NULL,outputFunc="Identity",tol=1e-5,
                    activationFunc="sigmoid",rate=0.01,weightDecay=FALSE,lambda=NULL,
                    gradientCheck=FALSE,traceObj=FALSE,traceWeights=FALSE,weightsVector=NULL){
  if(is.null(epochs)){epochs=15000}
  rsp=transformResponse(response,type)
  active=init(DW,Input,rsp$respMat,type,outputFunc,activationFunc,weightsVector)
  optimize=NNBGD(weight=active$W,resp=rsp,X=active$In,respT=respTest,XT=InputTest,tt=type,
                 ll=loss,outF=outputFunc,active=activationFunc,rr=rate,
                 wD=list(weightDecay,lambda),gradientCheck=gradientCheck,traceobj=traceObj,
                 trW=traceWeights,Epochs=epochs,Tolerance=tol,weightsVec=active$weightsVec)
  L=list(dw=DW,tt=type,ll=loss,outF=outputFunc,active=activationFunc,rr=rate,wD=weightDecay,
         lambda=lambda,epochs=epochs)
  return(list(yhat=optimize$yhat,yhatMat=optimize$yhatMat,y=response,CL=rsp$CL,W=optimize$W,
              D=optimize$D,Z=optimize$Z,r=optimize$reps,gradCheck=optimize$grad,
              lossTrain=optimize$lossTrain,lossTest=optimize$lossTest,
              scoreTrain=optimize$scoreTrain,scoreTest=optimize$scoreTest,
              trWeights=optimize$wTune,duration=t2-t1,argV=L))
}