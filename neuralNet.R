#Neural network method
#Train a neural network backpropagation using conj. grad. descent
#type ("Regression","Classification")
#loss function used: "RSS", "deviance" and the penalized version
#neurons architecture is referred via a vector DW:
#####takes a vector of length=#hidden layers, value[i]=#neurons layer i
#default is c(6): 1 hidden layer, 6 neurons 
# Type= "Classification" or "Regression"
# DW (Depth/Width of the network)=vector where each value indicates neurons in indexed layer
# loss (loss function)= "RSS" or "Deviance" (either logistic loss or residual sum of squares)
# outputFunc (output function)= "Sigmoid", "Softmax", "Identity"
# activationFunc (activation function)= "tanh", "sigmoid", "linear" (relu on the way)
# rate=learning rate
# weightDecay=TRUE/FALSE
# lambda=coefficient of the weight decay
# gradient checker=TRUE/FALSE debugging tool will perform gradient verification
# traceObj=TRUE/FALSE tracks score evolution
# traceWeights=TRUE/FALSE tracks weights variation
# weightsVector=permits for weights classification: weight of each class

#Train a neural network backpropagation using conj. grad. descent
#type ("Regression","Classification")
#loss function used: "RSS", "deviance" and the penalized version
#neurons architecture is referred via a vector DW:
#####takes a vector of length=#hidden layers, value[i]=#neurons layer i
#default is c(6): 1 hidden layer, 6 neurons 

neuralNet<-function(Input,response,InputTest=NULL,respTest=NULL, #input data
                    DW=c(6),#vector containing number of layers & number of neurons
                    type="Regression",loss="RSS",outputFunc="Identity",
                    epochs=NULL,#total count of iterations
                    weightsVector=NULL,#in case of classification, weight given to each class
                    tol=1e-7, #epsilon regulator value
                    activationFunc="sigmoid",#options= "tanh", "sigmoid", "linear", "ReLU"
                    rate=0.01,#learning rate
                    optAlg="GD", #Options= 'Adam', 'RMSProp', 'Momentum'
                    beta1=0.9,#momentum parameter
                    beta2=0.99,#RMS prop parameter
                    BN_momentum=0.9, #momentum for BN
                    mini_batch=NULL,#process in minibatches mini batch size are in 2^p so user input only exponent term
                    #if NULL then whole dateset is taken
                    BatchNorm=FALSE, #binary for mini-batch normalization
                    weightDecay=FALSE,lambda=NULL, #weight decay yes/No and decay amount 'lambda'
                    #dropout=FALSE,dropout yes/no, probs=0.7
                    probs=1, #probability
                    gradientCheck=FALSE,traceObj=FALSE,traceWeights=FALSE #debuggers
){
  t1=Sys.time()
  if(is.null(epochs)){epochs=15000}
  
  rsp=transformResponse(response,type)
  active=init(DW,Input,rsp$respMat,activationFunc,optAlg,BatchNorm)
  optimize=OptimizeCost(
    weight=active$W,biases=active$b,resp=rsp,X=active$In,respT=respTest,XT=InputTest, #load data + weights
    tt=type,ll=loss,outF=outputFunc,active=activationFunc,rr=rate,minib=mini_batch, #parameters
    wD=list(weightDecay,lambda),gradientCheck=gradientCheck,traceobj=traceObj, #debugging
    trW=traceWeights,Epochs=epochs,Tolerance=tol, #debugging
    optimization=optAlg,beta1=beta1,beta2=beta2, #optimizers
    momentum=active$momentum,momentum_b=active$momentum_b, #optimizers
    rmsprop=active$rmsprop,rmsprop_b=active$rmsprop_b,bnVars=active$bnList #optimizers
  )
  t2=Sys.time()
  L=list(dw=DW,tt=type,ll=loss,outF=outputFunc,active=activationFunc,rr=rate,wD=weightDecay,
         lambda=lambda,epochs=epochs,bnVars=active$bnList)
  return(list(yhat=optimize$yhat,yhatMat=optimize$yhatMat,y=response,CL=rsp$CL,
              W=optimize$W,b=optimize$b,popStats=optimize$popStats,
              D=optimize$D,Z=optimize$Z,r=optimize$reps,gradCheck=optimize$grad,
              lossTrain=optimize$lossTrain,lossTest=optimize$lossTest,
              scoreTrain=optimize$scoreTrain,scoreTest=optimize$scoreTest,
              trWeights=optimize$wTune,duration=t2-t1,argV=L))
}
