#1st forward propagation
#initiates neurons
#### parameters
#layers: count of hidden layers
#X: input data
#resp=response matrix
#weightsVector=weights
#### output
#W=initiated weights
#L=layers vector containing count of neurons/layer
#rmsprop=init of rms prop
#momentum:init of momentum
init<-function(layers, #hidden layers vector
               X, #input matrix
               resp, #response vector
               activation="sigmoid",
               optimization='GD', #optimization method
               BN=FALSE #batch normalization binary switch
              ){
  # set.seed(10)
  X=as.matrix(X)
  K=ncol(resp)
  cte=1
  moment=list()
  moment_b=list()
  rmsprop=list()
  rmsprop_b=list()
  gammas=list()
  betas=list()
  options(warn=-1)
  if(layers != 0){layers=as.matrix(c(ncol(X),layers,K))}# Layers x neurons
  if(layers == 0){layers=as.matrix(c(ncol(X),K))}
  L=nrow(layers)
  options(warn=0)
  if(activation=="ReLU") cte=2
  weights=lapply(2:L,function(x) matrix(rnorm((layers[x-1,1])*(layers[x,1]),0,1)*sqrt(cte/layers[x-1,1])*0.01,nrow=layers[x-1,1],ncol=layers[x,1]))
  biases=lapply(2:L,function(x) rep(0,layers[x,1]))
  weights=c(0,weights)
  biases=c(0,biases)
  if(BN){
    gammas=lapply(2:L,function(x) rep(1,layers[x,1]))
    betas=lapply(2:L,function(x) rep(0,layers[x,1]))
    gammas=c(0,gammas)
    betas=c(0,betas)
  }
  bnList=list(BN=BN,gammas=gammas,betas=betas)
  if(optimization%in%c("Adam","Momentum")){
    moment=lapply(2:L,function(x) matrix(0,nrow=layers[x-1,1],ncol=layers[x,1]))
    moment_b=lapply(2:L,function(x) rep(0,layers[x,1]))
    moment=c(0,moment)
    moment_b=c(0,moment_b)
  }
  if(optimization %in% c("Adam","RMSProp")){
    rmsprop=lapply(2:L,function(x) matrix(0,nrow=layers[x-1,1],ncol=layers[x,1]))
    rmsprop_b=lapply(2:L,function(x) rep(0,layers[x,1]))
    rmsprop=c(0,rmsprop)
    rmsprop_b=c(0,rmsprop_b)
  }
  return(list(W=weights,b=biases,L=layers,In=X,
              rmsprop=rmsprop,rmsprop_b=rmsprop_b,momentum=moment,momentum_b=moment_b,
              bnList=bnList))
}
