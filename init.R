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
init<-function(layers,#hidden layers vector
               X,#input data
               resp,#response
               activation="sigmoid",weightsVector=NULL,optimization='GD'){
  X=as.matrix(X)
  K=ncol(resp)
  cte=1
  moment=list()
  rmsprop=list()
  if(!is.null(weightsVector)){weightsVector=diag(weightsVector)}
  if(is.null(weightsVector)){weightsVector=diag(rep(1/K,K))}
  options(warn=-1)
  if(layers != 0){layers=as.matrix(c(ncol(X),layers,K))}# Layers x neurons
  if(layers == 0){layers=as.matrix(c(ncol(X),K))}
  options(warn=0)
  if(activation=="ReLU") cte=2
  weights=lapply(2:nrow(layers),function(x) matrix(rnorm((layers[x-1,1]+1)*(layers[x,1]),0,1)*sqrt(cte/ncol(X))*0.01,nrow=layers[x-1,1]+1,ncol=layers[x,1]))
  if(optimization%in%c("Adam","Momentum")){
    moment=lapply(2:nrow(layers),function(x) matrix(0,nrow=layers[x-1,1]+1,ncol=layers[x,1]))
  }
  if(optimization %in% c("Adam","RMSProp")){
    rmsprop=lapply(2:nrow(layers),function(x) matrix(0,nrow=layers[x-1,1]+1,ncol=layers[x,1]))
  }
  return(list(W=weights,L=layers,In=X,weightsVec=weightsVector,rmsprop=rmsprop,momentum=moment))
}
