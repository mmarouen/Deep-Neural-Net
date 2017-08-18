#1st forward propagation
#initiates neurons
#### parameters
#layers: count of hidden layers
#X: input data
#resp=response matrix
#weightsVector=weights
#### output
#W=initiated weights
#
init<-function(layers,X,resp,weightsVector=NULL){
  X=as.matrix(X)
  K=ncol(resp)
  if(!is.null(weightsVector)){weightsVector=diag(weightsVector)}
  if(is.null(weightsVector)){weightsVector=diag(rep(1/K,K))}
  options(warn=-1)
  if(layers != 0){layers=as.matrix(c(ncol(X),layers,K))}# Layers x neurons
  if(layers == 0){layers=as.matrix(c(ncol(X),K))}
  options(warn=0)
  #weights=lapply(2:nrow(layers),function(x) matrix(runif((layers[x-1,1]+1)*(layers[x,1]),-0.5,0.5),nrow=layers[x-1,1]+1,ncol=layers[x,1]))
  weights=lapply(2:nrow(layers),function(x) matrix(rnorm((layers[x-1,1]+1)*(layers[x,1]),0,sqrt(1/ncol(X))),nrow=layers[x-1,1]+1,ncol=layers[x,1]))
  return(list(W=weights,L=layers,In=X,weightsVec=weightsVector))
}
