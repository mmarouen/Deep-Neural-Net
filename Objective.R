#objective function
objective<-function(X,Y,W,ll="RSS",wD,weightsVector){
  K=ncol(X)
  N=nrow(X)
  #if(tt=="Regression" & ll=="RSS" & outF=="Identity"){
  if(ll=="RSS"){
    #objective=(1/2)*sum(rowMeans((Y-X)^2))
    objective=(1/2)*sum(rowSums(((Y-X)^2)%*%weightsVector))
  }
  if(ll=="CrossEntropy"){
    if(K==1){objective=-sum(Y*log(X)+(1-Y)*log(1-X))}
    if(K>1){objective=-sum((Y*log(X))%*%weightsVector)}
  }
  objective=(1/N)*objective
  #objective=(1/(N*K))*objective
  if(wD[[1]]){
    lambda=wD[[2]]
    s=0
    for(i in 1:length(W)){
      mat0=W[[i]][2:nrow(W[[i]]),]
      s=s+sum(mat0^2)}
    objective=objective+0.5*lambda*s
  }
  return(objective)
}
