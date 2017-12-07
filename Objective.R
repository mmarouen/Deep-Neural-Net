objective<-function(X,#input matrix
                    Y, #response vector
                    W, #network weights
                    ll="RSS", #loss function
                    wD){
  K=ncol(X)
  N=nrow(X)
  eps=0
  if(ll=="RSS"){
    objective=(1/2)*mean(rowSums(((Y-X)^2)))
  }
  if(ll=="CrossEntropy"){
    if(K==1){objective=-mean(Y*log(X+eps)+(1-Y)*log(1-X-eps))}
    if(K>1){objective=-mean(rowSums(Y*log(X+eps)))}
  }
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

