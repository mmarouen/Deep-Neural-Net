#softmax output function
softmax<-function(X){
  eps=1e-15
  Eps=1-eps
  M=max(X)
  product=apply(X,2,function(x) exp(-M-log(rowSums(exp(X-M-x)))))
  product=pmax(product,eps)
  product=pmin(product,Eps)
  return(product)
}
