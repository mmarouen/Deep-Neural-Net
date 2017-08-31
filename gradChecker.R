#Debug tool to make sure gradient descent is applied correctly
#sample draws at iteration r=50 & r=100 & r=200
#### parameters
#W=weights
#delta=gradient
#Z=input data
#resp=response variable
#tt="Regression" or "Classification"
#ll=loss function
#outF=output function
#active=activation function
#weightsVector=weighted classification

#### returns
#error

                       active="sigmoid",wD,tol=1e-5,weightsVector){
  l=sample(1:length(W),1)
  i=sample(1:nrow(W[[l]]),1)
  K=ncol(resp)
  biais=FALSE
  if(i==1){biais=TRUE}
  j=sample(1:ncol(W[[l]]),1)
  X=Z[[1]][,-1]
  #numerical gradient
  #right side
  W[[l]][i,j]=W[[l]][i,j]+tol
  fp1=forwardPropagate(W,X,outF,active)
  mat1=fp1$Z[[length(fp1$Z)]]
  estPlus=objective(mat1,resp,W,ll,wD,weightsVector)
  #left side
  W[[l]][i,j]=W[[l]][i,j]-2*tol
  fp2=forwardPropagate(W,X,outF,active)
  mat2=fp2$Z[[length(fp2$Z)]]
  estMinus=objective(mat2,resp,W,ll,wD,weightsVector)
  #restore weight
  W[[l]][i,j]=W[[l]][i,j]+tol
  #gradient
  gradEst=(estPlus-estMinus)/(2*tol)
  #Analytical gradient
  toCheck=mean(Z[[l]][,i]*delta[[l+1]][,j])
  if(wD[[1]] & !biais){toCheck=toCheck+wD[[2]]*W[[l]][i,j]}
  error=(gradEst-toCheck)/max(abs(gradEst),abs(toCheck))
  return(error)
}
