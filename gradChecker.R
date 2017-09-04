#Debug tool to make sure gradient descent is applied correctly
#sample draws at iteration r=50 & r=100 & r=200

#### returns
#error

gradChecker2<-function(W, #neurons
                       delta,#delta list
                       Z,#neurons outputs
                       resp,#response 
                       tt="Regression",#operation type
                       ll="RSS",#loss function
                       outF="Identity",#output function activation
                       active="sigmoid",#hidden layers activation
                       wD,tol=1e-5,weightsVector){
  l=sample(1:length(W),1)
  i=sample(1:nrow(W[[l]]),1)
  K=ncol(resp)
  bias=FALSE
  if(i==1){bias=TRUE}
  j=sample(1:ncol(W[[l]]),1)

  X=Z[[1]][,-1]
  #numerical gradient
  W_ij=W[[l]][i,j]
  #right side
  W[[l]][i,j]=W_ij+tol
  fp1=forwardPropagate(W,X,outF,active)
  mat1=fp1$output[[length(fp1$output)]]
  estPlus=objective(mat1,resp,W,ll,wD,weightsVector)
  #left side
  W[[l]][i,j]=W_ij-tol
  fp2=forwardPropagate(W,X,outF,active)
  mat2=fp2$output[[length(fp2$output)]]
  estMinus=objective(mat2,resp,W,ll,wD,weightsVector)

  #gradient
  gradEst=(estPlus-estMinus)/(2*tol)
  #Analytical gradient
  toCheck=mean(Z[[l]][,i]*delta[[l+1]][,j])
  if(wD[[1]] & !bias){toCheck=toCheck+wD[[2]]*W[[l]][i,j]}
  error=(gradEst-toCheck)/max(abs(gradEst),abs(toCheck))
  return(error)
}
