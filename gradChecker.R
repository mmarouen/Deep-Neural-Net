#Debug tool to make sure gradient descent is applied correctly
#sample draws at iteration r=50 & r=100 & r=200

#### returns
#error

gradChecker2<-function(W,#network weights
                       b, #network biases
                       delta, #delta = dY
                       deltaHat,#dY (batch norm)
                       Y, #activations
                       Zhat,#normalized layer output before non linearity
                       resp, #response matrix
                       tt="Regression", #mode
                       ll="RSS", #loss function
                       outF="Identity", #final layer output function 
                       active="sigmoid", #hidden layer activation function
                       wD,tol=1e-5,bnList){
  l=sample(2:length(W),1)
  l=3
  i=sample(1:nrow(W[[l]]),1)
  j=sample(1:ncol(W[[l]]),1) #indice for biases and weight
  # j=1
  K=ncol(resp)
  BN=bnList$BN
  
  
  X=Y[[1]]
  #####Weight gradient
  #numerical gradient
  W_ij=W[[l]][i,j]
  #right side
  W[[l]][i,j]=W_ij+tol
  fp=forwardPropagate(W,b,X,outF,active,bnList)
  mat=fp$Y[[length(fp$Y)]]
  estPlus=objective(mat,resp,W,ll,wD)
  #left side
  W[[l]][i,j]=W_ij-tol
  fp=forwardPropagate(W,b,X,outF,active,bnList)
  mat=fp$Y[[length(fp$Y)]]
  estMinus=objective(mat,resp,W,ll,wD)
  #gradient
  gradEst=(estPlus-estMinus)/(2*tol)
  #Analytical gradient
  toCheck=mean(Y[[l-1]][,i]*delta[[l]][,j])
  # print(paste(toCheck,gradEst))
  if(wD[[1]]){toCheck=toCheck+wD[[2]]*W[[l]][i,j]}
  err1=(gradEst-toCheck)/max(abs(gradEst),abs(toCheck))
  W[[l]][i,j]=W_ij #restore weight's value
  
  if(!BN){#compute bias gradient
    #####bias gradient
    #numerical gradient
    b_j=b[[l]][j]
    #right side
    b[[l]][j]=b_j+tol
    fp=forwardPropagate(W,b,X,outF,active,bnList)
    mat=fp$Y[[length(fp$Y)]]
    estPlus=objective(mat,resp,W,ll,wD)
    #left side
    b[[l]][j]=b_j-tol
    fp=forwardPropagate(W,b,X,outF,active,bnList)
    mat=fp$Y[[length(fp$Y)]]
    estMinus=objective(mat,resp,W,ll,wD)
    #gradient
    gradEst=(estPlus-estMinus)/(2*tol)
    #Analytical gradient
    toCheck=mean(delta[[l]][,j])
    err2=(gradEst-toCheck)/max(abs(gradEst),abs(toCheck))
    b[[l]][j]=b_j
    err=c(l,j,err1,err2)
  }
  if(BN){#compute gradients for gamma & betas
    g_j=bnList$gammas[[l]][j]
    b_j=bnList$betas[[l]][j]
    ### gammas
    #right side
    bnList$gammas[[l]][j]=g_j+tol
    fp=forwardPropagate(W,b,X,outF,active,bnList)
    mat=fp$Y[[length(fp$Y)]]
    estPlus=objective(mat,resp,W,ll,wD)
    #left side
    bnList$gammas[[l]][j]=g_j-tol
    fp=forwardPropagate(W,b,X,outF,active,bnList)
    mat=fp$Y[[length(fp$Y)]]
    estMinus=objective(mat,resp,W,ll,wD)
    #numerical gradient
    gradEst=(estPlus-estMinus)/(2*tol)
    #analytical gradient
    toCheck=mean(Zhat[[l]][,j]*deltaHat[[l]][,j])
    err2=(gradEst-toCheck)/max(abs(gradEst),abs(toCheck))
    bnList$gammas[[l]][j]=g_j
    ### betas
    #right side
    bnList$betas[[l]][j]=b_j+tol
    fp=forwardPropagate(W,b,X,outF,active,bnList)
    mat=fp$Y[[length(fp$Y)]]
    estPlus=objective(mat,resp,W,ll,wD)
    #left side
    bnList$betas[[l]][j]=b_j-tol
    fp=forwardPropagate(W,b,X,outF,active,bnList)
    mat=fp$Y[[length(fp$Y)]]
    estMinus=objective(mat,resp,W,ll,wD)
    #numerical gradient
    gradEst=(estPlus-estMinus)/(2*tol)
    #analytical gradient
    toCheck=mean(deltaHat[[l]][,j])
    err3=(gradEst-toCheck)/max(abs(gradEst),abs(toCheck))
    bnList$betas[[l]][j]=b_j
    err=c(l,j,err1,err2,err3)
  }
  return(err)
}
