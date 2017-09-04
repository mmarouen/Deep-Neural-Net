#performs forward propagation algorithm
#### Return
#output=updated neurons outputs by layer
#Z=scalar product within each neuron

forwardPropagate<-function(weight,#neurons list
                           X,#input data
                           outF="Identity",#final layer activation function
                           activation="sigmoid"#hidden layers activation
                          ){
  L=length(weight)+1
  output=list()
  Z=list()
  output[[1]]=as.matrix(cbind(1,X))
  Z[[1]]=output[[1]]
  for(i in 2:L){
    Z[[i]]=output[[i-1]]%*%as.matrix(weight[[i-1]])
    if(i<L){
      if (activation=="sigmoid"){
        output[[i]]=1/(1+exp(-Z[[i]]))
      }
      if (activation=="tanh"){
        output[[i]]=1.7159*tanh((2/3)*Z[[i]])
      }
      if(activation=="linear"){
        output[[i]]=Z[[i]]
      }
      if(activation=="ReLU"){
        output[[i]]=as.matrix(Z[[i]])
        output[[i]][output[[i]]<0]=0
      }
      output[[i]]=cbind(1,output[[i]])
    }
    if(i==L){
      if(outF=="Identity"){output[[L]]=Z[[i]]}
      if(outF=="Sigmoid"){output[[L]]=1/(1+exp(-Z[[i]]))}
      if(outF=="Tanh"){output[[L]]=tanh(Z[[i]])}
      if(outF=="Softmax"){
        K=ncol(Z[[i]])
        if(K==1) output[[L]]=1/(1+exp(-Z[[i]]))
        if(K>2) output[[L]]=softmax(Z[[i]])
      }
    }
  }
  return(list(output=output,Z=Z))
}

