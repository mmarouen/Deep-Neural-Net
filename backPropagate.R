backPropagate<-function(output,weight,resp,tt="Regression",ll="RSS",outF="Identity",
                        activation="sigmoid",rr,wD,trW,weightsVector){
  N=nrow(output[[1]])
  K=ncol(resp)
  L=length(weight)+1#total layers count
  delta=list()#delta[[1]]=NULL
  tmp=list(length=L-1)#new weights
  traceW=list(length=L-1)
  gradUpdate=list()
  for(i in L:2){#weights in layer i are one index below: Z[[i]]=f(Z[[i-1]]%*%W[[i-1]])
    if(i==L){#output layer
      mat0=output[[length(output)]]
      delta[[i]]=(mat0-resp)
      if(tt=="Regression" & outF=="Sigmoid"){
        mat1=delta[[i]]
        delta[[i]]=mat0*(1-mat0)*mat1
      }
      if(tt=="Regression" & outF=="Tanh"){
        mat1=delta[[i]]
        delta[[i]]=mat1*(1-mat0^2)
      }
      if(tt=="Regression" & outF=="Softmax"){
        mat1=matrix(0,ncol=ncol(mat0),nrow = nrow(mat0))
        mat2=delta[[i]]
        for(p in 1:K){
          for(j in 1:K){
            mat1[,p]=mat1[,p]+mat0[,j]*mat2[,j]*(delta(p,j)-mat0[,p])
          }
        }
        delta[[i]]=mat1
      }
      delta[[i]]=delta[[i]]%*%weightsVector
    }
    if(i<=(L-1)){#BP to hidden layers
      mat1=output[[i]][,-1]
      if(is.null(dim(mat1))){mat1=matrix(mat1,ncol=1,nrow=length(mat1))}
      if(is.null(dim(weight[[i]][2:nrow(weight[[i]]),]))){
        vec=matrix(weight[[i]][2:nrow(weight[[i]]),],
                  ncol = nrow(weight[[i]])-1,nrow = ncol(delta[[i+1]]))
        mat2=delta[[i+1]]%*%vec
      }
      if(!is.null(dim(weight[[i]][2:nrow(weight[[i]]),]))){
        mat2=delta[[i+1]]%*%t(weight[[i]][2:nrow(weight[[i]]),])
      }
      if(activation=="tanh"){
        delta[[i]]=apply(as.matrix(1:ncol(mat1)),1,function(x) 1.7159*(2/3)*(1-(mat1[,x]/1.7159)^2)*mat2[,x])
      }
      if(activation=="sigmoid"){
        delta[[i]]=apply(as.matrix(1:ncol(mat1)),1,function(x) (mat1[,x]*(1-mat1[,x]))*mat2[,x])
      }
      if(activation=="linear"){
        # delta[[i]]=apply(as.matrix(1:ncol(mat1)),1,function(x) mat1[,x]*mat2[,x])
        delta[[i]]=mat2
      }
    }
    #tmp[[i-1]]=weight[[i-1]]-rr*(1/(N*K))*t(output[[i-1]])%*%delta[[i]]
    tmp[[i-1]]=weight[[i-1]]-rr*(1/N)*t(output[[i-1]])%*%delta[[i]]
    if(wD[[1]]){#regularization
      lambda=wD[[2]]
      mat0=weight[[i-1]][2:nrow(weight[[i-1]]),]
      if(ncol(tmp[[i-1]])==1){mat0=c(0,mat0)}
      if(ncol(tmp[[i-1]])>1){mat0=rbind(0,mat0)}
      tmp[[i-1]]=tmp[[i-1]]-rr*lambda*mat0}
    if(trW){
      weightAbsMean=mean(abs(tmp[[i-1]][2:nrow(tmp[[i-1]]),]))
      biaisAbsMean=mean(abs(tmp[[i-1]][1,]))
      weightAbsUpdateMean=mean(abs(tmp[[i-1]][2:nrow(tmp[[i-1]]),]-weight[[i-1]][2:nrow(weight[[i-1]]),]))
      biaisAbsUpdateMean=mean(abs(tmp[[i-1]][1,]-weight[[i-1]][1,]))
      traceW[[i-1]]=as.matrix(c(weightAbsUpdateMean/weightAbsMean,biaisAbsUpdateMean/biaisAbsMean))
      gradUpdate[[i-1]]=weight[[i-1]]-tmp[[i-1]]
    }
  }
  return(list(W=tmp,D=delta,tr=traceW,gradupdate=gradUpdate))
}
