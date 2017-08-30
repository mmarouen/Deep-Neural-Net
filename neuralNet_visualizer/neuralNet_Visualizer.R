summaryModel<-function(model){
  if(model$arg$tt=="Regression"){
    err=mean((model$y-model$yhat)^2)
    print(paste("training error=",err))
  }
    
  if(model$arg$tt=="Classification"){
    acc=mean(model$y == as.numeric(as.character(model$yhat)))
    print(paste("training accuracy=",acc*100,"%"))
    print(paste("training error=",(1-acc)*100,"%"))
  }
  print(paste("r=",model$r))
  print("gradientCheck:")
  print(model$gradCheck)
}

summaryGradUpHist<-function(model){
  dev.new()
  gradH=model$trWeights[[2]]
  L=length(gradH)
  par(mfrow=c(L,2))
  for(i in 1:L){
    if(i==L){
      hist(gradH[[i]][1,],main = paste("gradient biais output layer"),xlab = NULL)
      hist(gradH[[i]][2:nrow(gradH[[i]]),],main = paste("gradient weight output layer"),xlab = NULL)
    }
    if(i<L){
      hist(gradH[[i]][1,],main = paste("gradient biais hidden layer",L-i),xlab = NULL)
      hist(gradH[[i]][2:nrow(gradH[[i]]),],main = paste("gradient weight hidden layer",L-i),xlab = NULL)
    }
  }
}

summaryWeightHist<-function(model){
  dev.new()
  weightH=model$W
  L=length(weightH)
  par(mfrow=c(L,2))
  for(i in 1:L){
    if(i==L){
      hist(weightH[[i]][1,],main = paste("biais output layer"),xlab = NULL)
      hist(weightH[[i]][2:nrow(weightH[[i]]),],main = paste("weight output layer"),xlab = NULL)
    }
    if(i<L){
      hist(weightH[[i]][1,],main = paste("biais hidden layer",L-i),xlab = NULL)
      hist(weightH[[i]][2:nrow(weightH[[i]]),],main = paste("weight hidden layer",L-i),xlab = NULL)
    }
  }
}

summaryTraceWeights<-function(model){
  plot.new()
  WR=model$trWeights[[1]]
  N=length(WR)
  maxWeight=c()
  for(i in N:1){
    if(i==N){
      plot(WR[[i]][1,],type='l',col=N-i+1,xlab = "epoch",ylab = "ratio",main="mean magnitudes ratio updates/weights")
      lines(WR[[i]][2,],lty=2,col=N-i+1)
      maxWeight=c(maxWeight,paste("output Layer biais",round(max(WR[[i]][1,]),2)),paste("output Layer weight",round(max(WR[[i]][2,]),2)))
    }
    if(i<N){
      lines(WR[[i]][1,],type='l',col=N-i+1)
      lines(WR[[i]][2,],lty=2,col=N-i+1)
      maxWeight=c(maxWeight,paste("hidden layer",N-i,"biais",round(max(WR[[i]][1,]),2)),paste("hidden layer",N-i,"weights",round(max(WR[[i]][2,]),2)))
    }
  }
  legend("topright",legend = maxWeight,cex = 0.7,title = "Max values",
         lty=rep(c(1,2),length=2*length(WR)),col=rep(1:length(WR),each=2))
}

summaryTraceLoss<-function(model){
  plot.new()
  m=min(model$lossTrain,model$lossTest)
  M=max(model$lossTrain,model$lossTest)
  plot(model$lossTrain,type='l',xlab="epochs",ylab="loss",main="Loss VS Iterations",col="blue",ylim=c(m,M))
  if(length(model$lossTest)>1){
    lines(model$lossTest,type='l',col="red")
    legend("topright",legend = c("train","test"),cex = 0.7,lty=c(1,1),col=c("blue","red"))
  }
  if(length(model$lossTest)<1){
    legend("topright",legend = "train",cex = 0.7,lty=1,col="blue")
  }
}

summaryTraceScore<-function(model){
  plot.new()
  m=min(model$scoreTrain,model$scoreTest)
  M=max(model$scoreTrain,model$scoreTest)
  plot(model$scoreTrain,type='l',xlab = "epochs",ylab = "score",main="Score VS Iterations",
       col="blue",ylim=c(m,M))
  if(length(model$scoreTest)>1){
    lines(model$scoreTest,type='l',col="red")
    legend("bottomright",legend = c("train","test"),cex = 0.7,lty=c(1,1),col=c("blue","red"))
  }
  if(length(model$lossTest)<1){
    legend("bottomright",legend = "train",cex = 0.7,lty=1,col="blue")
  }
}