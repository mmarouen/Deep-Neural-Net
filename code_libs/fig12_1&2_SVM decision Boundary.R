source("E:/R Project/Mixtures/chapII_Mixtures.R")
library(bmrm)
out=generate()
train=out$train
resp=out$resp
blue=out$BlueCentroids
orange=out$OrangeCentroids
dat0=cbind(resp,train)
####prepare grid
L=100#size of the grid
X=seq(min(train[,1]),max(train[,1]),length=L)
Y=seq(min(train[,2]),max(train[,2]),length=L)
XY=expand.grid(X1=X,X2=Y)
###########2. Estimators
library(e1071)
###Linear Kernel cost=0.01
fit=svm(resp~.,data = dat0,cost=0.01,type="C-classification",kernel="linear")
yhat=predict(fit,XY)
mixture=ESL.mixture

###Linear Kernel cost=10000
fit2=svm(resp~.,data = dat0,cost=10000,type="C-classification",kernel="linear")
yhat2=predict(fit2,XY)
####Bayesian Boundary
yhatB=bayesianBoundary(XY,blue,orange)
###Polynomial Kernel
fit3=svm(resp~., data=dat0,type="C-classification",kernel="polynomial",degree=4,gamma=1,cost=1)
yhat3=predict(fit3,XY)
###Radial Kernel
fit4=svm(resp~., data=dat0,type="C-classification",kernel="radial",gamma=1,cost=1)
yhat4=predict(fit4,XY)

##############3. Plot results: cost=0.01
###fig 12.2 upper plot: cost=0.01
plot(train, xlab="X1", ylab="X2", xlim=range(train[,1]), type="n",main="Linear Kernel, cost=0.01")
points(XY,col=yhat, pch=".")
contour(X, Y, matrix(as.numeric(yhat),L,L), levels=c(1,2), add=TRUE, drawlabels=FALSE)
contour(X, Y, matrix(yhatB,L,L), levels=c(1,2), add=TRUE, drawlabels=FALSE,col="blue")
points(train, col=resp,pch=20)
###fig 12.2 bottom plot: cost=10000
plot(train, xlab="X1", ylab="X2", xlim=range(train[,1]), ylim=range(train[,2]), type="n",main="Linear Kernel, cost=10000")
points(XY,col=yhat2, pch=".")
contour(X, Y, matrix(as.numeric(yhat2),L,L), levels=c(1,2), add=TRUE, drawlabels=FALSE)
contour(X, Y, matrix(yhatB,L,L), levels=c(1,2), add=TRUE, drawlabels=FALSE,col="blue")
points(train, col=resp,pch=20)
###fig 12.3 upper plot: polynomial Kernel
plot(train, xlab="X1", ylab="X2", xlim=range(train[,1]), ylim=range(train[,2]), type="n",main="Polynomial Kernel")
points(XY,col=yhat3, pch=".")
contour(X, Y, matrix(as.numeric(yhat3),L,L), levels=c(1,2), add=TRUE, drawlabels=FALSE)
contour(X, Y, matrix(yhatB,L,L), levels=c(1,2), add=TRUE, drawlabels=FALSE,col="blue")
points(train, col=resp,pch=20)
###fig 12.3 bottom plot: radial Kernel
plot(train, xlab="X1", ylab="X2", xlim=range(train[,1]), ylim=range(train[,2]), type="n",main="Radial Kernel")
points(XY,col=yhat4, pch=".")
contour(X, Y, matrix(as.numeric(yhat4),L,L), levels=c(1,2), add=TRUE, drawlabels=FALSE)
contour(X, Y, matrix(yhatB,L,L), levels=c(1,2), add=TRUE, drawlabels=FALSE,col="blue")
points(train, col=resp,pch=20)

