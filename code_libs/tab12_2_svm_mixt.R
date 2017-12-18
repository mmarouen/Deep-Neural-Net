source("E:/R Project/toolkits/Preprocess&FS.R")
source("E:/R Project/toolkits/ModelAssessment&Selection.R")
library(e1071)

outLin=svm(x = train,y = factor(yTrain),kernel = "linear")
mean(predict(outLin,test) != yTest)#0.461

outdeg2=svm(x=train,y=factor(yTrain),kernel = "polynomial",degree = 2)
mean(predict(outdeg2,test) != yTest)#0.063

outdeg3=svm(x=train,y=factor(yTrain),kernel = "polynomial",degree = 3)
mean(predict(outdeg3,test) != yTest)#0.342
