### running script
source("neuralNet.R")
library(datasets)

df = datasets::attitude
X=df[,2:ncol(df)]
out = neuralNet(X,df$rating,DW = c(2,2),gradientCheck = T)
