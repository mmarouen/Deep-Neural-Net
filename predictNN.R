#make  predictions from a generated model and preprocessed input data
predictNN<-function(model,X){
  L=model$argV
  out1=forwardPropagate(model$W,X,L$tt,L$outF,L$active)
  yhatTest=transformOutput(out1$Z,L$tt,L$active,model$CL)$yhat
  return(yhatTest)
}
