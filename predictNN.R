#make  predictions from a generated model and preprocessed input data
predictNN<-function(model,#model input
                    X #input test matrix
                   ){
  L=model$argV
  out1=forwardPropagate(model$W,model$b,X,L$outF,L$active,L$bnVars,model$popStats)
  yhatTest=transformOutput(out1$Y,L$tt,model$CL)$yhat
  return(yhatTest)
}
