# Deep-Neural-Net
Implements:  
neural networks regression  
binary, multinomial and weighted classification  
MLP & autoencoders  
NN is called using the "neuralNet" method  
The library also comes with a visualizer (neuralNet_visualizer sub-folder):  
1. visualize model summary (error rate, gradient check values, ...)  
2. plot objective function for validation and training  
3. plot weights evolution through network layers  

# Multilayer Perceptron  
NN parameters are very flexible:  
1. Type= "Classification" or "Regression"  
2. DW (Depth/Width of the network)=vector where each value indicates neurons in indexed layer  
3. loss (loss function)= "RSS" or "Deviance" (either logistic loss or residual sum of squares)  
4. outputFunc (output function)= "Sigmoid", "Softmax", "Identity"  
5. activationFunc (activation function)= "tanh", "sigmoid", "linear" (relu on the way)  
6. rate=learning rate  
7. weightDecay=TRUE/FALSE  
8. lambda=coefficient of the weight decay  
9. a host of <NN cheking tools such as gradient checker, traceObjective function, trace loss function, ...  
# Autoencoder:  
1. set type="Regression"  
2. set loss="RSS"  
3. if input is 0...1 then set output to "Softmax" or "Sigmoid" if parameters >1 or ==1 respectively  
4. Other parameters are tuned identically to other neural networks  
Remark: For runtime optimization reasons, the purpose of this implementation is experimentation and is not intended to be used in production environment.  
Please send all your remarks to <azzouz.marouen@gmail.com> 
