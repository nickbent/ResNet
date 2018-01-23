# ResNet
 
 ## Structure
 
This code is an implementation of "Deep Residual Learning for Image Recognition" by He et al from Microsoft in tensorlayer. ensorLayer is a deep learning and reinforcement learning library based on TensorFlow. I like tensorlayer because it is easier to use than tensorflow but not quite as high level as Keras, this gives you flexibility in impleneting new layers and using the tensorflow API. The code is structured as followed;  
 
full_layers  : are wrappers for convolutional, dense and resenet blocks, they provide flexibility to add pool, batch norm and dropout layers easily. 

convolution_nn : Creates a class structure for running the tensorflow graph. First it initializes the graphs and creates the optimiation ops. Then you can train it by calling the function cnn.train(batch_size,num_epoch)

resnet_run : Script to run the resnet, using either cifar10 or MNISt data. Here you can create the sizing of the layers, whether the layers contain batch_norm, drop_out or pool layers. 

batch_renorm : My implementation for  "Renormalization: Towards Reducing Minibatch Dependence in Batch-Normalized Models". I created a pull request in tensorlayer for this layer

global_step : A function used in batch_renorm, for batch renorm there is a decay on the paramaters so you need to keep track of what step you are in. This global step function makes sure you created an op that keeps track of the global step and increments it 



Dependencies: Tensorlayer, Tensorflow v1.3
