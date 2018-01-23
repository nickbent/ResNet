# ResNet
 
 ## Structure
 
This code is an implementation of "Deep Residual Learning for Image Recognition" by He et al from Microsoft in tensorlayer. ensorLayer is a deep learning and reinforcement learning library based on TensorFlow. I like tensorlayer because it is easier to use than tensorflow but not quite as high level as Keras, this gives you flexibility in impleneting new layers and using the tensorflow API. The code is structured as followed;  
 
full_layers  : are wrappers for convolutional, dense and resenet blocks, they provide flexibility to add pool, batch norm and dropout layers easily. 

convolution_nn : Creates a class structure for running the tensorflow graph. First it initializes the graphs and creates the optimiation ops. Then you can train it by calling the function cnn.train(batch_size,num_epoch)

resnet_run : Script to run the resnet, using either cifar10 or MNISt data. Here you can create the sizing of the layers, whether the layers contain batch_norm, drop_out or pool layers. 

## Batch ReNorm

batch_renorm : My implementation for  "Renormalization: Towards Reducing Minibatch Dependence in Batch-Normalized Models". I created a pull request in tensorlayer for this layer

global_step : A function used in batch_renorm, for batch renorm there is a decay on the paramaters so you need to keep track of what step you are in. This global step function makes sure you created an op that keeps track of the global step and increments it 

## Future Directions

Batch renorm truth values shold be included in the resnet_run layer. In the convolution_nn layer there should be an addition of a global step and increment op for the batch_renorm layer. Currently, global_step and increment_op have to be created and called outside the bacth norm because if they are created and run inside the bacth norm then it will increment once in every layer. I'm trying to see if these ops can be generated and run automatically within the batch_renorm layer without having to worry about them being called in every single layer. 



## Dependencies:

Tensorlayer, Tensorflow v1.3, numpy, time
