
import tensorflow as tf
import tensorlayer as tl
import numpy as np 
from convolution_nn import convolution_net

#are we using cifar_10 or MNISt
is_cifar = True
    
if is_cifar :
    
    #load data
    X_train, y_train, X_test, y_test = tl.files.load_cifar10_dataset(shape=(-1, 32, 32, 3))
    
    X_train = (X_train-np.mean(X_train, axis = 0))/np.std(X_train, axis = 0)
    X_test = (X_test-np.mean(X_test, axis = 0))/np.std(X_test, axis = 0)
    
    #create a data class object
    class data(object):
        def __init__(self,X_train, y_train, X_test, y_test ):
            self.x_train = X_train[0:45000,:,:,:]
            self.x_test = X_test
            self.y_train = y_train[0:45000]
            self.y_test = y_test
            self.y_val = y_train[45000:]
            self.x_val = X_train[45000:,:,:,:]
    
    
#        shape = [3,64,64,128,128,256,256, 512, 512]
#        pool = [True, True, True, True, True, True, True, True, True]
#        batch_norm = pool
#        keep = [0.8, 0.8, 0.8, 0.8, 0.8, 0.8,0.8,0.8,0.8]
#        input_shape = [None,32,32,3]
    
    
    #create the shape of the layers of resnet   
    shape = [3,32,32,64,64,128,128]
    #booleans for whether we use pool layers and batch norms
    pool = [True, True, True, True, True,True,True]
    batch_norm = pool
    #the probability that we keep a neuron for the dropout layer
    keep = [0.8, 0.8, 0.8, 0.8, 0.8,0.8,0.8]
    #input tensor shape
    input_shape = [None,32,32,3]
        
    data_cnn = data(X_train,y_train,X_test,y_test)
    #reshape = 32*32*3
    
else :
        
    x_train, y_train, x_val, y_val, x_test, y_test = tl.files.load_mnist_dataset(shape =(-1,28,28,1))
        
        
    class data(object):
        def __init__(self,X_train, y_train, x_val,y_val ,X_test, y_test ):
            self.x_train = X_train
            self.x_test = X_test
            self.y_train = y_train
            self.y_test = y_test
            self.y_val = y_val
            self.x_val = x_val
        
    shape = [1,32,32,64,64]
    pool = [True, True, True, True, True]
    batch_norm = pool
    keep = [0.8, 0.8, 0.8, 0.8, 0.8]
    input_shape = [None,28,28,1]
    data_cnn = data(x_train,y_train,x_val,y_val ,x_test,y_test)
                
    
        
        
        
# output size         
outputsz = 10
#leanring rate
learning_rate = 0.001

#creating the graph     
cnn = convolution_net(shape = shape,pool = pool, batch_norm=batch_norm, drop_out=batch_norm, keep=keep, input_shape=input_shape, outputsz = outputsz, learning_rate=learning_rate, is_res = True)
    
#training the network
cnn.train(data_cnn,64,100)
