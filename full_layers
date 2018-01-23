import tensorflow as tf 
import tensorlayer as tl
import numpy as  np 




def convolution(network,shape, layer, activation = tf.nn.relu,  std = 1.0, bias = 0.1, stride_conv = [1, 1, 1, 1] , k_size=[1, 2, 2, 1],stride_pool= [1, 2, 2, 1], Pool = True, batch_norm = True, batch_renorm = False,keep = 0.7, drop_out = True):
    """
    Creates a convolution layer with optional pool, batch_(re)norm and drop_out
    
    Paramaters
    --------
    Network : A tensorlayer layer 
        Layer is fed into the convolutional layer
    Shape : List or numpy array of length 4
        Shape of the convolutional layer of the form [kernel_sz kernel_sz n_units_input n_units_output] 
    Layer : String
        Name of the convolution layer
    Activation : Tensorflow activation
        Activation function for after the convolutional layer
    std : Float
        standard deviation for the W_init initializer
    bias :Float
        bias for the b_init initializer
    stride_conv : List or numpy array of length 4
        Stride for the convolutional kernel
    k_size: List or numpy array of length 4
        Kernel size for the pool
    stride_pool:List or numpy array of length 4
        Strides for the pool kernel
    Pool : Boolean
        Whether to use pool layer
    batch_norm : Boolean
        Whether to use batch_norm layer
    batch_renorm : Boolean 
        Whether to use batch_renorm layer
    Keep : Float
        Keep probability for the dropout layer
    Droput : Boolean
        Whether to use drop out layer
    
    Returns 
    ------
    Network : A tensorlayer layer
        The network 
        
    
    """
    #assert (batch_norm and not batch_renorm) or (not batch_norm and batch_renorm), "Can't have batch norm both batch renorm layers"

    
    with tf.variable_scope(layer):
        network = tl.layers.Conv2dLayer(network,
                        act = activation,
                        shape = shape,  
                        strides=stride_conv,
                        padding='SAME',
                        name ='cnn',
                        W_init = tf.truncated_normal_initializer(stddev=std*np.sqrt(2./(shape[2] + shape[3]))),
                        b_init =tf.constant_initializer(value=bias) )
            
            
        if Pool :
                network = tl.layers.PoolLayer(network,
                        ksize=k_size,
                        strides=stride_pool,
                        padding='SAME',
                        pool = tf.nn.max_pool,
                        name ='pool')
        if batch_norm :
                network = tl.layers.BatchNormLayer(network, name = "batch_norm") 
        if batch_renorm :
                network = tl.layers.BatchReNormLayer(network,name = "batch_renorm")
                
        if drop_out :
                network = tl.layers.DropoutLayer(network, keep=keep, name='drop')
            
        network.n_units = shape[3]
            
    return(network)
    
def dense(network, units,  layer, activation = tf.nn.relu, bias= 0.1, std = 1.0, flatten = False, batch_norm = True, drop_out = True,  keep = 0.5):
    """
    Creates a convolution layer with optional pool, batch_(re)norm and drop_out
    
    Paramaters
    --------
    Network : A tensorlayer network
        Layer is fed into the convolutional layer
    Layer : String
        Name of the convolution layer
    Activation : Tensorflow activation
        Activation function for after the convolutional layer
    std : Float
        standard deviation for the W_init initializer
    bias :Float
        bias for the b_init initializer
    Flatten : Boolean
        Whether to use flatten layer
    batch_norm : Boolean
        Whether to use batch_norm
    Keep : Float
        Keep probability for the dropout layer
    Droput : Boolean
        Whether to use drop out layer
    
    Returns 
    ------
    Network : A tensorlayer network
        The network 
        
    
    """
    
    with tf.variable_scope(layer):
        if flatten:
            network = tl.layers.FlattenLayer(network, name='flatten_layer')
        network = tl.layers.DenseLayer(network, 
                                           n_units = units, 
                                           act = activation, 
                                           name='dense',
                                           W_init = tf.truncated_normal_initializer(stddev=std*np.sqrt(2./(units + network.n_units))),
                                           b_init =tf.constant_initializer(value=bias)) 
                                          
        n_units = network.n_units
            
        if batch_norm :
            network = tl.layers.BatchNormLayer(network, name = "batch_norm")

                
        if drop_out :
            network = tl.layers.DropoutLayer(network, keep=keep, name='drop')
            
        network.n_units = n_units

                
        return(network)
        
def residual_block(network, block, output_layer, bottleneck = True, batch_norm = True, keep = 0.8, drop_out = False):
    """
    Creates a convolution layer with optional pool, batch_(re)norm and drop_out
    
    Paramaters
    --------
    Network : A tensorlayer layer 
        Layer is fed into the convolutional layer
    Block : String
        Name of the resnet block
    Bottleneck : Boolean
        Whether to use bottleneck
    batch_norm : Boolean
        Whether to use batch_norm
    Keep : Float
        Keep probability for the dropout layer
    Droput : Boolean
        Whether to use drop out layer
    
    Returns 
    ------
    Network : A tensorlayer network
        The network 
        
    
    """
    with tf.variable_scope(block):
        input_layer = network.n_units
            
            
            
        ds = False
        if input_layer * 2 == output_layer:
            stride = [1,2,2,1]
            s = 2
            ds = True
        elif input_layer == output_layer:
            stride = [1,1,1,1]
            s = 1
            ds = False
            #if downsampled then the number of featurs is doubled
            #for the identiy mapping the same thing has to be don
            # for the bottleneck effect
            #the 1x1 convolutions have to increase the dimension size
            
            
        if bottleneck :
                    #need to chage this to more like the else
            if ds :
                pass
            else:
                shape = [1, 1, input_layer, input_layer/s]
                layer = "identity"
                identity = convolution(network,shape,layer,stride_conv = stride, Pool = False, batch_norm = batch_norm, drop_out = drop_out)
            shape = [1, 1, input_layer, input_layer/4]
            layer = "conv1_inblock"
            network = convolution(network,shape,layer,stride_conv = stride, Pool = False, batch_norm = batch_norm, drop_out = drop_out)
                    #make the secodn convoution layer in block, reassign stride to make sure that even if we did downsample in the first block 
                    # we dont in this one
            layer = "conv2_inblock"
            stride = [1,1,1,1]
            shape = [3, 3, input_layer/4, input_layer/4]
            network = convolution(network,shape,layer,stride_conv = stride, Pool = False, batch_norm = batch_norm, drop_out = drop_out)
            layer = "conv3_inblock"
            shape = [1, 1, input_layer, output_layer]
                
            if ds:
                network = convolution(network,shape,layer,stride_conv = stride, Pool = False, batch_norm = batch_norm, drop_out = drop_out)
            else:
                network = convolution(network,shape,layer,activation = tf.identity,stride_conv = stride, Pool = False, batch_norm = batch_norm, drop_out = drop_out)
                n_units = network.n_units
        else:
            if ds : 
                pass
            else:
                shape = [1, 1, input_layer, output_layer]
                layer = "identity"
                identity = convolution(network,shape,layer,stride_conv = stride, Pool = False, batch_norm = batch_norm, drop_out = drop_out, keep = keep)
                    # make the first convolutional block, the strides have been defined before so we do not need to worry
                    #about whether it is increasing dimension or not
            shape = [3, 3, input_layer, output_layer]
            layer = "conv1_inblock"
            network = convolution(network,shape,layer,stride_conv = stride, Pool = False, batch_norm = batch_norm, drop_out = drop_out, keep = keep)
                    #make the secodn convoution layer in block, reassign stride to make sure that even if we did downsample in the first block 
                    # we dont in this one
            layer = "conv2_inblock"
            stride = [1,1,1,1]
            shape = [3, 3, output_layer, output_layer]
                
            if ds :
                network = convolution(network,shape,layer,stride_conv = stride, Pool = False, batch_norm = True, drop_out = drop_out, keep = keep)
            else : 
                network = convolution(network,shape,layer,activation = tf.identity, stride_conv = stride, Pool = False, batch_norm = False, drop_out = drop_out, keep = keep)
                n_units = network.n_units
                
        if ds :
            pass
            
        else :
            network = tl.layers.ElementwiseLayer(layer = [network, identity], combine_fn = tf.add,name = 'Residual_output')
            
            network.outputs = tf.nn.relu(network.outputs)
                
            network= tl.layers.BatchNormLayer(network, name = "batch_norm")
            network.n_units = n_units
                
        return(network) 
