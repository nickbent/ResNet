
import tensorflow as tf 
import tensorlayer as tl 
import numpy as np
from full_layers import convolution, residual_block, dense
import time


class convolution_net(object):
    def __init__(self,shape,pool,batch_norm, drop_out, keep, input_shape, outputsz,learning_rate,is_res):
        """
        Initialize the paramters for the NN
        
        Paramters 
        --------
        shape : Vector for the outputs of each layer
        pool : Vector containing the truth value for having a pool layer for each layer
        batch_norm :  Vector containing turth value for having batch_norm for each layer
        drop_out : Vector containing truth value for haing drop_out for each layer
        keep : Vector containing the keep probabilities for each layer
        input_shape : shape of the input 
        reshape : reshape of the 
        learning_rate : learning rate for the adam optimizer
        
        """
        
        self.n_conv_layers = len(keep)
        self.shape = shape
        self.pool = pool
        self.batch_norm = batch_norm
        self.drop_out = drop_out
        self.keep = keep
        self.outputsz = outputsz
        self.learning_rate = learning_rate
        
        
        
        self.x = tf.placeholder(tf.float32, shape = input_shape, name='input')
        #do i need to reshape??
        self.y_ = tf.placeholder(tf.int64, shape=[None,],name='correctLabels') 
        
        #create the layers of the neural net
        if is_res:
            self.create_graph_res()
        else :
            self.create_graph_cnn()
        
        #create the optimization ops
        self.optimize()
        
        
        # Initializing the tensor flow variables
        init = tf.global_variables_initializer()

        # Launch the session
        self.sess = tf.InteractiveSession()
        self.sess.run(init)
        
    def create_graph_cnn(self):
        """
        Create all the layers of the convolutional neural net
        """
        #make first layer
        
        self.network = tl.layers.InputLayer(self.x, name='input_layer')
        
        #make a loop over convolutional layers to create each convolutional layer
        for ii in range(self.n_conv_layers ):
            layer = "conv"+'{:d}'.format(ii)
            #needs tp be changed 
            self.network= convolution(self.network,self.shape[ii],layer, keep = self.keep[ii], Pool = self.pool[ii], drop_out = self.drop_out[ii], batch_norm = self.batch_norm[ii])
            
        
        #make two fully connected layers
        self.network = dense( self.network, self.shape[-2][0] , "Dense1", tf.nn.relu, keep = 0.5, flatten = True, drop_out = True, batch_norm = True)
        
        
        self.network = dense( self.network, self.shape[-1][0] ,  "Dense2", tf.nn.relu, keep = 0.5, flatten = True, drop_out = True, batch_norm = True)
        
        
        self.network = dense( self.network, self.outputsz ,  "Output", tf.identity, keep = 0.5, flatten = False, drop_out = False, batch_norm = False)
        
        
        
    def create_graph_res(self):
        """
        Create all the layers of the residual layer neural net
        """
        
        
        self.network = tl.layers.InputLayer(self.x, name='input_layer')
        
        self.network= convolution(self.network,[7,7,self.shape[0],self.shape[1]],"conv", keep = self.keep[0], Pool = self.pool[0], drop_out = self.drop_out[0], batch_norm = self.batch_norm[0])
        
        #loop over all the blocks
        for ii in range(self.n_conv_layers-1):
            layer = "res"+'{:d}'.format(ii)
            
            self.network = residual_block( self.network, layer, self.shape[ii+1], bottleneck = False, batch_norm = True, drop_out = True, keep = self.keep[ii])
        
        #avg pool everything
        n_units = self.network.n_units
        self.network.outputs =  tf.reduce_mean(self.network.outputs,[1,2], name="global_avg_pool")
        
        #Dense layer with 10 layers for output, 
        self.network = tl.layers.DenseLayer(self.network, 
                                           n_units = self.outputsz, 
                                           act =tf.identity, 
                                           name='final_layer',
                                           W_init = tf.truncated_normal_initializer(stddev=np.sqrt(2./(n_units + self.outputsz))),
                                           b_init =tf.constant_initializer(value=0.1)) 
        
        
    def optimize(self):
        """
        Run the optimization and accuracy ops for the conv net
        """
        
        #creating optimization ops
        self.y = self.network.outputs
        self.cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = self.y, labels =self.y_))
        self.correct_prediction = tf.equal(tf.argmax(self.y, 1), self.y_)
        self.acc = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        self.train_params = self.network.all_params
        self.train_op = tf.train.AdamOptimizer(self.learning_rate, beta1=0.9, beta2=0.999,
                                          epsilon=1e-08, use_locking=False).minimize(self.cost, var_list=self.train_params)
        
    def train(self,data, batch_size, n_epoch):
        """
        Train the models
        """
        print_freq = 5
        for epoch in range(n_epoch):
            start_time = time.time()
            for X_train_a, y_train_a in tl.iterate.minibatches(data.x_train, data.y_train, batch_size, shuffle=True):
                feed_dict = {self.x: X_train_a, self.y_: y_train_a}
                feed_dict.update( self.network.all_drop )
                self.sess.run(self.train_op, feed_dict=feed_dict)

            if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
                print("Epoch %d of %d took %fs" % (epoch + 1, n_epoch, time.time() - start_time))
                
                train_loss, train_acc, n_batch = 0, 0, 0
                for X_train_a, y_train_a in tl.iterate.minibatches(
                                     data.x_train, data.y_train, batch_size, shuffle=True):
                    dp_dict = tl.utils.dict_to_one( self.network.all_drop )
                    feed_dict = {self.x: X_train_a, self.y_: y_train_a}
                    feed_dict.update(dp_dict)
                    err, ac = self.sess.run([self.cost, self.acc], feed_dict=feed_dict)
                    train_loss += err; train_acc += ac; n_batch += 1
                print("   train loss: %f" % (train_loss/ n_batch))
                print("   train acc: %f" % (train_acc/ n_batch))
                test_loss, test_acc, n_batch = 0, 0, 0
                for X_test_a, y_test_a in tl.iterate.minibatches(
                                        data.x_val, data.y_val, batch_size, shuffle=False):
                    dp_dict = tl.utils.dict_to_one( self.network.all_drop )
                    feed_dict = {self.x: X_test_a, self.y_: y_test_a}
                    feed_dict.update(dp_dict)
                    err, ac = self.sess.run([self.cost, self.acc], feed_dict=feed_dict)
                    test_loss += err; test_acc += ac; n_batch += 1
                print("   test loss: %f" % (test_loss/ n_batch))
                print("   test acc: %f" % (test_acc/ n_batch))
    
