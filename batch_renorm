import tensorflow as tf

class BatchReNormLayer(Layer):
    """
    The :class:`BatchReNormLayer` class is a normalization layer, see Batch "Renormalization: Towards Reducing Minibatch Dependence
    in Batch-Normalized Models"
    Parameters
    -----------t
    layer : a :class:`Layer` instance
        The `Layer` class feeding into this layer.
    decay : float, default is 0.9.
        A decay factor for ExponentialMovingAverage, use larger value for large dataset.
    epsilon : float
        A small float number to avoid dividing by 0.
    act : activation function.
    is_train : boolean
        Whether train or inference.
    beta_init : beta initializer
        The initializer for initializing beta
    gamma_init : gamma initializer
        The initializer for initializing gamma
    dtype : tf.float32 (default) or tf.float16
    name : a string or None
        An optional name to attach to this layer.
    rmax : Float
        Maximum for multiplicative constant in affine transformation for x
    dmax : Float
        Maximum for additive constant in affine transformation for x
    use_decay : Boolean 
        Whether we start off with normal batch norm and decay to rmax, dmax
    decay_step : int
        The amount of steps before we start decaying 
    decay_rate : float
        base of the decay rate to decay to rmax and dmax
    global_step_name: String
        Name of the global step tensor that you have created
    increment_name : String
        Name of the increment operation, for incrementing global step
    
    
    ----------
    """
    def __init__(
        self,
        layer = None,
        decay = 0.9,
        epsilon = 0.00001,
        act = tf.identity,
        is_train = False,
        beta_init = tf.zeros_initializer,
        gamma_init = tf.random_normal_initializer(mean=1.0, stddev=0.002), # tf.ones_initializer,
        # dtype = tf.float32,
        name ='batchrenorm_layer',
        rmax = 3,
        dmax =5,
        use_decay = True,
        decay_step = 5000,
        decay_rate = 0.6,
        global_step_name = 'global_step:0',
        increment_name = 'increment'
        
    ):
        Layer.__init__(self, name=name)
        self.inputs = layer.outputs
        print("  [TL] BatchReNormLayer %s: dmax:%f, rmax:%f, decay:%f epsilon:%f act:%s is_train:%s" %
                            (self.name, dmax, rmax, decay, epsilon, act.__name__, is_train))
        x_shape = self.inputs.get_shape()
        params_shape = x_shape[-1:]

        from tensorflow.python.training import moving_averages
        from tensorflow.python.ops import control_flow_ops
        from global_step import get_global_step
        
        with tf.variable_scope(name) as vs:
            axis = list(range(len(x_shape) - 1))

            ## 1. beta, gamma
            if tf.__version__ > '0.12.1' and beta_init == tf.zeros_initializer:
                beta_init = beta_init()
            beta = tf.get_variable('beta', shape=params_shape,
                               initializer=beta_init,
                               dtype=D_TYPE,
                               trainable=is_train)#, restore=restore)

            gamma = tf.get_variable('gamma', shape=params_shape,
                                initializer=gamma_init,
                                dtype=D_TYPE,
                                trainable=is_train,
                                )#restore=restore)

            ## 2.
            if tf.__version__ > '0.12.1':
                moving_mean_init = tf.zeros_initializer()
            else:
                moving_mean_init = tf.zeros_initializer
            moving_mean = tf.get_variable('moving_mean',
                                      params_shape,
                                      initializer=moving_mean_init,
                                      dtype=D_TYPE,
                                      trainable=False)#   restore=restore)
            moving_variance = tf.get_variable('moving_variance',
                                          params_shape,
                                          initializer=tf.constant_initializer(1.),
                                          dtype=D_TYPE,
                                          trainable=False,)#   restore=restore)

            ## 3.
            # These ops will only be preformed when training.
            mean, variance = tf.nn.moments(self.inputs, axis)
            try:    # TF12
                update_moving_mean = moving_averages.assign_moving_average(
                                moving_mean, mean, decay, zero_debias=False)     # if zero_debias=True, has bias
                update_moving_variance = moving_averages.assign_moving_average(
                                moving_variance, variance, decay, zero_debias=False) # if zero_debias=True, has bias
                # print("TF12 moving")
            except Exception as e:  # TF11
                update_moving_mean = moving_averages.assign_moving_average(
                                moving_mean, mean, decay)
                update_moving_variance = moving_averages.assign_moving_average(
                                moving_variance, variance, decay)
                # print("TF11 moving")

            def mean_var_with_update():
                with tf.control_dependencies([update_moving_mean, update_moving_variance]):
                    return tf.identity(mean), tf.identity(variance)
                
            if is_train:
                if use_decay :
                    # get the global step and increment op
                    global_step, inc = get_global_step(global_step_name,increment_name)
                    #to do it this way I have to think of a way to only increment it once per graph not once per 
                    #layer
#                    if tf.get_default_session() is not None:
#                        #run the increment
#                        tf.get_default_session().run(inc)
#                        glob_step = tf.get_default_session().run(global_step)
                    if tf.get_default_session() is not None:
                        #if session is inirtialized get the step number
                        step_num = tf.get_default_session().run(global_step)
                    else :
                        #else set the step_num to 0
                        step_num = 0
                    if step_num < decay_step:
                        
                
                        mean, var = mean_var_with_update() # do I need to put this???

                        xn, batch_mean, batch_var = tf.nn.fused_batch_norm(self.inputs, gamma, beta,epsilon=epsilon, is_training=True)
                        self.output = act(xn)
                    else :
                        
                        mean, var = mean_var_with_update() # do I need to put this???

                        xn, batch_mean, batch_var = tf.nn.fused_batch_norm(self.inputs, gamma, beta,epsilon=epsilon, is_training=True)
                        moving_sigma = tf.sqrt(moving_variance, 'sigma')
                        r = tf.stop_gradient(tf.clip_by_value(tf.sqrt(batch_var / moving_variance), 1.0 / rmax, rmax))
                        d = tf.stop_gradient(tf.clip_by_value((batch_mean - moving_mean) / moving_sigma,-dmax, dmax))
                        decay = 1-decay_rate**((global_step-decay_step)/decay_step)
                        xn = xn * r*decay + d*decay
                        self.output = act(xn)
                else:
                    mean, var = mean_var_with_update() # do I need to put this???
                    xn, batch_mean, batch_var = tf.nn.fused_batch_norm(self.inputs, gamma, beta,epsilon=epsilon, is_training=True)
                    moving_sigma = tf.sqrt(moving_variance, 'sigma')
                    r = tf.stop_gradient(tf.clip_by_value(tf.sqrt(batch_var / moving_variance), 1.0 / rmax, rmax))
                    d = tf.stop_gradient(tf.clip_by_value((batch_mean - moving_mean) / moving_sigma,-dmax, dmax))
                    xn = xn * r + d
                    self.output = act(xn)
                            
                    
            else:
                self.outputs = act( tf.nn.batch_normalization(self.inputs, moving_mean, moving_variance, beta, gamma, epsilon) )

            variables = [beta, gamma, moving_mean, moving_variance]

            # print(len(variables))
            # for idx, v in enumerate(variables):
            #     print("  var {:3}: {:15}   {}".format(idx, str(v.get_shape()), v))
            # exit()

        self.all_layers = list(layer.all_layers)
        self.all_params = list(layer.all_params)
        self.all_drop = dict(layer.all_drop)
        self.all_layers.extend( [self.outputs] )
        self.all_params.extend( variables )
