import tensorflow as tf

def get_global_step(global_step_name, increment_name):
    """
    Returns global step and increment, creates them if they have not been created
    
    Parameters 
    -----
    global_step_name: String
        Name of the global step tensor that you have created
    increment_name : String
        Name of the increment operation, for incrementing global step
    
    Returns
    -------
    global_step : Tensor
        Tensor that counts the steps of the graph
    inc : Operation
        Operation that increment the global_step
    """
    try:
        global_step = tf.get_default_graph().get_tensor_by_name(global_step_name)
        inc = tf.get_default_graph().get_operation_by_name(increment_name)
    except KeyError:
        print("Must create global step tensor and increment operation" )
#       use this if I figure out how to increment per layer within the batchNorm
#        with tf.Graph().as_default() as g:
#                with g.name_scope(""):
#                    global_step = tf.get_variable("global_step", shape=[],
#                    initializer=tf.zeros_initializer,
#                    trainable=False, dtype=tf.int32)
#                    inc = tf.assign_add(global_step, 1, name='increment')
        
    return global_step, inc
