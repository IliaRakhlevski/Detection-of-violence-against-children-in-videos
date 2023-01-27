# Utililies

from datetime import datetime


# set using CPU/GPU  
def set_cpu_gpu(CPU = True):
    
    import tensorflow as tf
    from keras import backend as K
    import psutil 
    
    num_cores = psutil.cpu_count(logical = False)
    
    #if GPU:
    num_GPU = 1
    num_CPU = 1
    
    if CPU:
        num_GPU = 0
    
    config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=num_cores,\
            inter_op_parallelism_threads=num_cores, allow_soft_placement=True,\
            device_count = {'CPU' : num_CPU, 'GPU' : num_GPU})
    session = tf.compat.v1.Session(config=config)
    K.set_session(session)
    
    
# get current date/time string
def get_current_time():
    # datetime object containing current date and time
    now = datetime.now()
    
    # dd/mm/YY H:M:S
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    
    return dt_string