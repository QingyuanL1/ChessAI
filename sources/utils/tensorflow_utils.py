
def set_session_config(per_process_gpu_memory_fraction=None, allow_growth=None, device_list='0'):
    """

    :param allow_growth: When necessary, reserve memory
    :param float per_process_gpu_memory_fraction: specify GPU memory usage as 0 to 1

    :return:
    """
    import tensorflow as tf
    import os
    
    # Force TensorFlow to use CPU only on Apple Silicon to avoid NCHW format issues
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    tf.config.set_visible_devices([], 'GPU')
    
    # Set TensorFlow to use CPU with better performance settings
    tf.config.threading.set_intra_op_parallelism_threads(0)  # Use all available cores
    tf.config.threading.set_inter_op_parallelism_threads(0)  # Use all available cores
