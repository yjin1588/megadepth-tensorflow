import tensorflow as tf

def eval_siRMSE(log_pred, gt, mask):
    '''
    - output:
        - data_loss:
            shape: (B,)
    '''
    gt_one = tf.where(tf.cast(mask, dtype=tf.bool), gt, tf.ones_like(gt))
    log_gt = tf.log(gt_one)

    N = tf.reduce_sum(mask, axis=[1,2,3])
    log_diff = tf.multiply(log_pred - log_gt, mask)
    s1 = tf.divide(tf.reduce_sum(tf.pow(log_diff, 2), axis=[1,2,3]), N)
    s2 = tf.divide(tf.pow(tf.reduce_sum(log_diff, axis=[1,2,3]),2), tf.pow(N,2))
    data_loss = tf.sqrt(s1 - s2)

    return data_loss

def eval_SDR(log_pred, gt, has_sfm, y_A_arr, x_A_arr, x_B_arr, y_B_arr):
    sfm_mask = tf.cast(has_sfm, tf.float32)
    pred = tf.exp(log_pred)
    A_arr = 
    tf.stack
    tf.zeros_like
    B_arr
    z_A_arr = tf.gather_nd(pred, A_arr)
    z_B_arr = tf.gather_nd(pred, B_arr)

    _batch_classify(z_A_arr, z_B_arr, gt)

def _batch_classify(z_A_arr, z_B_arr, gt):
    threshold = 1.1
    depth_ratio = tf.divide(z_A_arr, z_B_arr)

    estimated_labels = tf.zeros_like()

    equal_error_count
