#coding=utf-8
import  tensorflow as tf

def cout_zeros():
    zeros_num = 0.
    all_num = 0.
    for v in tf.trainable_variables():
        zeros_num += tf.reduce_sum(tf.to_float(tf.less(tf.abs(v), tf.ones_like(v) * 0.0001)))  # 统计0的个数
        all_num += tf.reduce_sum(tf.ones_like(v))
    return [zeros_num, all_num]
        #梯度下降
def optimer(optimizer,loss,lr=1e-05):
    threshold = 0.01


    grads=optimizer.compute_gradients(loss, tf.trainable_variables())
    train_optimizers=[]
    for grad,v in zip(grads,tf.trainable_variables()):
        if grad is not None:
            # 只对权重处理，如果是偏置项不做处理，偏置项是一维的矩阵
            grad=grad[0]
            mask = tf.cond(tf.rank(v) > 1,
                               lambda: tf.to_float(tf.greater_equal(tf.abs(v), tf.ones_like(v) * threshold)),
                               lambda: tf.ones_like(v))




            prunev = tf.multiply(mask, v)
            newv = v.assign(prunev)



            lrate=tf.multiply(mask,tf.ones_like(v)*lr)#在mask中，0就是被修剪去的参数
            #print grad[0].get_shape(),lrate.get_shape(),v.get_shape()
            train_optimizer =v.assign(v-tf.multiply(grad,lrate))
            train_optimizers.append([train_optimizer,newv])


    return train_optimizers