'''
Restore session paramters W and B which were saved in my_model2.ckpt

'''

import tensorflow as tf
W =tf.Variable(tf.zeros([2,3]),dtype=tf.float32,name='w')
b=tf.Variable([1,1,1,],dtype=tf.float32,name='b')

saver=tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess,'C:/Users/qatsh/OneDrive/Documents/CNN/my_model2.ckpt')
    w1=sess.run(W)
    b1=sess.run(b)
    print(w1)
    print(b1)