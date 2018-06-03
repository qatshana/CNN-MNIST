'''
 In this program, we use CNN to train NN to classify MNIST dataset

We use 5,000 samples in a batch of 50 to train the NN and plot performance againts each iteration


'''


import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
'''
Ther is an issue with using input_data and loading the set
'''
%matplotlib inline


from tensorflow.examples.tutorials.mnist import input_data



mnist=input_data.read_data_sets("MNIST_data/",one_hot=True)


# print out number of sampels for training

print ("\n Number of taining data",mnist.train.num_examples)


print ("\n Number of test data",mnist.test.num_examples)


print ("\n shape of an impage",mnist.train.images[0].shape)

# display one image in gray scale

single_image=mnist.train.images[1].reshape(28,28)

plt.imshow(single_image,cmap='gist_gray')




# Functions

def init_weights(shape):
    init_random_dist=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(init_random_dist)

def init_bias(shape):
	init_bias_vals=tf.constant(.1,shape=shape)
	return tf.Variable(init_bias_vals) 

def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool_2by2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

#CONV Layer
def convolutional_layer(input_x,shape):
    W=init_weights(shape)
    b=init_bias([shape[3]])
    return tf.nn.relu(conv2d(input_x,W)+b)
#Fully Connected
def normal_fulll_layer(input_layer,size):
    input_size=int(input_layer.get_shape()[1])
    W=init_weights([input_size,size])
    b=init_bias([size])
    return tf.matmul(input_layer,W)+b

# PLACEHOLDERS
x=tf.placeholder(tf.float32,shape=[None,784])
y_true=tf.placeholder(tf.float32,shape=[None,10])

#Layers
x_image=tf.reshape(x,[-1,28,28,1])
convo_1=convolutional_layer(x_image,shape=[5,5,1,32])
convo_1_pooling=max_pool_2by2(convo_1)

convo_2=convolutional_layer(convo_1_pooling,shape=[5,5,32,64])
convo_2_pooling=max_pool_2by2(convo_2)

convo_2_flat=tf.reshape(convo_2_pooling,[-1,7*7*64])
full_layer_one=tf.nn.relu(normal_fulll_layer(convo_2_flat,1024))

#Dropout
hold_pro=tf.placeholder(tf.float32)
full_one_dropout=tf.nn.dropout(full_layer_one,keep_prob=hold_pro)

y_pred=normal_fulll_layer(full_one_dropout,10)

#LOSS FUNCTION
cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true,logits=y_pred))

#Optimizer
optimizer=tf.train.AdamOptimizer(learning_rate=.001)
train=optimizer.minimize(cross_entropy)

init=tf.global_variables_initializer()

steps=5000
accA=[] # variable to save test data accuracy by iteration
with tf.Session() as sess:
    sess.run(init)
    for i in range(steps):
        batch_x,batch_y=mnist.train.next_batch(50)
        sess.run(train,feed_dict={x:batch_x,y_true:batch_y,hold_pro:.5})
        if i%100 == 0:
            print("ON STEP:{}".format(i))
            print("ACCURACY:")
            matches=tf.equal(tf.argmax(y_pred,1),tf.argmax(y_true,1))
            acc=tf.reduce_mean(tf.cast(matches,tf.float32))
            accT=sess.run(acc,feed_dict={x:mnist.test.images,y_true:mnist.test.labels,hold_pro:1})
            accA.append(accT)
            print (accT)
            print('\n')


#plot results for accuracy vs. iteration
n1=list(range(len(accA)))
fig = plt.figure()
plt.plot(n1,accA,label='accuracy')

# Add title and axis names
plt.title('5-NN with RELU')
plt.xlabel('iteration')
plt.ylabel('accuracy')
fig.savefig('accuracy-vs-iteration.png', bbox_inches='tight')