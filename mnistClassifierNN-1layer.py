'''
 In this program, we use 1 layer NN to train NN to classify MNIST dataset

We use 1,000 samples in a batch of 100 to train the NN and plot performance againts each iteration


'''


import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
'''
Ther is an issue with using input_data and loading the set
'''
#%matplotlib inline


from tensorflow.examples.tutorials.mnist import input_data



mnist=input_data.read_data_sets("MNIST_data/",one_hot=True)


# print out number of sampels for training

print ("\n Number of taining data",mnist.train.num_examples)


print ("\n Number of test data",mnist.test.num_examples)


print ("\n shape of an impage",mnist.train.images[0].shape)

# display one image in gray scale

single_image=mnist.train.images[1].reshape(28,28)

plt.imshow(single_image,cmap='gist_gray')



# code for NN

#PLACEHOLDERS
x=tf.placeholder(tf.float32,shape=[None,784])
y_true=tf.placeholder(tf.float32,[None,10])

#VARIABLES
W=tf.Variable(tf.zeros([784,10]))
b=tf.Variable(tf.zeros([10]))

#CREATE GRAPH OPERATIONS
y=tf.matmul(x,W)+b

#LOSS FUNCTION
cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true,logits=y))

#optimize
optimizer=tf.train.GradientDescentOptimizer(learning_rate=.01)
train=optimizer.minimize(cross_entropy)

#Create Session
init=tf.global_variables_initializer()

accA=[] # variable to save test data accuracy by iteration
with tf.Session() as sess:
    sess.run(init)
    for step in range(5000):
        batch_x,batch_y=mnist.train.next_batch(100)
        sess.run(train,feed_dict={x:batch_x,y_true:batch_y})
        # Evaluate the model on test data for each iteration of W and b
        correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_true,1))
        acc=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        accA.append(sess.run(acc,feed_dict={x:mnist.test.images,y_true:mnist.test.labels}))
        print ("iteration number: %d" %step)
    #run final analysis     
    print(sess.run(acc,feed_dict={x:mnist.test.images,y_true:mnist.test.labels}))
#plot results for accuracy vs. iteration
n1=list(range(len(accA)))
fig = plt.figure()
plt.plot(n1,accA,label='accuracy')

# Add title and axis names
plt.title('1-NN')
plt.xlabel('iteration')
plt.ylabel('accuracy')

fig.savefig('accuracy-vs-iteration-1-layer.png', bbox_inches='tight')
