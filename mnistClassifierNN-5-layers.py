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




# code for NN

# Define Layers

K=200
L=100
M=60
N=30

#VARIABLEs
#Layer 1
W1=tf.Variable(tf.truncated_normal([784,K],stddev=.1))
B1=tf.Variable(tf.zeros([K]))
#Layer 2
W2=tf.Variable(tf.truncated_normal([K,L],stddev=.1))
B2=tf.Variable(tf.zeros([L]))
#Layer 3
W3=tf.Variable(tf.truncated_normal([L,M],stddev=.1))
B3=tf.Variable(tf.zeros([M]))
#Layer 4
W4=tf.Variable(tf.truncated_normal([M,N],stddev=.1))
B4=tf.Variable(tf.zeros([N]))
#Layer 5
W5=tf.Variable(tf.truncated_normal([N,10],stddev=.1))
B5=tf.Variable(tf.zeros([10]))

#PLACEHOLDERS
x=tf.placeholder(tf.float32,shape=[None,784])
y_true=tf.placeholder(tf.float32,[None,10])
#CREATE GRAPH OPERATIONS
Y1=tf.nn.relu(tf.matmul(x,W1)+B1)
Y2=tf.nn.relu(tf.matmul(Y1,W2)+B2)
Y3=tf.nn.relu(tf.matmul(Y2,W3)+B3)
Y4=tf.nn.relu(tf.matmul(Y3,W4)+B4)
y=tf.nn.softmax(tf.matmul(Y4,W5)+B5)



#LOSS FUNCTION
cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true,logits=y))

#optimize
optimizer=tf.train.GradientDescentOptimizer(learning_rate=.2)
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
        #print ("iteration number: %d and accuracy %d " %(step,sess.run(acc)))
        print ("iteration number: %d " %step)
    #run final analysis     
    print(sess.run(acc,feed_dict={x:mnist.test.images,y_true:mnist.test.labels}))
#plot results for accuracy vs. iteration
n1=list(range(len(accA)))
plt.plot(n1,accA,label='accuracy')

# Add title and axis names
plt.title('5-NN with relu')
plt.xlabel('iteration')
plt.ylabel('accuracy')