
import tensorflow as tf
W =tf.Variable([[1,3,4],[5,6,7]],dtype=tf.float32,name='w')
b=tf.Variable([10,20,30],dtype=tf.float32,name='b')
init=tf.global_variables_initializer()

saver=tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    save_path=saver.save(sess,'C:/Users/qatsh/OneDrive/Documents/CNN/my_model3.ckpt')
    print("Saved the file to",save_path)