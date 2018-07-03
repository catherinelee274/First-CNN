# -*- coding: utf-8 -*-
"""
Catherine Lee
Basic Neural Network 
"""
#works 7/2/18 
#10 epochs: .971 accuracy!!!

#import
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#load mnist data 
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#variables for the NN
learning_rate = 0.5
epochs = 10
batch_size = 100

# declare the training data placeholders
# input x - for 28 x 28 pixels = 784
x = tf.placeholder(tf.float32, [None, 784]) #change this later depending on the image size
# now declare the output data placeholder - 10 digits
y = tf.placeholder(tf.float32, [None, 10])

#weight and bias variables for input and hidden layer

# now declare the weights connecting the input to the hidden layer
W1 = tf.Variable(tf.random_normal([784, 300], stddev=0.03), name='W1')
b1 = tf.Variable(tf.random_normal([300]), name='b1')
# and the weights connecting the hidden layer to the output layer
W2 = tf.Variable(tf.random_normal([300, 10], stddev=0.03), name='W2')
b2 = tf.Variable(tf.random_normal([10]), name='b2')

#activation function / calc output of hidden layer
hidden_out = tf.add(tf.matmul(x,W1),b1)
hidden_out = tf.nn.relu(hidden_out) #relu activation layer

#softmax activation for output layer 
y_ = tf.nn.softmax(tf.add(tf.matmul(hidden_out,W2),b2))

#cross entropy cost function for the optimization/backpropagation to work on
y_clipped = tf.clip_by_value(y_, 1e-10, 0.9999999) #makes sure we never get log(0), aka parametrizes trainig
cross_entropy = -tf.reduce_mean(tf.reduce_sum(y * tf.log(y_clipped) #cross entropy calculation
                         + (1 - y) * tf.log(1 - y_clipped), axis=1))

#optimizer/ peforms gradient descent and backpropogation
#optimizer - makes training faster ?
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

#var init
init_op = tf.global_variables_initializer() 

#accuracy assesment operation
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# start the session
with tf.Session() as sess:
   # initialise the variables
   sess.run(init_op)
   
   total_batch = int(len(mnist.train.labels) / batch_size) #calc total num of batches we are running in each trianing epoch 

   for epoch in range(epochs):
       avg_cost = 0 #to keep track of avg cross entropy cost for each epoch 

       for i in range(total_batch):
           batch_x, batch_y = mnist.train.next_batch(batch_size=batch_size) #get random batch x and batch y from MNIST 
           _, c = sess.run([optimizer, cross_entropy], 
                           feed_dict={x: batch_x, y: batch_y})
           avg_cost += c / total_batch
           print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost))
   print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))
