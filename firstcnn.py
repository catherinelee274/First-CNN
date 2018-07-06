# -*- coding: utf-8 -*-
"""
Catherine Lee
First Convolutional Neural Network  
"""


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
#### SQUARE IMAGES ONLYYYYYYYY
origX = 28
x = tf.placeholder(tf.float32, [None, origX*origX]) #change this later depending on the image size
# now declare the output data placeholder - 10 digits
#for mangrove this would be 2 
y = tf.placeholder(tf.float32, [None, 10])


#reshape x for pooling
#-1 so total size remains constant bc ocnvolution requries 4d tensor
x_shaped = tf.reshape(x,[-1,origX,origX,1])
#[i,j,k,l]
#i is batch size (something we don't know yet so it is set to -1)
#l is the channel number equal to 1 (if RGB image, 3 channels )


def create_new_conv_layer(input_data, channels, filters, filter_shape, pool_shape, name):
    # setup filter input shape for tf.nn.conv_2d
    conv_filt_shape = [ filter_shape[0], filter_shape[1], channels, filters]
    
    #init weights and bias 
    weights = tf.Variable(tf.truncated_normal(conv_filt_shape,stddev = 0.03), name = name+'_W')
    bias = tf.Variable(tf.truncated_normal([filters]),name=name+'_b')

    #set upt convolutional layer operation 
    #size of weights indicate what size convol. filter should be
    #[1,1,1,1] is the strides
    #strides[1] and strides [2] indicate stride in x and y
    #strides[0] and strides[3] are always 1 
    out_layer = tf.nn.conv2d(input_data, weights, [1,1,1,1], padding='SAME')
    
    #add biasese
    out_layer += bias
    
    #apply ReLU
    out_layer = tf.nn.relu(out_layer) 
    
    #max pooling
    ksize = [1,pool_shape[0], pool_shape[1],1]
    strides = [1,2,2,1]
    out_layer = tf.nn.max_pool(out_layer, ksize=ksize, strides = strides, padding='SAME')
    
    return out_layer 

# create some convolutional layers
layer1 = create_new_conv_layer(x_shaped, 1, 32, [5, 5], [2, 2], name='layer1')
layer2 = create_new_conv_layer(layer1, 32, 64, [5, 5], [2, 2], name='layer2') #final layer

#flatten output from final convolutional layer
flattened = tf.reshape(layer2, [-1,7*7*64])

#weights and bias for the layer
wd1 = tf.Variable(tf.truncated_normal([7*7*64,1000],stddev = 0.03), name = 'wd1')
bd1 = tf.Variable(tf.truncated_normal([1000],stddev = 0.01), name = 'bd1')

##multiplying weights of fully connected layer with flattened convolutional output, and adding bias 
#hidden_out = tf.add(tf.matmul(x,W1),b1)

dense_layer1 = tf.matmul(flattened,wd1) + bd1

#activation relu
dense_layer1 = tf.nn.relu(dense_layer1) #Activation layer



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
