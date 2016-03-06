# Convolutional net


# load mnist data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# start TF interactive session
import tensorflow as tf
sess = tf.InteractiveSession()

# create data placeholders. y_ 10hot encoded
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
# reshape X input to 28*28 image (only one colour channel)
x_image = tf.reshape(x, [-1, 28, 28, 1])


# functions to create weight and bias variables
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

# convolution and pooling functions
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# first convolutional layer.  5*5 patch with 32 features. 2*2 maxpooling
# convolve image x with weights, bias for first layer.  apply Relu to resulting features.
# 28*28*1 -> 14*14*32
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# second convolutional layer. 5*5 patch with 64 features.  2*2 maxpooling
# 14*14*32 -> 7*7*64
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# Fully connected layer.  Apply weights&bias to flattened vector. then relu activation
# 7*7*64 -> 1024 neurons
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Apply dropout with probability keep_prob (used for training but not test)
# 1024 -> 1024
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# softmax to get classes
# 1024 -> 10
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


# run training
# cross entropy loss function
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
# use adam optimiser for training steps, lr 0.0001
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
# compare predictions to actuals
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
# calculate accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# initialise and then run for 20000 iterations in batches of 50. 50% dropout
sess.run(tf.initialize_all_variables())
for i in range(20000):
  batch = mnist.train.next_batch(50)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

# run on test data (no dropout)
print("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
