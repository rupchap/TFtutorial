# SOFTMAX REGRESSION MODEL


# load mnist data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# start TF interactive session
import tensorflow as tf
sess = tf.InteractiveSession()

# create data placeholders. y_ 10hot encoded
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# set up variables, to be initialised as zeros
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# initialise variables in TF session
sess.run(tf.initialize_all_variables())

# implement regression model to predict y
y = tf.nn.softmax(tf.matmul(x, W) + b)

# set up cost function - cross entropy between predicted y and actuals y_
# reduce_sum sums across all dimensions - all images in batch and all 10 output dimensions
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

# define training step - gradient descent on cross-entropy loss function with 0.01 stepsize
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# run the training - 1000 iterations on batches of size 50.
for i in range(1000):
  batch = mnist.train.next_batch(50)
  train_step.run(feed_dict={x: batch[0], y_: batch[1]})

# compare predictions to actuals
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

# calculate accuracy %
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))