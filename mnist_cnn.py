import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Read data
mnist_data = input_data.read_data_sets('MNIST_data', one_hot = True)

# Constants
input_size = 784
no_classes = 10
batch_size = 50
total_batches = 400

# Placeholders
x_input = tf.placeholder(tf.float32, shape=[None, input_size])
y_input = tf.placeholder(tf.float32, shape=[None, no_classes])

x_input_reshape = tf.reshape(x_input, [-1, 28, 28, 1])

def convolution_layer(input_layer, filters, kernel_size = [3, 3], activation = tf.nn.relu):
	layer = tf.layers.conv2d(
		inputs = input_layer,
		filters = filters,
		kernel_size = kernel_size,
		activation = activation)

	return layer

def pooling_layer(input_layer, pool_size = [2, 2], strides = 2):
	layer = tf.layers.max_pooling2d(
		inputs = input_layer,
		pool_size = pool_size,
		strides = strides)

	return layer

def dense_layer(input_layer, units, activation = tf.nn.relu):
	layer = tf.layers.dense(
		inputs = input_layer,
		units = units,
		activation = activation)

	return layer

convolution_layer_1 = convolution_layer(x_input_reshape, 64)
pooling_layer_1 = pooling_layer(convolution_layer_1)

convolution_layer_2 = convolution_layer(pooling_layer_1, 128)
pooling_layer_2 = pooling_layer(convolution_layer_2)

flattened_pool = tf.reshape(pooling_layer_2, [-1, 5 * 5 * 128])
dense_layer_bottleneck = dense_layer(flattened_pool, 1024)

dropout_bool = tf.placeholder(tf.bool)
dropout_layer = tf.layers.dropout(
	inputs = dense_layer_bottleneck,
	rate = 0.5,
	training = dropout_bool
)

logits = dense_layer(dropout_layer, no_classes)

softmax_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels = y_input, logits = logits)
loss_operation = tf.reduce_mean(softmax_cross_entropy)

optimizer = tf.train.AdamOptimizer().minimize(loss_operation)

session = tf.Session()
session.run(tf.global_variables_initializer())

predictions = tf.argmax(logits, 1)
correct_predictions = tf.equal(predictions, tf.argmax(y_input, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

test_images, test_labels = mnist_data.test.images, mnist_data.test.labels

for batch_no in range(total_batches):
	mnist_batch = mnist_data.train.next_batch(batch_size)
	
	session.run([optimizer], feed_dict = {
		x_input: mnist_batch[0],
		y_input: mnist_batch[1],
		dropout_bool: True
	})

	if batch_no % 10 == 0:
		print("Batch number: ", end=" ")
		print(batch_no, end=" ")
		print("Accuracy: ", end=" ")
		print(session.run([accuracy_operation], feed_dict = {
			x_input: test_images,
			y_input: test_labels,
			dropout_bool: False
		}))

