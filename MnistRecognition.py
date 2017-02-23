from __future__ import print_function
import tensorflow as tf
from PIL import Image
import numpy as np

# Parameters
learning_rate = 0.001
batch_size = 100
display_step = 1
model_path = "./MnistModels/Model1/model1.ckpt"
load_model = False

# Network Parameters
n_hidden_1 = 256 # 1st layer number of features
n_hidden_2 = 256 # 2nd layer number of features
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = multilayer_perceptron(x, weights, biases)

# Initializing the variables
init = tf.global_variables_initializer()

# 'Saver' op to restore all the variables
saver = tf.train.Saver()

def recognition(img):
    global load_model
    with tf.Session() as sess:
        sess.run(init)
        # Restore model weights from previously saved model
        if False == load_model:
            saver.restore(sess, model_path)
            print("Model restored from file: %s" % model_path)
            load_model = True
        mtr = np.array(img, dtype = np.float32) / 255.0
        dstMtr = mtr.reshape(28*28)
        prediction = tf.argmax(pred, 1)
        return int(prediction.eval(feed_dict={x: [dstMtr]}, session=sess))