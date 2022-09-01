import tensorflow as tf
from d2l import tensorflow as d2l

# input X with shape of 6 rows and 8 columns
# X = tf.Variable(tf.ones((6, 8)))
# X[:, 2:6].assign(tf.zeros(X[:, 2:6].shape))
X = tf.eye(6, 8)

# construct kernel K which detects if the horizontally adjacent elements are the same. 0 for the same
K = tf.constant([[1.0, -1.0]])

# True output Y, col2 is 1: black to white, col6 is -1: white to black
Y = d2l.corr2d(X, K)

# Train a kernel that implements the same function----------------------------------------------------
# Construct a two-dimensional convolutional layer with 1 output channel and a
# kernel of shape (1, 2). For the sake of simplicity, we ignore the bias here
conv2d = tf.keras.layers.Conv2D(1, (1, 2), use_bias=False)

# The two-dimensional convolutional layer uses four-dimensional input and
# output in the format of (example, height, width, channel), where the batch
# size (number of examples in the batch) and the number of channels are both 1
X = tf.reshape(X, (1, 6, 8, 1))
Y = tf.reshape(Y, (1, 6, 7, 1))

lr = 3e-2

Y_hat = conv2d(X)
for i in range(10):
    with tf.GradientTape(watch_accessed_variables=False) as g:
        g.watch(conv2d.weights[0])
        Y_hat = conv2d(X)
        l = (abs(Y_hat - Y)) ** 2
        # update kernel
        update = tf.multiply(lr, g.gradient(l, conv2d.weights[0]))
        weights = conv2d.get_weights()
        weights[0] = conv2d.weights[0] - update
        conv2d.set_weights(weights)
        print(f'epoch {i}, loss {tf.reduce_sum(l):.3f}')

# tf.reshape(conv2d.get_weights()[0], (1, 2))
