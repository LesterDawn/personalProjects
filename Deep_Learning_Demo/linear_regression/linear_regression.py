import random
import tensorflow as tf
import matplotlib as plt
from d2l import tensorflow as d2l


def synthetic_data(w, b, num_examples):  # @save
    """generate y=Xw+b+e, where e is the noise following Normal Distribution"""
    X = tf.zeros((num_examples, w.shape[0]))
    X += tf.random.normal(shape=X.shape)  # generate X
    y = tf.matmul(X, tf.reshape(w, (-1, 1))) + b  # w -> wT
    y += tf.random.normal(shape=y.shape, stddev=0.01)  # add noise
    y = tf.reshape(y, (-1, 1))
    return X, y


def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # Randomly read
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        j = tf.constant(indices[i: min(i + batch_size, num_examples)])
        yield tf.gather(features, j), tf.gather(labels, j)


def linreg(X, w, b):  # @save
    """Linear Regression model"""
    return tf.matmul(X, w) + b


def squared_loss(y_hat, y):  # @save
    """Mean square loss
       Reshape the true y to observed y(y_hat)
    """
    return (y_hat - tf.reshape(y, y_hat.shape)) ** 2 / 2


def sgd(params, grads, lr, batch_size):  # @save
    """
    stochastic gradient decent
    :param params:
    :param grads:
    :param lr: learning rate
    :param batch_size:
    :return:
    """
    for param, grad in zip(params, grads):
        param.assign_sub(lr * grad / batch_size)


# Generate parameters and features
true_w = tf.constant([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

# Check linear relationship between X[1] and y
d2l.set_figsize()
d2l.plt.scatter(features[:, 1].numpy(), labels.numpy(), 1)
# d2l.plt.show()

# Create minibatch
batch_size = 10
for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break

# Initialize model parameters
w = tf.Variable(tf.random.normal(shape=(2, 1), mean=0, stddev=0.01), trainable=True)
b = tf.Variable(tf.zeros(1), trainable=True)

# Training
lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        with tf.GradientTape() as g:
            l = loss(net(X, w, b), y)  # minibatch loss of x and y
        # calculate the gradient of l[w,b]
        dw, db = g.gradient(l, [w, b])
        # update parameters
        sgd([w, b], [dw, db], lr, batch_size)
    train_l = loss(net(features, w, b), labels)
    print(f'epoch {epoch + 1}, loss {float(tf.reduce_mean(train_l)):f}')

# Estimated results
print(f'w estimate error: {true_w - tf.reshape(w, true_w.shape)}')
print(f'b estimate error: {true_b - b}')
