# Concise Implementation of Linear Regression
import numpy as np
import tensorflow as tf
from d2l import tensorflow as d2l


def load_array(data_arrays, batch_size, is_train=True):  # @save
    """Construct TensorFlow data iterator"""
    dataset = tf.data.Dataset.from_tensor_slices(data_arrays)
    if is_train:
        dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size)
    return dataset


# Define original true parameters
true_w = tf.constant([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)

# Define batch size and load data
batch_size = 10
data_iter = load_array((features, labels), batch_size)

next(iter(data_iter))

"""Define the model of DL Net: Instance of Sequential class, which defines a container for several layers that will 
be chained together Given input data, a Sequential instance passes it through the first layer, in turn passing the 
output as the second layer’s input and so forth. In the following example, our model consists of only one layer, 
so we do not really need Sequential. But since nearly all of our future models will involve multiple layers, 
we will use it anyway just to familiarize you with the most standard workflow. """
# keras is the advanced API of TensorFlow
net = tf.keras.Sequential()
net.add(tf.keras.layers.Dense(1))

# Initialize model parameters
initializer = tf.initializers.RandomNormal(stddev=0.01)
net = tf.keras.Sequential()
net.add(tf.keras.layers.Dense(1, kernel_initializer=initializer))

# Define loss function using MeanSquareError
loss = tf.keras.losses.MeanSquaredError()

# Define optimize algorithm (Stochastic Minibatch Gradient Descent)
trainer = tf.keras.optimizers.SGD(learning_rate=0.03)

# Training
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        with tf.GradientTape() as tape:
            l = loss(net(X, training=True), y)
        grads = tape.gradient(l, net.trainable_variables)
        trainer.apply_gradients(zip(grads, net.trainable_variables))
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')

# Result comparision
w = net.get_weights()[0]
print('w estimate error: ', true_w - tf.reshape(w, true_w.shape))
b = net.get_weights()[1]
print('b estimate error: ', true_b - b)
