import tensorflow as tf
from d2l import tensorflow as d2l
import matplotlib.pyplot as plt

"""
Structure (Data flow) of cnn LeNet:
Input: 28x28 image, handwritten number 0-9
Convolutional layer 1: 28x28 feature map_________Convolution block: 5x5 convolution kernel + sigmoid activation + 2x2 pooling
Pooling layer 1: 14x14 feature map____________|       
Convolutional layer 2: 10x10 feature map
Pooling layer 2: 5x5 feature map
3 dense (full connection layer):120, 84, 10 outputs
Output: 10-D classification
"""


def net():
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=6, kernel_size=9, activation='sigmoid',  # filter: output channels
                               padding='same'),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        tf.keras.layers.Conv2D(filters=16, kernel_size=9,
                               activation='sigmoid'),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(120, activation='sigmoid'),
        tf.keras.layers.Dense(84, activation='sigmoid'),
        tf.keras.layers.Dense(10)])


# Check layer shapes
X = tf.random.uniform((1, 28, 28, 1))
for layer in net().layers:
    X = layer(X)
    print(layer.__class__.__name__, 'output shape: \t', X.shape)
"""
Conv2D output shape:         (1, 28, 28, 6)
AveragePooling2D output shape:       (1, 14, 14, 6)
Conv2D output shape:         (1, 10, 10, 16)
AveragePooling2D output shape:       (1, 5, 5, 16)
Flatten output shape:        (1, 400)
Dense output shape:          (1, 120)
Dense output shape:          (1, 84)
Dense output shape:          (1, 10)
"""
# Model training
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)
lr, num_epochs = 0.9, 10
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
