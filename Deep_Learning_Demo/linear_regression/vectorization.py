import math
import time
import numpy as np
import tensorflow as tf


def gpu_test():
    if tf.test.gpu_device_name():
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
    else:
        print("Please install GPU version of TF")


n = 10000
a = tf.ones(n)
b = tf.ones(n)


#
class Timer:  # @save
    """????????"""

    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """?????"""
        self.tik = time.time()

    def stop(self):
        """???????????????"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """??????"""
        return sum(self.times) / len(self.times)

    def sum(self):
        """??????"""
        return sum(self.times)

    def cumsum(self):
        """??????"""
        return np.array(self.times).cumsum().tolist()


c = tf.Variable(tf.zeros(n))
timer = Timer()
# Scalar computing
for i in range(n):
    c[i].assign(a[i] + b[i])
# Vectorized computing
timer.start()
d = a + b
