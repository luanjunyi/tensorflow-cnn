import tensorflow as tf
from layer import Layer

class BatchNormalizationLayer(Layer):
  def __init__(self, dim, eps = 1e-5, momentum = 0.999):
    self._mean = tf.Variable(tf.constant(0.0, tf.float32, shape = [dim]), trainable = False)
    self._variance = tf.Variable(tf.constant(0.0, tf.float32, shape = [dim]), trainable = False)
    self._gamma = tf.Variable(tf.constant(1.0, tf.float32, shape=[dim,]))
    self._beta = tf.Variable(tf.constant(0.0, tf.float32, shape=[dim,]))
    self._ewma = tf.train.ExponentialMovingAverage(decay=momentum)
    self._eps = eps
    self._mode = 'train'
    self._ewma_op = self._ewma.apply([self._mean, self._variance])


  def forward(self, x):
    if self._mode == 'train':
      mean, variance = tf.nn.moments(x, [0])
      mean = tf.cast(mean, tf.float32)
      variance = tf.cast(variance, tf.float32)
      self._update_mean = self._mean.assign(mean)
      self._update_variance = self._variance.assign(variance)
      with tf.control_dependencies([self._update_mean, self._update_variance, self._ewma_op]):
        return tf.nn.batch_normalization(x, mean, variance,
                                         self._beta,
                                         self._gamma,
                                         self._eps)
    else:
      mean = self._ewma.average(self._mean)
      variance = self._ewma.average(self._variance)
      gamma = tf.identity(self._gamma)
      beta = tf.identity(self._beta)
      return tf.nn.batch_normalization(x, mean, variance, beta, gamma, self._eps)
