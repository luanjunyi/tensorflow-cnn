import tensorflow as tf
from layer import Layer

class AffineLayer(Layer):
  def __init__(self, fan_in, fan_out, mode='train', weight_init_std = 1.0):
    self._mode = mode
    self._fan_in = fan_in
    self._fan_out = fan_out
    self.bias = tf.Variable(tf.zeros([fan_out]))
    self.weights = tf.Variable(tf.truncated_normal((fan_in, fan_out),
                                                   stddev = weight_init_std))

  def forward(self, X):
    n_input = X.get_shape().as_list()[0]
    X = tf.reshape(X, [-1, self._fan_in])
    # n_input, dim = X.get_shape().as_list()
    # assert dim == self._fan_in, 'input is of dimention (%d X %d), expecting (* X %d)' % \
    #   (n_input, dim, self._fan_in)

    y = tf.matmul(X, self.weights) + self.bias
    return y

  def warm_up(self):
    pass
