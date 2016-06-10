import tensorflow as tf
from layer import Layer

class Convolution2DLayer(Layer):
  def __init__(self,
               fan_in,
               fan_out,
               kernel_size,
               stride,
               weight_init_std = 1.0, mode = 'train', data_format = 'NHWC'):
    self._fan_in = fan_in
    self._fan_out = fan_out
    self._kernel_size = kernel_size
    if data_format == 'NHWC':
      self._strides = [1, stride, stride, 1]
    else:
      assert data_format == 'NCHW', 'only NHWC or HCWH is supported as data_format'
      self._strides = [1, 1, stride, stride]
    self._data_format = data_format

    self._bias = tf.Variable(tf.zeros([fan_out]))
    self._weights = tf.Variable(tf.truncated_normal([kernel_size, kernel_size, fan_in, fan_out],
                                                    stddev = weight_init_std))

  def forward(self, X):
    return tf.nn.conv2d(X, self._weights, strides = self._strides, padding = 'SAME',
                        data_format = self._data_format)
    
  
