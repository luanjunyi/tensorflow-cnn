import tensorflow as tf
from layer import Layer

class MaxPoolingLayer(Layer):
  def __init__(self, kernel_size, stride, data_format = 'NHWC'):
    self._data_format = data_format
    if data_format == 'NHWC':
      self._kernel_size = [1, kernel_size, kernel_size, 1]
      self._strides = [1, stride, stride, 1]
    else:
      assert data_format == 'NCHW', 'only NHWC or HCWH is supported as data_format'
      self._kernel_size = [1, 1, kernel_size, kernel_size]
      self._strides = [1, 1, stride, stride]

  def forward(self, X):
    return tf.nn.max_pool(X,
                          ksize = self._kernel_size,
                          strides = self._strides,
                          padding = 'SAME',
                          data_format = self._data_format)
