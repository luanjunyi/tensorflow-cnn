import tensorflow as tf
from layer import Layer

class DropoutLayer(Layer):
  def __init__(self, keep_prob):
    self._keep_prob = keep_prob
    self._mode = 'train'

  def forward(self, X):
    return tf.nn.dropout(X, self._keep_prob) if (self._mode == 'train') else X
