import tensorflow as tf
from layer import Layer

class SoftmaxLayer(Layer):
  def forward(self, X):
    y = tf.nn.softmax(X)
    return y
