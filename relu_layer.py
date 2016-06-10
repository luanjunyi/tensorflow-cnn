import tensorflow as tf
from layer import Layer

class ReLuLayer(Layer):
  def forward(self, X):
    return tf.nn.relu(X)
