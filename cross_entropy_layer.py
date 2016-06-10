import tensorflow as tf
import numpy as np
from layer import CostLayer

class CrossEntropyLayer(CostLayer):
  def __init__(self, n_classes):
    self._n_classes = n_classes

  def loss(self, y_true, y):
    # if type(y_true_labels) != tf.Tensor:
    #   n = len(y_true_labels)      
    #   y_true = np.zeros([n, self._n_classes], np.float32)
    #   y_true[np.arange(n), y_true_labels] = 1
    #   y_true = tf.constant(y_true)
    # else:
    #   y_true = y_true_labels

    return tf.reduce_mean(-tf.reduce_sum(y_true * tf.log(y), reduction_indices = [1]))
