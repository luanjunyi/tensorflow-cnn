class Layer(object):
  def set_mode(self, mode):
    self._mode = mode

  def forward(self, X):
    return X

class CostLayer(object):
  def loss(self, y_true, y):
    return 0

