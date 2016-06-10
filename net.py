class Net(object):
  def __init__(self, cost_layer):
    self._layers = []
    self._cost_layer = cost_layer
    self._mode = 'test'

  def add(self, layer):
    self._layers.append(layer)

  def set_mode(self, mode):
    if self._mode == mode:
      return
    self._mode = mode
    for layer in self._layers:
      layer.set_mode(mode)

  def loss(self, X, y = None):
    mode = 'test' if (y is None) else 'train'
    self.set_mode(mode)

    t = X
    for i, layer in enumerate(self._layers):
      t = layer.forward(t)
    if self._mode == 'test':
      return t

    loss = self._cost_layer.loss(y, t)
    return loss

  def forward(self, X):
    return self.loss(X)

  def mode(self):
    return self._mode
