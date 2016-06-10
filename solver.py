import tensorflow as tf
import numpy as np

class Solver(object):
  def __init__(self, model, data, algo, **kwargs):
    self._model = model
    self._data = data
    self._X_train = data['X_train'].astype(np.float32)
    self._y_train = data['y_train']
    self._X_val = data['X_val'].astype(np.float32)
    self._y_val = data['y_val']
    self._algo = algo

    self._num_epochs = kwargs.pop('num_epochs', 20)
    self._batch_size = kwargs.pop('batch_size', 30)
    self._verbose = kwargs.pop('verbose', 0)

    self.initialize_graph()

  def initialize_graph(self):
    x_shape = list(self._X_train.shape)
    self.train_size = x_shape[0]
    x_shape[0] = None
    y_shape = list(self._y_train.shape)
    y_shape[0] = None
    self._train_X_placeholder = tf.placeholder(tf.float32, x_shape)
    self._train_y_placeholder = tf.placeholder(tf.float32, y_shape)
    self._loss_op = self._model.loss(self._train_X_placeholder, self._train_y_placeholder)
    self._train_step = self._algo.minimize(self._loss_op)
    self.val_accuracy = self.val_accuracy_op()
    self.train_accuracy = self.train_accuracy_op()

  def train(self, init=True):
    self._model.set_mode('train')
    self._loss_history = []
    self._train_acc_history = []
    self._val_acc_history = []

    num_iteration = max(self.train_size / self._batch_size, 1)

    if init:
      self._sess = tf.Session()
      self._sess.run(tf.initialize_all_variables())
    for epoch in xrange(self._num_epochs):
      if self._verbose > 0:
        print 'epoch %d...' % (epoch + 1)
      for b in xrange(num_iteration):
        X_batch, y_batch = self.get_batch(self._batch_size)
        batch_loss, _ = self._sess.run([self._loss_op, self._train_step], feed_dict = {
          self._train_X_placeholder: X_batch,
          self._train_y_placeholder: y_batch,
        })
        self._loss_history.append(batch_loss)

        if self._verbose >= 4:
          print 'batch loss: %f' % batch_loss
      self.check_step()

  def val_accuracy_op(self):
    m = self._model.mode()
    self._model.set_mode('test')

    y = self._model.forward(tf.constant(self._X_val))
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(self._y_val, 1))

    self._model.set_mode(m)
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  def train_accuracy_op(self):
    x_shape = list(self._X_train.shape)
    train_size = x_shape[0]
    x_shape[0] = None
    y_shape = list(self._y_train.shape)
    y_shape[0] = None

    x = tf.placeholder(tf.float32, x_shape)
    y = tf.placeholder(tf.float32, y_shape)

    m = self._model.mode()
    self._model.set_mode('test')

    y_pred = self._model.forward(x)
    correct = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    self._train_acc_x_placeholder = x
    self._train_acc_y_placeholder = y

    self._model.set_mode(m)
    return accuracy

  def get_batch(self, batch_size):
    n = self._X_train.shape[0]
    mask = np.random.choice(n, batch_size)
    return self._X_train[mask], self._y_train[mask]

  def check_step(self, sample_size = 1000):
    train_size = self._X_train.shape[0]
    if train_size < sample_size:
      sample_size = train_size
    mask = np.random.choice(train_size, sample_size, False)
    X_sample = self._X_train[mask]
    y_sample = self._y_train[mask]

    train_acc, val_acc = self._sess.run([self.train_accuracy,
                                         self.val_accuracy],
                                        feed_dict = {
                                          self._train_acc_x_placeholder: X_sample,
                                          self._train_acc_y_placeholder: y_sample,
                                        })

    self._train_acc_history.append(train_acc)
    self._val_acc_history.append(val_acc)
    if self._verbose > 0:
      print 'train accuracy: %.4f, validation accuracy: %.4f, loss: %.4f' % \
        (train_acc, val_acc, self._loss_history[-1])

  def freeze(self):
    self._sess.close()
