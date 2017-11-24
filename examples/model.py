import chainer
import chainer.functions as F
import chainer.links as L


class Model(chainer.Chain):

    """Model definition.

    This is a template of model definition.

    """

    def __init__(self, n_class):
        super().__init__()
        with self.init_scope():
            pass

    def __call__(self, x):
        pass


# You can delete this! It's a sample model
class LeNet5(chainer.Chain):

    def __init__(self, n_class):
        super().__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, 6, 5, 1)
            self.conv2 = L.Convolution2D(6, 16, 5, 1)
            self.conv3 = L.Convolution2D(16, 120, 4, 1)
            self.fc4 = L.Linear(None, 84)
            self.fc5 = L.Linear(84, n_class)

    def __call__(self, x):
        h = F.relu(self.conv1(x))
        h = F.max_pooling_2d(h, 2, 2)
        h = F.relu(self.conv2(h))
        h = F.max_pooling_2d(h, 2, 2)
        h = F.relu(self.conv3(h))
        h = F.relu(self.fc4(h))
        return self.fc5(h)
