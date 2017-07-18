import chainer
import chainer.functions as F
import chainer.links as L


class Model(chainer.Chain):

    def __init__(self, n_class):
        super(Model, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(None, 64, 3, 1, 1)
            self.conv2 = L.Convolution2D(64, 128, 3, 1, 1)
            self.conv3 = L.Convolution2D(128, 256, 3, 1, 1)
            self.l4 = L.Linear(None, 1024)
            self.l5 = L.Linear(1024, n_class)

    def __call__(self, x):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = F.relu(self.l4(h))
        return self.l5(h)
