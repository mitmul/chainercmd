import chainer
from chainer.datasets import get_mnist


class MNIST(chainer.dataset.DatasetMixin):

    def __init__(self, split='train', ndim=3):
        super(MNIST, self).__init__()
        train, valid = get_mnist(ndim=ndim)
        self.d = train if split == 'train' else valid

    def __len__(self):
        return len(self.d)

    def get_example(self, i):
        return self.d[i]
