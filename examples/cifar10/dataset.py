import chainer
from chainer.datasets import get_cifar10


class Dataset(chainer.dataset.DatasetMixin):

    def __init__(self, split='train'):
        super(Dataset, self).__init__()
        train, valid = get_cifar10()
        self.d = train if split == 'train' else valid

    def __len__(self):
        return len(self.d)

    def get_example(self, i):
        return self.d[i]
