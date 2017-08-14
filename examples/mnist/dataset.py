import chainer


class Dataset(chainer.dataset.DatasetMixin):

    def __init__(self, split='train'):
        super().__init__()

    def __len__(self):
        pass

    def get_example(self, i):
        pass


# You can delete this
class MNIST(chainer.dataset.DatasetMixin):

    def __init__(self, split='train', ndim=3):
        super().__init__()
        train, valid = chainer.datasets.get_mnist(ndim=ndim)
        self.d = train if split == 'train' else valid

    def __len__(self):
        return len(self.d)

    def get_example(self, i):
        return self.d[i]
