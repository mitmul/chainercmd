import chainer


class Dataset(chainer.dataset.DatasetMixin):

    def __init__(self, split='train'):
        super(Dataset, self).__init__()

    def __len__(self):
        pass

    def get_example(self, i):
        pass
