import chainer


class Model(chainer.Chain):

    def __init__(self, n_class):
        super(Model, self).__init__()
        with self.init_scope():
            pass

    def __call__(self, x):
        pass
