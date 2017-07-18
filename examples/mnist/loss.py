from chainer import link
from chainer import reporter
from chainer.functions.evaluation import accuracy
from chainer.functions.loss import softmax_cross_entropy


class Loss(link.Chain):

    def __init__(self, predictor):
        super(Loss, self).__init__()
        self.lossfun = softmax_cross_entropy.softmax_cross_entropy
        self.accfun = accuracy.accuracy
        self.y = None
        self.loss = None
        self.accuracy = None

        with self.init_scope():
            self.predictor = predictor

    def __call__(self, *args):
        assert len(args) >= 2
        x = args[:-1]
        t = args[-1]
        self.y = None
        self.loss = None
        self.accuracy = None
        self.y = self.predictor(*x)
        self.loss = self.lossfun(self.y, t)
        reporter.report({'loss': self.loss}, self)
        self.accuracy = self.accfun(self.y, t)
        reporter.report({'accuracy': self.accuracy}, self)
        return self.loss
