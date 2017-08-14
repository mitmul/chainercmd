import chainer


class CustomExtension(chainer.training.Extension):

    def __init__(self, message):
        self._message = message

    def initialize(self, trainer):
        self._message += ' and Trainer ID is: {}'.format(id(trainer))

    def __call__(self, trainer):
        pass

    def serialize(self, serializer):
        self._message = serializer('_message', self._message)
