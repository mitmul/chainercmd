import chainer


class CustomExtension(chainer.training.Extension):

    def __init__(self, message):
        self._message = message

    def initialize(self, trainer):
        print('I will send the message blow:')
        print(self._message)

    def __call__(self, trainer):
        print(self._message)

    def serialize(self, serializer):
        self._message = serializer('_message', self._message)
