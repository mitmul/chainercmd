import chainer
from chainer import optimizers

from chainercmd.config.base import ConfigBase


class Optimizer(ConfigBase):

    def __init__(self, **kwargs):
        required_keys = [
            'method'
        ]
        optional_keys = [
            'args',
            'weight_decay',
            'lr_drop_ratio',
            'lr_drop_trigger',
        ]
        super().__init__(
            required_keys, optional_keys, kwargs, self.__class__.__name__)


def get_optimizer(model, method, optimizer_args, weight_decay=None):
    optimizer = getattr(optimizers, method)(**optimizer_args)
    optimizer.setup(model)
    if weight_decay is not None:
        optimizer.add_hook(chainer.optimizer.WeightDecay(weight_decay))
    return optimizer


def get_optimizer_from_config(model, config):
    opt_config = Optimizer(**config['optimizer'])
    optimizer = get_optimizer(
        model, opt_config.method, opt_config.args, opt_config.weight_decay)
    return optimizer
