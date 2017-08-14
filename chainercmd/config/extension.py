from importlib import import_module
from importlib.machinery import SourceFileLoader

from chainercmd.config.base import ConfigBase


class Extension(ConfigBase):

    def __init__(self, **kwargs):
        required_keys = []
        optional_keys = [
            'dump_graph',
            'Evaluator',
            'ExponentialShift',
            'LinearShift',
            'LogReport',
            'observe_lr',
            'observe_value',
            'snapshot',
            'PlotReport',
            'PrintReport',
        ]
        super().__init__(
            required_keys, optional_keys, kwargs, self.__class__.__name__)


class Custom(ConfigBase):

    def __init__(self, **kwargs):
        required_keys = [
            'file',
            'name'
        ]
        optional_keys = [
            'args',
        ]
        super().__init__(
            required_keys, optional_keys, kwargs, self.__class__.__name__)


def get_custum_extension_from_config(evaluator_config):
    config = Custom(**evaluator_config)
    loader = SourceFileLoader(config.name, config.file)
    mod = loader.load_module()
    if hasattr(config, 'args'):
        ext = getattr(mod, evaluator_config['name'])(**config.args)
    else:
        ext = getattr(mod, evaluator_config['name'])()
    return ext
