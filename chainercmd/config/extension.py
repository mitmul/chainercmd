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
