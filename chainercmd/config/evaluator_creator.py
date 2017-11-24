from functools import partial
from importlib.machinery import SourceFileLoader

from chainercmd.config.base import ConfigBase


class EvaluatorCreator(ConfigBase):

    def __init__(self, **kwargs):
        required_keys = [
            'file',
            'name',
        ]
        optional_keys = [
            'args',
        ]
        super().__init__(
            required_keys, optional_keys, kwargs, self.__class__.__name__)


def get_evaluator_creator(file, name, args):
    loader = SourceFileLoader(name, file)
    mod = loader.load_module()
    evaluator_creator = getattr(mod, name)
    if args is not None:
        return partial(evaluator_creator, **args)
    else:
        return evaluator_creator


def get_evaluator_creator_from_config(config):
    evaluator_creator_config = EvaluatorCreator(**config)
    evaluator_creator = get_evaluator_creator(
        evaluator_creator_config.file, evaluator_creator_config.name,
        evaluator_creator_config.args)
    return evaluator_creator
