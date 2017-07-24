from functools import partial
from importlib.machinery import SourceFileLoader

from chainercmd.config.base import ConfigBase


class UpdaterCreator(ConfigBase):

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


def get_updater_creator(file, name, args):
    loader = SourceFileLoader(name, file)
    mod = loader.load_module()
    updater_creator = getattr(mod, name)
    if args is not None:
        return partial(updater_creator, **args)
    else:
        return updater_creator


def get_updater_creator_from_config(config):
    updater_creator_config = UpdaterCreator(**config['updater_creator'])
    updater_creator = get_updater_creator(
        updater_creator_config.file, updater_creator_config.name,
        updater_creator_config.args)
    return updater_creator
