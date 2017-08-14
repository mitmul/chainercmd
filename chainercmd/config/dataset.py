import os
import shutil
from importlib.machinery import SourceFileLoader
from importlib import import_module
from chainercmd.config.base import ConfigBase


class Dataset(ConfigBase):

    def __init__(self, **kwargs):
        required_keys = [
            'name',
            'batchsize'
        ]
        optional_keys = [
            'file',
            'module',
            'args',
        ]
        super().__init__(
            required_keys, optional_keys, kwargs, self.__class__.__name__)


def get_dataset(class_name, module_name, file_name, args):
    if module_name is not None:
        mod = import_module(module_name)
    elif file_name is not None:
        loader = SourceFileLoader(class_name, file_name)
        mod = loader.load_module()
    else:
        raise ValueError(
            'Please specify EITHER \'module\' or \'file\' in the dataset '
            'config.')
    dataset = getattr(mod, class_name)
    dataset = dataset(**args)
    return dataset


def get_dataset_from_config(config):
    for key in config['dataset']:
        d = Dataset(**config['dataset'][key])
        if key == 'train':
            train = get_dataset(d.name, d.module, d.file, d.args)
        elif key == 'valid':
            valid = get_dataset(d.name, d.module, d.file, d.args)
        else:
            raise ValueError(
                'The dataset key should be either "train" or "valid", '
                'but {} was given.'.format(key))
        bname = os.path.basename(d.file)
        shutil.copy(
            d.file, '{}/{}_{}'.format(config['result_dir'], key, bname))
    return train, valid
