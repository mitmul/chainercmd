import os
import shutil

from chainercmd import template


def init(args):
    model_template = os.path.abspath(template.model.__file__)
    dataset_template = os.path.abspath(template.dataset.__file__)
    loss_template = os.path.abspath(template.loss.__file__)

    if args.create_subdirs:
        if not os.path.exists('model'):
            os.mkdir('model')
        shutil.copy(model_template, 'model/')
        if not os.path.exists('dataset'):
            os.mkdir('dataset')
        shutil.copy(dataset_template, 'dataset/')
        if not os.path.exists('loss'):
            os.mkdir('loss')
        shutil.copy(loss_template, 'loss/')
    else:
        shutil.copy(model_template, './')
        shutil.copy(dataset_template, './')
        shutil.copy(loss_template, './')

    dname = os.path.dirname(model_template)
    shutil.copy('{}/config.yml'.format(dname), './')
