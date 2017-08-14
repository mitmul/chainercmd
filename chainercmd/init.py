import os
import shutil

from chainercmd import template


def init_basic(args):
    model_template = os.path.abspath(template.model.__file__)
    dataset_template = os.path.abspath(template.dataset.__file__)
    loss_template = os.path.abspath(template.loss.__file__)
    custom_ext_template = os.path.abspath(template.custom_extension.__file__)

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
        if not os.path.exists('extension'):
            os.mkdir('extension')
        shutil.copy(custom_ext_template, 'extension/')
    else:
        shutil.copy(model_template, './')
        shutil.copy(dataset_template, './')
        shutil.copy(loss_template, './')
        shutil.copy(custom_ext_template, './')
    init_config(args)


def init_config(args):
    model_template = os.path.abspath(template.model.__file__)
    dname = os.path.dirname(model_template)
    shutil.copy('{}/config.yml'.format(dname), './')


def init_full(args):
    init_basic(args)
    updater_creator_template = os.path.abspath(
        template.updater_creator.__file__)
    if args.create_subdirs:
        if not os.path.exists('updater'):
            os.mkdir('updater')
        shutil.copy(updater_creator_template, 'updater/')
    else:
        shutil.copy(updater_creator_template, './')
