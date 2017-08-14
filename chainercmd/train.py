#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import shutil
import time
from importlib import import_module

import yaml

import chainer
from chainer import iterators
from chainer import serializers
from chainer import training
from chainer.training import extensions
from chainer.training import triggers
from chainer.training import updaters
from chainercmd.config import get_custum_extension_from_config
from chainercmd.config import get_dataset_from_config
from chainercmd.config import get_model_from_config
from chainercmd.config import get_optimizer_from_config
from chainercmd.config import get_updater_creator_from_config

try:
    HAVE_NCCL = updaters.MultiprocessParallelUpdater.available()
except Exception:
    HAVE_NCCL = False


def create_result_dir(prefix='result'):
    result_dir = 'result/{}_{}_0'.format(
        prefix, time.strftime('%Y-%m-%d_%H-%M-%S'))
    while os.path.exists(result_dir):
        i = result_dir.split('_')[-1]
        result_dir = re.sub('_[0-9]+$', result_dir, '_{}'.format(i))
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    return result_dir


def create_result_dir_from_config_path(config_path):
    config_name = os.path.splitext(os.path.basename(config_path))[0]
    return create_result_dir(config_name)


def save_config_get_log_fn(result_dir, config_path):
    save_name = os.path.basename(config_path)
    a, b = os.path.splitext(save_name)
    save_name = '{}_0{}'.format(a, b)
    i = 0
    while os.path.exists('{}/{}'.format(result_dir, save_name)):
        i += 1
        save_name = '{}_{}{}'.format(a, i, b)
    shutil.copy(config_path, '{}/{}'.format(result_dir, save_name))
    return 'log_{}'.format(i)


def create_iterators(train_dataset, batchsize, valid_dataset, valid_batchsize,
                     devices):
    if HAVE_NCCL and len(devices) > 1:
        train_iter = [
            iterators.MultiprocessIterator(i, batchsize)
            for i in chainer.datasets.split_dataset_n_random(
                train_dataset, len(devices))]
    else:
        train_iter = iterators.MultiprocessIterator(
            train_dataset, batchsize)
    valid_iter = iterators.MultiprocessIterator(
        valid_dataset, valid_batchsize, repeat=False, shuffle=False)
    return train_iter, valid_iter


def create_updater(train_iter, optimizer, devices):
    if HAVE_NCCL and len(devices) > 1:
        updater = training.updaters.MultiprocessParallelUpdater(
            train_iter, optimizer, devices=devices)
    elif len(devices) > 1:
        optimizer.lr /= len(devices)
        updater = training.ParallelUpdater(
            train_iter, optimizer, devices=devices)
    else:
        updater = training.StandardUpdater(
            train_iter, optimizer, device=devices['main'])
    return updater


def train(args):
    config = yaml.load(open(args.config))

    print('==========================================')

    # Set workspace size
    if 'max_workspace_size' in config:
        chainer.cuda.set_max_workspace_size(config['max_workspace_size'])

    # Output version info
    print('chainer version: {}'.format(chainer.__version__))
    print('cuda: {}, cudnn: {}, nccl: {}'.format(
        chainer.cuda.available, chainer.cuda.cudnn_enabled, HAVE_NCCL))

    # Create result_dir
    if args.result_dir is not None:
        config['result_dir'] = args.result_dir
    else:
        config['result_dir'] = create_result_dir_from_config_path(args.config)
    log_fn = save_config_get_log_fn(config['result_dir'], args.config)
    print('result_dir:', config['result_dir'])

    # Instantiate model
    model = get_model_from_config(config)
    print('model:', model.__class__.__name__)

    # Initialize optimizer
    optimizer = get_optimizer_from_config(model, config)
    print('optimizer:', optimizer.__class__.__name__)

    # Setting up datasets
    train_dataset, valid_dataset = get_dataset_from_config(config)
    print('train_dataset: {}'.format(len(train_dataset)),
          train_dataset.__class__.__name__)
    print('valid_dataset: {}'.format(len(valid_dataset)),
          valid_dataset.__class__.__name__)

    # Prepare devices
    devices = {'main': args.gpus[0]}
    for gid in args.gpus[1:]:
        devices['gpu{}'.format(gid)] = gid

    # Create iterators
    train_iter, valid_iter = create_iterators(
        train_dataset, config['dataset']['train']['batchsize'],
        valid_dataset, config['dataset']['valid']['batchsize'], devices)
    print('train_iter:', train_iter.__class__.__name__)
    print('valid_iter:', valid_iter.__class__.__name__)

    # Create updater
    if 'updater_creator' in config:
        updater_creator = get_updater_creator_from_config(config)
        updater = updater_creator(train_iter, optimizer, devices)
    else:
        updater = create_updater(train_iter, optimizer, devices)
    print('updater:', updater.__class__.__name__)

    # Create trainer
    trainer = training.Trainer(
        updater, config['stop_trigger'], out=config['result_dir'])
    print('Trainer stops:', config['stop_trigger'])

    # Trainer extensions
    for ext in config['trainer_extension']:
        ext, values = ext.popitem()
        if ext == 'LogReport':
            trigger = values['trigger']
            trainer.extend(extensions.LogReport(
                trigger=trigger, log_name=log_fn))
        elif ext == 'observe_lr':
            trainer.extend(extensions.observe_lr(), trigger=values['trigger'])
        elif ext == 'dump_graph':
            trainer.extend(extensions.dump_graph(**values))
        elif ext == 'Evaluator':
            assert 'module' in values
            mod = import_module(values['module'])
            evaluator = getattr(mod, values['name'])
            if evaluator is extensions.Evaluator:
                evaluator = evaluator(
                    valid_iter, model, device=devices['main'])
            else:
                evaluator = evaluator(valid_iter, model.predictor)
            trainer.extend(
                evaluator, trigger=values['trigger'], name=values['prefix'])
        elif ext == 'PlotReport':
            trainer.extend(extensions.PlotReport(**values))
        elif ext == 'PrintReport':
            trigger = values.pop('trigger')
            trainer.extend(extensions.PrintReport(**values),
                           trigger=trigger)
        elif ext == 'ProgressBar':
            upd_int = values['update_interval']
            trigger = values['trigger']
            trainer.extend(extensions.ProgressBar(
                update_interval=upd_int), trigger=trigger)
        elif ext == 'snapshot':
            filename = values['filename']
            trigger = values['trigger']
            trainer.extend(extensions.snapshot(
                filename=filename), trigger=trigger)
        elif ext == 'ParameterStatistics':
            links = []
            for link_name in values.pop('links'):
                lns = [ln.strip() for ln in link_name.split('.') if ln.strip()]
                target = model.predictor
                for ln in lns:
                    target = getattr(target, ln)
                links.append(target)
            trainer.extend(extensions.ParameterStatistics(links, **values))
        elif ext == 'custom':
            custom_extension = get_custum_extension_from_config( values)
            trainer.extend(custom_extension)

    # LR decay
    if 'lr_drop_ratio' in config['optimizer'] \
            and 'lr_drop_triggers' in config['optimizer']:
        ratio = config['optimizer']['lr_drop_ratio']
        points = config['optimizer']['lr_drop_triggers']['points']
        unit = config['optimizer']['lr_drop_triggers']['unit']
        drop_trigger = triggers.ManualScheduleTrigger(points, unit)

        def lr_drop(trainer):
            trainer.updater.get_optimizer('main').lr *= ratio
        trainer.extend(lr_drop, trigger=drop_trigger)

    # Resume
    if args.resume is not None:
        fn = '{}.bak'.format(args.resume)
        shutil.copy(args.resume, fn)
        serializers.load_npz(args.resume, trainer)
        print('Resumed from:', args.resume)

    print('==========================================')

    trainer.run()
    return 0
