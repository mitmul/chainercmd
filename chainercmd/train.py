#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import random
import re
import shutil
import time

import chainer
import numpy as np
import yaml
from chainer import iterators
from chainer import serializers
from chainer import training
from chainer.training import extensions
from chainer.training import updaters
from chainer.training import triggers
from chainercmd.config import get_dataset_from_config
from chainercmd.config import get_model_from_config
from chainercmd.config import get_optimizer_from_config

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

    # Setting random seed
    random.seed(config['seed'])
    np.random.seed(config['seed'])

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

    # Initialize optimizer
    optimizer = get_optimizer_from_config(model, config)

    # Setting up datasets
    train_dataset, valid_dataset = get_dataset_from_config(config)
    print('train: {}'.format(len(train_dataset)))
    print('valid: {}'.format(len(valid_dataset)))

    # Prepare devices
    devices = {'main': args.gpus[0]}
    for gid in args.gpus[1:]:
        devices['gpu{}'.format(gid)] = gid

    # Create iterators
    train_iter, valid_iter = create_iterators(
        train_dataset, config['batchsize'], valid_dataset,
        config['valid_batchsize'], devices)

    # Create updater and trainer
    updater = create_updater(train_iter, optimizer, devices)
    trainer = training.Trainer(
        updater, (config['stop_epoch'], 'epoch'), out=config['result_dir'])

    if 'log_trigger' in config:
        log_trigger = config['log_trigger']
        trainer.extend(extensions.LogReport(
            trigger=log_trigger, log_name=log_fn))

    if 'snapshot_trigger' in config:
        st = config['snapshot_trigger'][1]
        tx = '{' + '.updater.{}'.format(st) + '}'
        trainer.extend(extensions.snapshot(
            filename='snapshot_trainer_{}_{}'.format(st, tx)),
            trigger=tuple(config['snapshot_trigger']))

    if 'valid_trigger' in config:
        trainer.extend(
            extensions.Evaluator(valid_iter, model, device=args.gpus[0]),
            trigger=config['valid_trigger'])

    for ext in config['trainer_extension']:
        if isinstance(ext, dict):
            ext, values = ext.popitem()
        if ext == 'dump_graph':
            trainer.extend(extensions.dump_graph(**values))
        elif ext == 'PlotReport':
            values['trigger'] = log_trigger
            trainer.extend(extensions.PlotReport(**values))
        elif ext == 'PrintReport':
            if 'lr' in values:
                trainer.extend(extensions.observe_lr())
            trainer.extend(extensions.PrintReport(values))
        elif ext == 'ProgressBar':
            trainer.extend(extensions.ProgressBar(
                update_interval=log_trigger[0]), trigger=log_trigger)

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

    trainer.run()
    return 0
