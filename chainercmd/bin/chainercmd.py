#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import glob
import imp
import json
import os
import random
import re
import shutil
import sys
import tempfile
import time

import chainer
import chainer.functions as F
import numpy as np
import yaml
from chainer import iterators
from chainer import optimizers
from chainer import serializers
from chainer import training
from chainer.training import extensions
from chainer.training import updaters

try:
    HAVE_NCCL = updaters.MultiprocessParallelUpdater.available()
except:
    HAVE_NCCL = False


def create_result_dir(config_path):
    config_name = os.path.splitext(os.path.basename(config_path))[0]
    result_dir = 'result/{}_{}_0'.format(
        config_name, time.strftime('%Y-%m-%d_%H-%M-%S'))
    while os.path.exists(result_dir):
        i = result_dir.split('_')[-1]
        result_dir = re.sub('_[0-9]+$', result_dir, '_{}'.format(i))
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    return result_dir


def get_model(
        model_file, model_name, model_args, loss_file, loss_name, loss_args,
        result_dir, train=True):
    model = imp.load_source(model_name, model_file)
    model = getattr(model, model_name)

    # Copy model file
    if train:
        dst = '{}/{}'.format(result_dir, os.path.basename(model_file))
        if not os.path.exists(dst):
            shutil.copy(model_file, dst)

    # Initialize
    model = model(**model_args)

    if train:
        loss = imp.load_source(loss_name, loss_file)
        loss = getattr(loss, loss_name)
        if loss_args:
            model = loss(model, **loss_args)
        else:
            model = loss(model)

        # Copy loss file
        dst = '{}/{}'.format(result_dir, os.path.basename(loss_file))
        if not os.path.exists(dst):
            shutil.copy(loss_file, dst)
    return model


def get_optimizer(model, method, optimizer_args, weight_decay=None):
    optimizer = getattr(optimizers, method)(**optimizer_args)
    optimizer.setup(model)
    if weight_decay is not None:
        optimizer.add_hook(chainer.optimizer.WeightDecay(weight_decay))
    return optimizer


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


def create_datasets(config):
    config = config['dataset']
    train_dataset = imp.load_source(
        config['train']['name'], config['train']['file'])
    train_dataset = getattr(train_dataset, config['train']['name'])
    train_dataset = train_dataset(**config['train']['args'])
    valid_dataset = imp.load_source(
        config['valid']['name'], config['valid']['file'])
    valid_dataset = getattr(valid_dataset, config['valid']['name'])
    valid_dataset = valid_dataset(**config['valid']['args'])
    return train_dataset, valid_dataset


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
        updater = training.SerialUpdater(
            train_iter, optimizer, device=devices['main'])
    return updater


def train(args):
    config = yaml.load(open(args.config))

    # Setting random seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    print('chainer version: {}'.format(chainer.__version__))
    print('cuda: {}, cudnn: {}, nccl: {}'.format(
        chainer.cuda.available, chainer.cuda.cudnn_enabled, HAVE_NCCL))
    if args.result_dir is not None:
        config['result_dir'] = args.result_dir
    else:
        config['result_dir'] = create_result_dir(args.config)
    log_fn = save_config_get_log_fn(config['result_dir'], args.config)
    print('result_dir:', config['result_dir'])

    # Instantiate model
    model = get_model(
        config['model']['file'], config['model']['name'],
        config['model']['args'], config['loss']['file'],
        config['loss']['name'], config['loss']['args'],
        config['result_dir'], train=True)

    # Initialize optimizer
    if 'weight_decay' in config['optimizer']:
        weight_decay = config['optimizer']['weight_decay']
    else:
        weight_decay = None
    optimizer = get_optimizer(
        model, config['optimizer']['method'], config['optimizer']['args'],
        weight_decay)

    # Prepare devices
    devices = {'main': args.gpus[0]}
    for gid in args.gpus[1:]:
        devices['gpu{}'.format(gid)] = gid

    # Setting up datasets
    train_dataset, valid_dataset = create_datasets(config)
    print('train: {}'.format(len(train_dataset)))
    print('valid: {}'.format(len(valid_dataset)))

    # Create iterators
    train_iter, valid_iter = create_iterators(
        train_dataset, config['batchsize'], valid_dataset,
        config['valid_batchsize'], devices)

    # Create updater and trainer
    updater = create_updater(train_iter, optimizer, devices)
    trainer = training.Trainer(
        updater, (config['stop_epoch'], 'epoch'), out=config['result_dir'])

    # Add evaluator
    use_classes = config['dataset']['valid']['args']['use_classes']
    trainer.extend(
        TestModeEvaluator(valid_iter, model, device=args.gpus[0]),
        trigger=config['valid_trigger'])

    # Add dump_graph
    trainer.extend(extensions.dump_graph('main/loss'))

    # Add snapshot
    st = config['snapshot_trigger'][1]
    tx = '{' + '.updater.{}'.format(st) + '}'
    trainer.extend(extensions.snapshot(
        filename='snapshot_trainer_{}_{}'.format(st, tx),
        trigger=tuple(config['snapshot_trigger'])))
    trainer.extend(extensions.snapshot_object(
        model.predictor, filename='snapshot_model_{}_{}'.format(st, tx),
        trigger=tuple(config['snapshot_trigger'])))

    # Add Logger
    log_trigger = config['log_trigger']
    trainer.extend(extensions.ProgressBar(
        update_interval=log_trigger[0]), trigger=log_trigger)
    trainer.extend(extensions.LogReport(
        trigger=log_trigger, log_name=log_fn))
    for plot_setting in config['plot_settings']:
        trainer.extend(extensions.PlotReport(
            plot_setting['plot_values'], plot_setting['unit'], log_trigger,
            file_name=plot_setting['file_name']))

    # Values to be printed
    print_values = config['print_values']

    # LR decay
    if 'lr_drop_ratio' in config['optimizer']:
        ratio = config['optimizer']['lr_drop_ratio']
        trigger = config['optimizer']['lr_drop_trigger']

        @training.make_extension(trigger=trigger)
        def learning_rate_dropping(trainer):
            trainer.updater.get_optimizer('main').lr *= ratio

        trainer.extend(extensions.observe_lr())
        trainer.extend(learning_rate_dropping)
        print_values.append('lr')

    trainer.extend(extensions.PrintReport(print_values))

    # Resume
    if args.resume is not None:
        fn = '{}.bak'.format(args.resume)
        shutil.copy(args.resume, fn)
        serializers.load_npz(args.resume, trainer)
        print('Resumed from:', args.resume)

    trainer.run()
    return 0


def test(args):
    print(args)


def main():
    parser = argparse.ArgumentParser(description='ChainerCMD')
    subparsers = parser.add_subparsers()

    # train command
    parser_train = subparsers.add_parser('train', help='Training mode')
    parser_train.add_argument('--config', type=str)
    parser_train.add_argument('--gpus', type=int, nargs='*')
    parser_train.add_argument('--seed', type=int, default=0)
    parser_train.add_argument('--result_dir', type=str, default=None)
    parser_train.add_argument('--resume', type=str, default=None)
    parser_train.set_defaults(handler=train)

    # test command
    parser_test = subparsers.add_parser('test', help='Inference mode')
    parser_test.add_argument('--config', type=str)
    parser_test.add_argument('--gpu', type=int)
    parser_test.add_argument('--snapshot', type=str)
    parser_test.set_defaults(handler=test)

    args = parser.parse_args()

    if hasattr(args, 'handler'):
        args.handler(args)
