#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

from chainercmd import init
from chainercmd import test
from chainercmd import train


def main():
    parser = argparse.ArgumentParser(description='ChainerCMD')
    subparsers = parser.add_subparsers()

    # train command
    parser_train = subparsers.add_parser('train', help='Training mode')
    parser_train.add_argument('config', type=str)
    parser_train.add_argument('--gpus', type=int, nargs='*', default=[-1])
    parser_train.add_argument('--seed', type=int, default=0)
    parser_train.add_argument('--result_dir', type=str, default=None)
    parser_train.add_argument('--resume', type=str, default=None)
    parser_train.set_defaults(handler=train.train)

    # test command
    parser_test = subparsers.add_parser('test', help='Inference mode')
    parser_test.add_argument('config', type=str)
    parser_test.add_argument('--snapshot', type=str)
    parser_test.add_argument('--gpu', type=int, default=-1)
    parser_test.set_defaults(handler=test.test)

    # init command
    parser_init = subparsers.add_parser('init', help='Inference mode')
    parser_init.add_argument(
        '--create_subdirs', action='store_true', default=False,
        help='If you want to create subdirs ("model", "loss", "dataset"), '
             'give this flag.')
    parser_init.set_defaults(handler=init.init)

    args = parser.parse_args()

    if hasattr(args, 'handler'):
        args.handler(args)
