#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import unittest
from collections import namedtuple

import yaml

from chainercmd import template
from chainercmd import train


class TestTrain(unittest.TestCase):

    def setUp(self):
        config_fn = 'examples/config.yml'

        args = namedtuple(
            'args', ['config', 'gpus', 'seed', 'result_dir', 'resume'])
        self.args = args(config=config_fn,
                         gpus=[-1],
                         seed=0,
                         result_dir=None,
                         resume=None)

    def test_train(self):
        train.train(self.args)
