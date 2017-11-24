# ChainerCMD
![pypi](https://img.shields.io/pypi/v/chainercmd.svg)
![Build Status](https://travis-ci.org/mitmul/chainercmd.svg?branch=master)
![MIT License](https://img.shields.io/github/license/mitmul/chainercmd.svg)

ChainerCMD is a project instantiation tool for Chainer.

## Installation

```bash
$ pip install chainercmd
```

## Requirement

- Python>=3.6.2+
- Chainer>=3.1.0
- PyYAML>=3.12

## Quick Start

```
$ chainer init
```

It produces the below files

- config.yml
- custom_extension.py
- dataset.py
- evaluator_creator.py
- loss.py
- model.py
- updater_creator.py

You can modify these files and start training by

```bash
$ MPLBACKEND=Agg chainer train config.yml --gpus 0
```

See the details by giving `--help` argument to the subcommand:

```bash
$ chainer train --help
```
