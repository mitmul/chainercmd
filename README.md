# ChainerCMD
![pypi](https://img.shields.io/pypi/v/chainercmd.svg)
![Build Status](https://travis-ci.org/mitmul/chainercmd.svg?branch=master)
![MIT License](https://img.shields.io/github/license/mitmul/chainercmd.svg)

You can write all configuration of training in a YAML file, and start training with one line:

```bash
$ chainer train config.yml
```

## Installation

```bash
$ pip install chainercmd
```

## Requirement

- Python 3.4.4+
- Chainer 3.0.0b1+
- PyYAML 3.12+

## Create new project

```
$ chainer init
```

It produces the below files

- config.yml
- model.py
- loss.py
- dataset.py

You can modify these files and start training by running

```bash
$ MPLBACKEND=Agg chainer train config.yml --gpus 0
```

See the details by giving `--help` argument to the subcommand:

```bash
$ chainer train --help
```
