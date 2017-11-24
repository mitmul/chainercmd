MNIST Example
==================

Initialize a project with `chainer init` and then use the default `config.yml` for training the MNIST example.

```
$ chainer init
$ MPLBACKEND=Agg CHAINER_SEED=0 chainer train config.yml --gpus 0
```