from chainer.training import updater
from chainer.training import updaters
from chainer import training

try:
    HAVE_NCCL = updaters.MultiprocessParallelUpdater.available()
except Exception:
    HAVE_NCCL = False


def updater_creator(iterator, optimizer, devices, **kwargs):
    """A sample updater creator.

    An updater creator method should return an Updater object.
    Once an updter creator method is specified in the config YAML, the method
    will take iterator object, optimizer object, device dictionary, and "args"
    dictionary defined in the config YAML. You can make a custom Updater with
    those objects and return it.

    """
    if HAVE_NCCL and len(devices) > 1:
        updater = training.updaters.MultiprocessParallelUpdater(
            iterator, optimizer, devices=devices)
    elif len(devices) > 1:
        optimizer.lr /= len(devices)
        updater = training.ParallelUpdater(
            iterator, optimizer, devices=devices)
    else:
        updater = training.StandardUpdater(
            iterator, optimizer, device=devices['main'])
    return updater
