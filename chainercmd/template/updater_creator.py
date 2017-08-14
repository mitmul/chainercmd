from chainer.training import updater


def updater_creator(iterator, optimizer, devices, **kwargs):
    """A sample updater creator.

    You need to return an Updater object in a updater creator method.
    Once a updter creator method is specified in the config YAML, the method
    will take iterator object, optimizer object, device dictionary, and "args"
    dictionary defined in the config YAML. You can make a custom Updater with
    those objects and return it.

    """
    return updater.StandardUpdater(
        iterator, optimizer, device=devices['main'])
