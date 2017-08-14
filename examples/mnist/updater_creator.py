from chainer.training import updater


def updater_creator(iterator, optimizer, devices, *args, **kwargs):
    return updater.StandardUpdater(iterator, optimizer, device=devices['main'])
