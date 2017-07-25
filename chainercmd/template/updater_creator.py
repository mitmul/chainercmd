from chainer.training import updater


def updater_creator(iterator, optimizer, devices, **kwargs):
    updater = updater.StandardUpdater(
        iterator, optimizer, device=devices['main'])
    return updater
