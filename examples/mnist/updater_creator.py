from chainer.training import updater


def updater_creator(iterator, optimizer, devices, *args, **kwargs):
    print(args)
    print(kwargs)
    return updater.StandardUpdater(iterator, optimizer, device=devices['main'])
