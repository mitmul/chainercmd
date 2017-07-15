class Loss(object):

    def __init__(self, **kwargs):
        required_keys = [
            'file',
            'name',
        ]
        for key in required_keys:
            if key not in kwargs:
                raise KeyError(
                    'loss config should have the key {}'.format(key))
            setattr(self, key, kwargs[key])

        optional_keys = [
            'args'
        ]
        for key in optional_keys:
            if key in kwargs:
                setattr(self, key, kwargs[key])
            else:
                setattr(self, key, None)
