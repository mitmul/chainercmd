class ConfigBase(object):

    def __init__(self, required_keys, optional_keys, kwargs, name):
        for key in required_keys:
            if key not in kwargs:
                raise KeyError(
                    '{} config should have the key {}'.format(name, key))
            setattr(self, key, kwargs[key])
        for key in optional_keys:
            if key in kwargs:
                setattr(self, key, kwargs[key])
            elif key == 'args':
                setattr(self, key, {})
            else:
                setattr(self, key, None)
