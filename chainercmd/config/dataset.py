class Dataset(object):

    def __init__(self, **kwargs):
        required_keys = [
            {'train': {'file': None, 'name': None, 'args': None}}
        ]

        for key in required_keys:
            if key not in kwargs:
                raise KeyError(
                    'dataset config should have the key {}'.format(key))
            for kk in kwargs[key]:
                required_keys[key][kk] = kwargs[key][kk]
            setattr(self, key, required_keys[key])

        optional_keys = [
            {'valid': ['file', 'name', 'args']},
            {'test': ['file', 'name', 'args']}
        ]
