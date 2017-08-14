import yaml
import os

from chainercmd.template import custom_extension  # NOQA
from chainercmd.template import dataset  # NOQA
from chainercmd.template import loss  # NOQA
from chainercmd.template import model  # NOQA
from chainercmd.template import updater_creator  # NOQA

dname = os.path.dirname(__file__)
config_base = yaml.load(open('{}/config.yml'.format(dname)))
