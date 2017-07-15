import yaml
import os

from chainercmd.template import dataset
from chainercmd.template import loss
from chainercmd.template import model

dname = os.path.dirname(__file__)
config_base = yaml.load(open('{}/config.yml'.format(dname)))
