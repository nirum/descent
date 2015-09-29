"""
Storage options
"""

from .utils import datum
from toolz import merge, keyfilter

defaults = {
    'every': 1,
    'iter': True,
    'obj': True,
    'grad': False,
    'params': False
}

class Storage(object):

    def __init__(self):
        pass

    def __call__(self, d):
        """
        takes a datum and prints it
        """
        raise NotImplementedError

    def cleanup(self):
        """
        cleanup!
        """
        raise NotImplementedError

class List(list, Storage):

    def __init__(self, **kwargs):

        opts = merge(defaults, kwargs)
        self.every = opts['every']
        self.keys = []

        if opts['iter']:
            self.keys.append('iteration')

        if opts['obj']:
            self.keys.append('obj')

        if opts['grad']:
            self.keys.append('grad')

        if opts['params']:
            self.keys.append('params')

    def __call__(self, d):

        self.append(keyfilter(lambda k: k in self.keys, d._asdict()))
