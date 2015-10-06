"""
Display options
"""

from __future__ import print_function
import tableprint as tp
from .utils import datum, destruct
from toolz import merge
from numpy.linalg import norm

defaults = {
    'every': 1,
    'iter': True,
    'obj': True,
    'gradnorm': True,
    'runtime': True,
    'width': 15,
    'spec': '5g'
}

class Display(object):

    def __init__(self):
        pass

    def __call__(self, d):
        """
        takes a datum and prints it
        """
        raise NotImplementedError

    def cleanup(self, d, runtimes):
        """
        cleanup!

        takes; current (final) datum and runtimes
        """
        raise NotImplementedError

class Ascii(Display):

    def __init__(self, **kwargs):

        opts = merge(defaults, kwargs)

        self.every = opts['every']
        self.width = opts['width']
        self.spec = opts['spec']
        self.column_names = []
        self.columns = []

        if opts['iter']:
            self.column_names.append('Iteration')
            self.columns.append(lambda d: d.iteration)

        if opts['obj']:
            self.column_names.append('Objective')
            self.columns.append(lambda d: d.obj)

        if opts['gradnorm']:
            self.column_names.append('||Gradient||')
            self.columns.append(lambda d: norm(destruct(d.grad)))

        if opts['runtime']:
            self.column_names.append('Runtime')
            self.columns.append(lambda d: tp.humantime(d.runtime))

        self.ncols = len(self.column_names)

    def _transform(self, d):
        return [f(d) for f in self.columns]

    @property
    def hr(self):
        return tp.hr(self.ncols, self.width)

    def start(self):
        print('\n'.join((self.hr,
                        tp.header(self.column_names, self.width),
                        self.hr)), flush=True)


    def __call__(self, d):

        # iteration update
        if d.iteration % self.every == 0:
            print(tp.row(self._transform(d), self.width, format_spec=self.spec), flush=True)

    def cleanup(self, d, runtimes):

        print(self.hr)
        print(u'\u279b Final objective: {}'.format(d.obj))
        print(u'\u279b Total runtime: {}'.format(tp.humantime(sum(runtimes))))
        print(u'\u279b All done!\n')
