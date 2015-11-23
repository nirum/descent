from .utils import coroutine

__all__ = ['printer']


@coroutine
def printer():
    k = 0
    while True:
        k += 1
        data = yield
        print('[{}] Got:'.format(k), data)

