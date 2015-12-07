from collections import defaultdict
import tableprint as tp

__all__ = ['printer', 'store']


def printer(*keys, every=1, width=15):

    while True:

        data = yield
        curiter = data.get('iteration', -1)

        if data.get('cleanup', False):
            print(tp.hr(len(keys), column_width=width), flush=True)
            yield None

        if curiter == 0:
            print(tp.header(keys, column_width=width), flush=True)

        if curiter % every == 0:

            values = [data.get(key, 'Not found') for key in keys]
            print(tp.row(list(map(float, values)), column_width=width), flush=True)


def store(*keys, every=1):

    cache = defaultdict(list)

    while True:

        data = yield
        curiter = data.get('iteration', -1)

        if data.get('cleanup', False):
            yield cache

        if curiter % every == 0:
            for key in keys:
                if key in data:
                    cache[key].append(data[key])
