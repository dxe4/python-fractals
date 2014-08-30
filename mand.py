import time
from matplotlib.pyplot import imshow, cm, close
import mand
import numpy as np


_target_x = -0.9223327810370947027656057193752719757635
_target_y = 0.31025983508745764327087374959177248360100
_length = 2.5 / 2
scale = 0.01
half = 0.5


def save(arr, count):
    # spectral cmap="hot"
    img = imshow(arr.T, origin='lower left', cmap=cm.gist_stern,
                 interpolation='nearest')
    img.write_png('abc/abc_%05d.png' % count, noscale=True)
    close()


def get_initial_input():
    x1 = -2.0
    x2 = 0.5
    y1 = 1.3
    y2 = 1.3
    resolution = 1500
    iterations = 1000
    return x1, x2, y1, y2, resolution, iterations


def center_point(x1, x2, y1, y2, target_x, target_y, length):
    x_min = target_x - half * length
    x_max = target_x + half * length
    y_min = target_y - half * length
    y_max = target_y + half * length

    return [x_min, x_max, y_min, y_max]


def find_zoom_edges(x, y, scale, n):
    pos = int(scale * n)

    new_x1 = x[pos]
    new_y1 = y[pos]
    new_x2 = x[n - pos]
    new_y2 = y[n - pos]

    return [new_x1, new_y1, new_x2, new_y2]


def run(x1, x2, y1, y2, n, iterations):
    nj = complex(n)
    # I = mandel(400, 400, 100, -2, .5, -1.25, 1.25)
    x1, x2, y1, y2 = center_point(x1, x2, y1, y2, _target_x, _target_y,
                                  _length)
    for i in range(150):
        t = time.time()
        # x = np.arange(x1, x2, (x2-x1) / n, dtype=np.dtype(Decimal))
        # y = np.arange(y1, y2, (y2-y1) / n, dtype=np.dtype(Decimal))
        x = np.r_[x1:x2:nj]
        y = np.r_[y1:y2:nj]

        d = mand.generate(x, y, iterations)
        print('Exec time {}'.format(time.time() - t))
        save(d, i)
        x1, y1, x2, y2 = find_zoom_edges(x, y, scale, n)


if __name__ == '__main__':
    args = get_initial_input()
    run(*args)
