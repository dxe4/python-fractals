from copy import copy
import numpy as np
cimport numpy as np
import time
from matplotlib.pyplot import imshow, show, cm, close
from cython.parallel import prange
import cython
import math

# pragne(1, nogil=true)

cpdef double _target_x = -0.9223327810370947027656057193752719757635
cpdef double _target_y = 0.3102598350874576432708737495917724836010
cdef double _length = 2.5 / 2
cdef double scale = 0.01


cdef int calculate(double x, double y, int iterations):
    cdef double z_x = x
    cdef double z_y = y
    cdef int i

    for i in range(iterations):
        z_x, z_y = z_x ** 2 - z_y ** 2 + x, 2 * z_x * z_y + y
        if z_x ** 2 + z_y ** 2 >= 4.0:
            break
    else:
        i = -1
    # math.log(i + 1 - math.log(abs(z_x + z_y)) / math.log(2))
    return i


@cython.boundscheck(False)
cdef int[:, ::1] generate(double[::1] xs, double[::1] ys, int iterations):
    '''
    [::1] -> 1d array (cython memoryview)
    [:, ::1] -> 2d array
    '''
    cdef int i, j
    cdef int M = len(ys)
    cdef int N = len(xs)

    cdef int[:, ::1] d = np.empty(shape=(M, N), dtype=np.int32)
    #with nogil:
    for i in range(M):
        for j in range(N):
            d[j, i] = calculate(xs[j], ys[i], iterations)
    return d

# Move rest of the functions to a py file

cdef void save(int[:, ::1] arr, int count):
    # spectral cmap="hot"
    img = imshow(arr.T, origin='lower left', cmap=cm.gist_stern)
    # 'abc/abc_%05d.png' % count
    img.write_png('abc/abc_%05d.png' % count, noscale=True)
    close()


# cdef int[:, ::1] _run(double x, double y, int n):
#     cdef int[:, ::1] d = generate(x, y, n)
#     return d


cpdef get_initial_input():
    cpdef double x1 = -2.0
    cpdef double x2 = 0.5
    cpdef double y1 = -1.3
    cpdef double y2 = 1.3

    return x1, x2, y1, y2, 2000, 2000j


cdef list center_point(double x1, double x2, double y1, double y2,
                       double target_x, double target_y, double length):
    cdef double x, y, x_min, x_max, y_min, y_max

    x_min = target_x - 0.5 * length
    x_max = target_x + 0.5 * length
    y_min = target_y - 0.5 * length
    y_max = target_y + 0.5 * length

    return [x_min, y_min, y_min, y_max]

cdef list find_zoom_edges(double[::1] x, double[::1] y, double scale, int n):
    cdef double new_x1, new_x2, new_y1, new_y2
    cdef int pos
    pos = int(scale * n)

    new_x1 = x[pos]
    new_y1 = y[pos]
    new_x2 = x[n - pos]
    new_y2 = y[n - pos]

    return [new_x1, new_y1, new_x2, new_y2]


cpdef run(double x1, double x2, double y1, double y2, int n, complex nj):
    t = time.time()
    # I = mandel(400, 400, 100, -2, .5, -1.25, 1.25)
    cdef int[:, ::1] d
    cdef int i
    cdef double[::1] x
    cdef double[::1] y
    cdef int iterations = 800

    x1, x2, y1, y2 = center_point(x1, x2, y1, y2, _target_x, _target_y,
                                  _length)
    for i in range(150):
        x = np.r_[x1:x2:nj]
        y = np.r_[y1:y2:nj]
        d = generate(x, y, iterations)
        print('Exec time {}'.format(time.time() - t))
        save(d, i)
        x1, y1, x2, y2 = find_zoom_edges(x, y, scale, n)
        iterations += 1


def initial_run():
    args = get_initial_input()
    run(*args)

'''
cython: profile=True
cython: cdivision=True

cython -a foo.pyx
cnp.ndarray[cnp.float64, ndim=2] arr

imshow(d, extent=[x_a, x_b, y_a, y_b], cmap=cm.gist_stern)
show()
run(-2.13, 0.77, -1.3, 1.3, 100, 2000j)

cdef extern from "math.h":
    double cos(double x)
    double sin(double x)
    double tan(double x)

    double M_PI
    # libraries=["m"] (ext_modules setup.py)
'''
