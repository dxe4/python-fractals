import numpy as np
import time
from matplotlib.pyplot import imshow, show, cm, close
from cython.parallel import prange
import cython

# pragne(1, nogil=true)

cpdef double _target_x = -0.9223327810370947027656057193752719757635
cpdef double _target_y = 0.3102598350874576432708737495917724836010
cdef double _length = 2.5


cdef int calculate(double x, double y, int n) nogil:
    cdef double z_x = x
    cdef double z_y = y
    cdef int i

    for i in range(n):
        z_x, z_y = z_x ** 2 - z_y ** 2 + x, 2 * z_x * z_y + y
        if z_x ** 2 + z_y ** 2 >= 4.0:
            break
    else:
        i = -1
    return i


@cython.boundscheck(False)
cdef int[:, ::1] generate(double[::1] xs, double[::1] ys, int n):
    '''
    [::1] -> 1d array (cython memoryview)
    [:, ::1] -> 2d array
    '''
    cdef int i, j
    cdef int M = len(ys)
    cdef int N = len(xs)

    cdef int[:, ::1] d = np.empty(shape=(M, N), dtype=np.int32)
    with nogil:
        for i in prange(M):
            for j in prange(N):
                d[j, i] = calculate(xs[j], ys[i], n)
    return d


cdef void save(int[:, ::1] arr, int count):
    # spectral cmap="hot"
    img = imshow(arr.T, origin='lower left', cmap=cm.gist_stern)
    # 'abc/abc_%05d.png' % count
    img.write_png('abc/abc_%05d.png' % count, noscale=True)
    close()
    pass


cdef int[:, ::1] _run(double x1, double x2, double y1, double y2, int n,
                      complex nj):
    cdef double[::1] x = np.r_[x1:x2:nj]
    cdef double[::1] y = np.r_[y1:y2:nj]
    cdef int[:, ::1] d = generate(x, y, n)
    return d


cpdef get_initial_input():
    cpdef double x1 = -2.0
    cpdef double x2 = 0.5
    cpdef double y1 = -1.3
    cpdef double y2 = 1.3

    return x1, x2, y1, y2, 1000, 1000j


cdef list center_point(double x1, double x2, double y1, double y2,
                       double target_x, double target_y, double length):
    cdef double x, y, x_min, x_max, y_min, y_max

    x_diff = abs(x1 - x2)
    y_diff = abs(y1 - y2)

    x_min = target_x - 0.5 * length
    x_max = target_x + 0.5 * length
    y_min = target_y - 0.5 * length
    y_max = target_y + 0.5 * length

    return [x_min, x_max, y_min, y_max]

cdef zoom(double scale):
    pass


cpdef run(double x1, double x2, double y1, double y2, int n, complex nj):
    t = time.time()
    # I = mandel(400, 400, 100, -2, .5, -1.25, 1.25)
    cdef int[:, ::1] d
    cdef int i
    cdef double scale = 0.5

    x1, x2, y1, y2 = center_point(x1, x2, y1, y2, _target_x, _target_y,
                                  _length)
    for i in range(5):
        d = _run(x1, x2, y1, y2, n, nj)
        print('Exec time {}'.format(time.time() - t))
        save(d, i)


def initial_run():
    args = get_initial_input()
    run(*args)

    # imshow(d, extent=[x_a, x_b, y_a, y_b], cmap=cm.gist_stern)
    # show()
 # run(-2.13, 0.77, -1.3, 1.3, 100, 2000j)
