# cython: cdivision=True
import numpy as np
from cython.parallel import prange
import cython
from libc.math cimport log

# pragne(1, nogil=true)

cdef extern from "mand_gmp.c":
    object _choose2 "choose2" (unsigned char[] x, unsigned char[] y)


def choose(x, y):
    return _choose2(x, y)


cdef double calculate(double x, double y, int iterations) nogil:
    cdef:
        double z_x = x
        double z_y = y
        double zn, nu
        double i = 0

    while i <= iterations:
        z_x, z_y = z_x ** 2 - z_y ** 2 + x, 2 * z_x * z_y + y
        if z_x ** 2 + z_y ** 2 >= 4.0:
            zn = (z_x * z_x + z_y * z_y) ** 2
            nu = log(log(zn)) / log(2)
            return (i + 1 - nu) / iterations * 255
        i = i + 1
    return -1


@cython.boundscheck(False)
cpdef double[:, ::1] generate(double[::1] xs, double[::1] ys, int iterations):
    '''
    [::1] -> 1d array (cython memoryview)
    [:, ::1] -> 2d array
    '''
    cdef:
        int i, j
        int M = len(ys)
        int N = len(xs)

    cdef double[:, ::1] d = np.empty(shape=(M, N), dtype=np.double)
    with nogil:
        for i in prange(M):
            for j in prange(N):
                d[j, i] = calculate(xs[j], ys[i], iterations)
    return d

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
