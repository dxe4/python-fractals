import numpy as np
from matplotlib.pyplot import imshow, close
from numpy import copy, multiply, add
import time
import math
import os
import sys
import os

from  concurrent.futures import ProcessPoolExecutor
from multiprocessing.pool import Pool
# http://thesamovar.wordpress.com/2009/03/22/fast-fractals-with-python-and-numpy/

def mandel(img_width, img_height, itermax, xmin, xmax, ymin, ymax):
    '''
    Fast mandelbrot computation using numpy.

    (img_width, img_height) are the output image dimensions
    itermax is the maximum number of iterations to do
    xmin, xmax, ymin, ymax specify the region of the set to compute.
    '''
    # The point of ix and iy is that they are 2D arrays giving the x-coord and y-coord at each point in
    # the array. The reason for doing this will become clear below...
    #
    # 2x 2D arrays size=W*H elms = [0,0,0...],[1,1,1...],[n-2,n-2,n-1...], [n-1,n-1,n-1...]
    ix, iy = np.mgrid[0:img_width, 0:img_height]
    # Now x and y are the x-values and y-values at each point in the array, linspace(start, end, n)
    # is an array of n linearly spaced points between  start and end, and we then index this array using
    # numpy fancy indexing. If A is an array and I is an array of indices, then A[I] has the same shape
    # as I and at each place i in I has the value A[i].
    #
    # 2x 2D arrays size= W*H
    # W arrays elms = [xmin, xmin+c, xmin+2c... xmax], [xmin, xmin+c, xmin+2c... xmax],
    x = np.linspace(xmin, xmax, img_width)[ix]
    y = np.linspace(ymin, ymax, img_height)[iy]

    # c is the complex number with the given x, y coords
    # 2D array W*H
    # add all values of x with all values of y*i
    # elms = [x[0]y[0]*complex(0,1)...x[n-1]y[n-1]*complex(0,1)]
    c = x + complex(0, 1) * y
    del x, y  # save a bit of memory, we only need z
    # the output image coloured according to the number
    # of iterations it takes to get to the boundary
    # abs(z)>2
    img = np.zeros(c.shape, dtype=np.uint8)
    # Here is where the improvement over the standard
    # algorithm for drawing fractals in numpy comes in.
    # We flatten all the arrays ix, iy and c. This
    # flattening doesn't use any more memory because
    # we are just changing the shape of the array, the
    # data in memory stays the same. It also affects
    # each array in the same way, so that index i in
    # array c has x, y coords ix[i], iy[i]. The way the
    # algorithm works is that whenever abs(z)>2 we
    # remove the corresponding index from each of the
    # arrays ix, iy and c. Since we do the same thing
    # to each array, the correspondence between c and
    # the x, y coords stored in ix and iy is kept.
    ix.shape = img_width * img_height
    iy.shape = img_width * img_height
    c.shape = img_width * img_height
    # we iterate z->z^2+c with z starting at 0, but the
    # first iteration makes z=c so we just start there.
    # We need to copy c because otherwise the operation
    # z->z^2 will send c->c^2.
    z = copy(c)
    for i in range(itermax):
        if not len(z):
            break  # all points have escaped
        # equivalent to z = z*z+c but quicker and uses
        # less memory
        f = z[0]
        print(f.real, f.imag)
        multiply(z, z, z)
        add(z, c, z)
        # these are the points that have escaped
        rem = abs(z) > 2.0

        # colour them with the iteration number, we
        # add one so that points which haven't
        # escaped have 0 as their iteration number,
        # this is why we keep the arrays ix and iy
        # because we need to know which point in img
        # to colour
        img[ix[rem], iy[rem]] = i + 1

        # print()
        # -rem is the array of points which haven't
        # escaped, in numpy -A for a boolean array A
        # is the NOT operation.
        rem = -rem
        # So we select out the points in
        # z, ix, iy and c which are still to be
        # iterated on in the next step
        z = z[rem]
        ix, iy = ix[rem], iy[rem]
        c = c[rem]
    return img


def foo(W, H, iter):
    # http://batchloaf.wordpress.com/2012/12/15/visualising-the-mandelbrot-set/
    x1, y1, r1 = -1.339623, 0.071429988, 0.00000000009
    x2, y2, r2 = -1.339623, 0.071429988, 0.00000000009
    N = 150

    rscale = pow(r2 / r1, 1 / float(N - 1))

    for n in range(N):
        x = (1 - n / float(N - 1)) * x1 + (n / float(N - 1)) * x2
        y = (1 - n / float(N - 1)) * y1 + (n / float(N - 1)) * y2
        r = r1 * math.pow(rscale, n)
        x_min = x - 0.5 * r
        x_max = x + 0.5 * r
        y_min = y - 0.5 * H * r / W
        y_max = y + 0.5 * H * r / W
        yield W, H, iter, x_min, x_max, y_min, y_max


def run(args):
    I = mandel(*args)
    I[I == 1] = 3
    I[I == 0] = 1
    return I


def save(arr, count):
    img = imshow(arr.T, origin='lower left', cmap="spectral")
    img.write_png('abc/abc_%03d.png' % count, noscale=True)
    close()


if __name__ == '__main__':
    # convert  abc* ms.gif
    gen = foo(600, 600, 100)
    for count, i in enumerate(gen):
        start = time.time()
        arr = run(i)
        save(arr, count)
        print(i)
        print('Time taken: {}'.format(str(time.time() - start)))
        break

