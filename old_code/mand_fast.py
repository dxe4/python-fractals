import numpy as np
from matplotlib.pyplot import imshow, close
from numpy import copy, multiply, add
import time
import math
from numpy import float128
import random

big_number = float128(0.66778899666677889966667788996666778899666677889966)
# big_number = float128(0.55544433322221111111111111111111111111111111111111111111111111111111)
# http://thesamovar.wordpress.com/2009/03/22/fast-fractals-with-python-and-numpy/


def mandel(img_width, img_height, itermax, iteration, xmin, xmax, ymin, ymax):
    # 2x 2D arrays size=W*H elms = [0,0,0...],[1,1,1...],[n-2,n-2,n-1...], [n-1,n-1,n-1...]
    ix, iy = np.mgrid[0:img_width, 0:img_height]
    # 2x 2D arrays size= W*H
    # W arrays elms = [xmin, xmin+c, xmin+2c... xmax], [xmin, xmin+c, xmin+2c... xmax],
    x = np.linspace(xmin, xmax, img_width)[ix]
    y = np.linspace(ymin, ymax, img_height)[iy]
    # 2D array W*H
    # add all values of x with all values of y*i
    # elms = [x[0]y[0]*complex(0,1)...x[n-1]y[n-1]*complex(0,1)]
    c = x + complex(0, 1) * y
    img = np.zeros(c.shape, dtype=np.uint8)
    ix.shape = img_width * img_height
    iy.shape = img_width * img_height
    c.shape = img_width * img_height
    z = copy(c)
    for i in range(itermax):
        if not len(z):
            break
        multiply(z, z, z)
        add(z, c, z)
        rem = abs(z) > 2.0
        img[ix[rem], iy[rem]] = i + 1 + int(iteration - iteration*0.9)
        new_iter = np.where(img==i+1)
        if len(new_iter) == 2 and len(new_iter[0] > 0) and len(new_iter[1] > 0):
            last_iter = new_iter
        rem = -rem
        z = z[rem]
        ix, iy = ix[rem], iy[rem]
        c = c[rem]

    x_index, y_index = last_iter[0][0], last_iter[1][0]

    # possible_x_zoom = [(i[0], i[0] - (xmax-xmin)) for i in x[last_iter[0]]]
    # possible_y_zoom = [(i[y_index], - (ymax-ymin)) for i in y]
    # #  x[x_index][0], y[0][y_index]
    # mx = min(possible_x_zoom, key=lambda k: k[1])[0]
    # my = min(possible_y_zoom, key=lambda k: k[1])[0]
    r = random.randint(0, img_height-1)
    return img, x[x_index][r], y[r][y_index]


def get_input(W, H, iter, iter2, x1, y1, x2, y2, r1, r2):
    # http://batchloaf.wordpress.com/2012/12/15/visualising-the-mandelbrot-set/
    # x1, y1, r1 = -1.339623, 0.071429988, 2
    # x2, y2, r2 = -1.339623, 0.071429988, 0.00000000009
    # x1, x2 = x, x
    # y1, y2 = y, y
    # x1, y1, r1 = -1.76960793855, -0.00251916221504, 0.009
    # x2, y2, r2 = -1.76960793855, -0.00251916221504, 0.00000000009
    # -1.75920978129 0.000175114702115
    N = 3

    rscale = pow(r2 / r1, 1 / float(N - 1))

    for n in range(N-1):
        x = (1 - n / float(N - 1)) * x1 + (n / float(N - 1)) * x2
        y = (1 - n / float(N - 1)) * y1 + (n / float(N - 1)) * y2
        r = r1 * math.pow(rscale, n)
        x_min = x - 0.5 * r
        x_max = x + 0.5 * r
        y_min = y - 0.5 * H * r / W
        y_max = y + 0.5 * H * r / W
        yield W, H, iter, iteration, x_min, x_max, y_min, y_max


def run(args):
    I, x, y = mandel(*args)
    I[I == 1] = 3
    I[I == 0] = 1
    return I, x, y


def save(arr, count):
    # spectral cmap="hot"
    img = imshow(arr.T, origin='lower left', cmap="hot")
    img.write_png('abc/abc_%05d.png' % count, noscale=True)
    close()


if __name__ == '__main__':
    # convert  abc* ms.gif
    x1, y1 = -1.339623, 0.071429988
    x2, y2 = x1, y1

    r1 = 4.141234
    r2 = r1 * big_number
    count_all = 0
    to_iter = 150
    for iteration in range(1, 1220):
        gen = get_input(2048, 2048, to_iter, iteration, x1, y1, x2, y2, r1, r2)
        x1, y1 = x2, y2
        for count, i in enumerate(gen):
            start = time.time()
            # if iteration % 35 == 0 or iteration % 65 == 0:
            #     to_iter+=1
            arr, x2, y2 = run(i)
            if not np.unique(arr).size > 1:
                raise Exception
            save(arr, count_all)
            count_all += 1
        r1 = r2
        r2 = r1 * big_number
#ffmpeg -f image2 -r 20 -pattern_type glob -i '*.png' output.mp4
