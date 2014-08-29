from matplotlib.pyplot import close
import numpy as np
from pylab import imshow, show
from timeit import default_timer as timer


def mandel(x, y, max_iters):
    """
      Given the real and imaginary parts of a complex number,
      determine if it is a candidate for membership in the Mandelbrot
      set given a fixed number of iterations.
    """
    c = complex(x, y)
    z = 0.0j
    for i in range(max_iters):
        z = z * z + c
        if (z.real * z.real + z.imag * z.imag) >= 4:
            return i

    return max_iters


def create_fractal(min_x, max_x, min_y, max_y, image, iters):
    height = image.shape[0]
    width = image.shape[1]

    pixel_size_x = (max_x - min_x) / width
    pixel_size_y = (max_y - min_y) / height

    for x in range(width):
        real = min_x + x * pixel_size_x
        for y in range(height):
            imag = min_y + y * pixel_size_y
            color = mandel(real, imag, iters)
            image[y, x] = color


def run():
    image = np.zeros((2000, 2000), dtype=np.uint8)
    start = timer()
    create_fractal(-2, .5, -1.25, 1.25, image, 2000)

    # create_fractal(-.00001, .000001, -.000001, .000001, image, 20)
    dt = timer() - start

    print("Mandelbrot created in %f s" % dt)

    img = imshow(image.T, origin='lower left')
    img.write_png('mandel5.png', noscale=True)
    close()
