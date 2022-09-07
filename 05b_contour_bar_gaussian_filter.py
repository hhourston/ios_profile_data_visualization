# import numpy as np
# from scipy import signal, misc
from imageio.v2 import imread, imsave
from scipy.ndimage import gaussian_filter
import glob
import os

"""Credit:
https://stackoverflow.com/questions/42864823/how-to-smoothen-2d-color-map-in-matplotlib
"""


def gaussian_cbar_blur(img_path, output_dir):
    img = imread(img_path)

    # print(img.shape)

    #I used msPaint to get coords... there's probably a better way
    # x0, y0, x1, y1 = 87,215,764,1270 #chart area (pixel coords)
    x0, y0, x1, y1 = 37, 849, 543, 872 #chart area (pixel coords)

    #you could use a gaussian filter to get a rounder blur pattern
    # kernel = np.ones((5,5),)/25 #mean value convolution

    # #convolve roi with averaging filter
    # #red
    # img[x0:x1, y0:y1, 0] = signal.convolve2d(
    #     img[x0:x1, y0:y1, 0], kernel, mode='same', boundary='symm')
    # #green
    # img[x0:x1, y0:y1, 1] = signal.convolve2d(
    #     img[x0:x1, y0:y1, 1], kernel, mode='same', boundary='symm')
    # #blue
    # img[x0:x1, y0:y1, 2] = signal.convolve2d(
    #     img[x0:x1, y0:y1, 2], kernel, mode='same', boundary='symm')

    # Try gaussian filter instead
    # red
    img[x0:x1, y0:y1, 0] = gaussian_filter(
        img[x0:x1, y0:y1, 0], 20, mode='nearest') #sigma=4 originally
    # blue
    img[x0:x1, y0:y1, 1] = gaussian_filter(
        img[x0:x1, y0:y1, 1], 20, mode='nearest')
    # green
    img[x0:x1, y0:y1, 2] = gaussian_filter(
        img[x0:x1, y0:y1, 2], 20, mode='nearest')

    #do it again for legend area
    #...
    output_path = os.path.join(
        output_dir, os.path.basename(img_path).replace('.png', '_gaussian2.png'))
    imsave(output_path, img)
    return


contour_plot_path = 'C:\\Users\\HourstonH\\Documents\\ctd_visualization\\' \
                    'png\\0075_latlon\\'

output_folder = contour_plot_path + 'blurred_cbar\\'
# gaussian_cbar_blur(contour_plot_path)

plot_list = glob.glob(contour_plot_path + '*contourf*.png')

for pl in plot_list:
    gaussian_cbar_blur(pl, output_folder)