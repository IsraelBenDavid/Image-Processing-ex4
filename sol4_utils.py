from scipy.signal import convolve2d
import numpy as np
import skimage
from imageio import imread
from scipy.signal import convolve as convolve1d
from scipy.ndimage.filters import convolve

from skimage.color import rgb2gray

R, G, B = 0, 1, 2
MIN_IMG_SIZE = 16


def gaussian_kernel(kernel_size):
    conv_kernel = np.array([1, 1], dtype=np.float64)[:, None]
    conv_kernel = convolve2d(conv_kernel, conv_kernel.T)
    kernel = np.array([1], dtype=np.float64)[:, None]
    for i in range(kernel_size - 1):
        kernel = convolve2d(kernel, conv_kernel, 'full')
    return kernel / kernel.sum()


def blur_spatial(img, kernel_size):
    kernel = gaussian_kernel(kernel_size)
    blur_img = np.zeros_like(img)
    if len(img.shape) == 2:
        blur_img = convolve2d(img, kernel, 'same', 'symm')
    else:
        for i in range(3):
            blur_img[..., i] = convolve2d(img[..., i], kernel, 'same', 'symm')
    return blur_img


def read_image(filename, representation):
    """
    Reads an image and converts it into a given representation
    :param filename: filename of image on disk
    :param representation: 1 for greyscale and 2 for RGB
    :return: Returns the image as an np.float64 matrix normalized to [0,1]
    """
    image = imread(filename)
    image = image.astype(np.float64)
    image /= 255
    if representation == 1:
        image = rgb2gray(image)
    return image


def reduce(im, blur_filter):
    """
    Reduces an image by a factor of 2 using the blur filter
    :param im: Original image
    :param blur_filter: Blur filter
    :return: the downsampled image
    """
    return convolve(convolve(im, blur_filter), blur_filter.T)[::2, ::2]


def expand(im, blur_filter):
    """
    Expand an image by a factor of 2 using the blur filter
    :param im: Original image
    :param blur_filter: Blur filter
    :return: the expanded image
    """
    expanded_image = np.zeros((im.shape[0] * 2, im.shape[1] * 2), dtype=im.dtype)
    expanded_image[::2, ::2] = im
    return convolve(convolve(expanded_image, 2 * blur_filter), 2 * blur_filter.T)


def _get_filer_row(filter_size):
    filter_row = np.array([1])
    base_row = np.array([1, 1])
    for i in range(filter_size - 1):
        filter_row = convolve1d(filter_row, base_row)
    return np.array([filter_row / filter_row.sum()])


def build_gaussian_pyramid(im, max_levels, filter_size):
    """
    Builds a gaussian pyramid for a given image
    :param im: a grayscale image with double values in [0, 1]
    :param max_levels: the maximal number of levels in the resulting pyramid.
    :param filter_size: the size of the Gaussian filter
            (an odd scalar that represents a squared filter)
            to be used in constructing the pyramid filter
    :return: pyr, filter_vec. Where pyr is the resulting pyramid as a
            standard python array with maximum length of max_levels,
            where each element of the array is a grayscale image.
            and filter_vec is a row vector of shape (1, filter_size)
            used for the pyramid construction.
    """
    pyr = [im]
    filter_vec = _get_filer_row(filter_size)
    for i in range(max_levels - 1):
        if pyr[-1].shape[0] <= MIN_IMG_SIZE or pyr[-1].shape[1] <= MIN_IMG_SIZE:
            break
        pyr.append(reduce(pyr[-1], filter_vec))
    return pyr, filter_vec


def build_laplacian_pyramid(im, max_levels, filter_size):
    """
    Builds a laplacian pyramid for a given image
    :param im: a grayscale image with double values in [0, 1]
    :param max_levels: the maximal number of levels in the resulting pyramid.
    :param filter_size: the size of the Gaussian filter
            (an odd scalar that represents a squared filter)
            to be used in constructing the pyramid filter
    :return: pyr, filter_vec. Where pyr is the resulting pyramid as a
            standard python array with maximum length of max_levels,
            where each element of the array is a grayscale image.
            and filter_vec is a row vector of shape (1, filter_size)
            used for the pyramid construction.
    """
    gauss_pyr, filter_vec = build_gaussian_pyramid(im, max_levels, filter_size)
    lap_pyr = [gauss_pyr[i] - expand(gauss_pyr[i + 1], filter_vec) for i in range(len(gauss_pyr) - 1)]
    lap_pyr.append(gauss_pyr[-1])
    return lap_pyr, filter_vec


def laplacian_to_image(lpyr, filter_vec, coeff):
    """
    :param lpyr: Laplacian pyramid
    :param filter_vec: Filter vector
    :param coeff: A python list in the same length as the number of levels in
            the pyramid lpyr.
    :return: Reconstructed image
    """

    def _get_gauss(n=0):
        return lpyr[n] * coeff[n] if n == len(lpyr) - 1 else \
            (lpyr[n] * coeff[n]) + expand(_get_gauss(n + 1), filter_vec)

    return _get_gauss()


def pyramid_blending(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):
    """
     Pyramid blending implementation
    :param im1: input grayscale image
    :param im2: input grayscale image
    :param mask: a boolean mask
    :param max_levels: max_levels for the pyramids
    :param filter_size_im: is the size of the Gaussian filter (an odd
            scalar that represents a squared filter)
    :param filter_size_mask: size of the Gaussian filter(an odd scalar
            that represents a squared filter) which defining the filter used
            in the construction of the Gaussian pyramid of mask
    :return: the blended image
    """
    l1, (l2, fil), gm = build_laplacian_pyramid(im1, max_levels, filter_size_im)[0], \
                        build_laplacian_pyramid(im2, max_levels, filter_size_im), \
                        build_gaussian_pyramid(mask.astype(np.float64), max_levels, filter_size_mask)[0]
    out_lap = [gm[i] * l1[i] + ((-1 * gm[i]) + 1) * l2[i] for i in range(len(l1))]
    return np.clip(laplacian_to_image(out_lap, fil, [1] * max_levels), 0, 1)
