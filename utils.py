
import numpy as np
import imageio
from PIL import Image
from skimage import color


def imread(path, mode="RGB"):
    # Loads data in HxW format, then transposes to correct format
    img = np.array(imageio.imread(path, pilmode=mode))
    return img
    

def imresize(img, size, interp='bilinear'):
    """
    Resizes an image

    :param img:
    :param size: (Must be H, W format !)
    :param interp:
    :return:
    """
    if interp == 'bilinear':
        interpolation = Image.BILINEAR
    elif interp == 'bicubic':
        interpolation = Image.BICUBIC
    else:
        interpolation = Image.NEAREST

    # Requires size to be HxW
    size = (size[1], size[0])

    if type(img) != Image:
        img = Image.fromarray(img, mode='RGB')

    img = np.array(img.resize(size, interpolation))
    return img
    
    
def imsave(path, img):
    imageio.imwrite(path, img)
    return
    

def fromimage(img, mode='RGB'):
    if mode == 'RGB':
        img = color.lab2rgb(img)
    else:
        img = color.rgb2lab(img)
    return img


def toimage(arr, mode='RGB'):
    return Image.fromarray(arr, mode)
