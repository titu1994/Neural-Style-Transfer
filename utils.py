
import numpy as np
import imageio
from PIL import Image


def imread(path, mode="RGB"):
    img = np.array(imageio.imread(path, pilmode=mode))
    return img
    

def imresize(img, size, interp='bilinear'):
    if interp == 'bilinear':
        interpolation = Image.BILINEAR
    elif interp == 'bicubic':
        interpolation = Image.BICUBIC
    else:
        interpolation = Image.NEAREST
        
    img = np.array(Image.fromarray(img, interpolation).resize(size))
    return img
    
    
def imsave(path, img):
    imageio.imwrite(path, img)
    return
    

def fromimage(arr):
    return np.array(arr)


def toimage(arr):
    return Image.fromarray(arr)
