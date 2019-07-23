
import numpy as np
import imageio
from PIL import Image


def imread(path, mode="RGB"):
    img = np.array(imageio.imread(path, pilmode=mode))
    return img
    

def imresize(img, size):
    img = np.array(Image.fromarray(img).resize(size))
    return img
    
    
def imsave(path, img):
    imageio.imwrite(path, img)
    return
    

def fromimage(arr):
    return np.array(arr)


def toimage(arr):
    return Image.fromarray(arr)
