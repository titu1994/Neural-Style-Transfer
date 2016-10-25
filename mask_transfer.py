import argparse
import os
import numpy as np
from scipy.misc import imread, imresize, imsave


# util function to load masks
def load_mask(mask_path, shape):
    mask = imread(mask_path, mode="L") # Grayscale mask load
    width, height, _ = shape
    mask = imresize(mask, (width, height), interp='bicubic').astype('float32')

    # Perform binarization of mask
    mask[mask <= 127] = 0
    mask[mask > 128] = 255

    max = np.amax(mask)
    mask /= max

    return mask


# util function to apply mask to generated image
def mask_content(content, generated, mask):
    width, height, channels = generated.shape

    for i in range(width):
        for j in range(height):
            if mask[i, j] == 0.:
                generated[i, j, :] = content[i, j, :]

    return generated

parser = argparse.ArgumentParser(description='Neural style transfer color preservation.')

parser.add_argument('content_image', type=str, help='Path to content image')
parser.add_argument('generated_image', type=str, help='Path to generated image')
parser.add_argument('content_mask', type=str, help='Path to content mask')

args = parser.parse_args()

image_path = os.path.splitext(args.generated_image)[0] + "_masked.png"

generated_image = imread(args.generated_image, mode="RGB")
img_width, img_height, channels = generated_image.shape

content_image = imread(args.content_image, mode='RGB')
content_image = imresize(content_image, (img_width, img_height), interp='bicubic')

mask = load_mask(args.content_mask, generated_image.shape)

img = mask_content(content_image, generated_image, mask)
imsave(image_path, img)

print("Image saved at path : %s" % image_path)
