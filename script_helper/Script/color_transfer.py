import argparse
import os
import numpy as np
from scipy.misc import imread, imresize, imsave, fromimage, toimage

# util function to preserve image color
def original_color_transform(content, generated, mask=None):
    generated = fromimage(toimage(generated, mode='RGB'), mode='YCbCr')  # Convert to YCbCr color space

    if mask is None:
        generated[:, :, 1:] = content[:, :, 1:]  # Generated CbCr = Content CbCr
    else:
        width, height, channels = generated.shape

        for i in range(width):
            for j in range(height):
                if mask[i, j] == 1:
                    generated[i, j, 1:] = content[i, j, 1:]

    generated = fromimage(toimage(generated, mode='YCbCr'), mode='RGB')  # Convert to RGB color space
    return generated


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


parser = argparse.ArgumentParser(description='Neural style transfer color preservation.')

parser.add_argument('content_image', type=str, help='Path to content image')
parser.add_argument('generated_image', type=str, help='Path to generated image')
parser.add_argument("--mask", default=None, type=str, help='Path to mask image')

args = parser.parse_args()

image_path = os.path.splitext(args.generated_image)[0] + "_original_color.png"

generated_image = imread(args.generated_image, mode="RGB")
img_width, img_height, _ = generated_image.shape

content_image = imread(args.content_image, mode='YCbCr')
content_image = imresize(content_image, (img_width, img_height), interp='bicubic')

mask_transfer = args.mask is not None
if mask_transfer:
    mask_img = load_mask(args.mask, generated_image.shape)
else:
    mask_img = None

img = original_color_transform(content_image, generated_image, mask_img)
imsave(image_path, img)

print("Image saved at path : %s" % image_path)


