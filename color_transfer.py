import argparse
import os
import numpy as np
from scipy.interpolate import interp1d
from scipy.misc import imread, imresize, imsave, fromimage, toimage


# Util function to match histograms
def match_histograms(A, B, rng=(0.0, 255.0), bins=64):
    (Ha, Xa), (Hb, Xb) = [np.histogram(i, bins=bins, range=rng, density=True) for i in [A, B]]
    X = np.linspace(rng[0], rng[1], bins, endpoint=True)
    Hpa, Hpb = [np.cumsum(i) * (rng[1] - rng[0]) ** 2 / float(bins) for i in [Ha, Hb]]
    inv_Ha = interp1d(X, Hpa, bounds_error=False, fill_value='extrapolate')
    map_Hb = interp1d(Hpb, X, bounds_error=False, fill_value='extrapolate')
    return map_Hb(inv_Ha(A).clip(0.0, 255.0))


# util function to preserve image color
def original_color_transform(content, generated, mask=None, hist_match=0, mode='YCbCr'):
    generated = fromimage(toimage(generated, mode='RGB'), mode=mode)  # Convert to YCbCr color space

    if mask is None:
        if hist_match == 1:
            for channel in range(3):
                generated[:, :, channel] = match_histograms(generated[:, :, channel], content[:, :, channel])
        else:
            generated[:, :, 1:] = content[:, :, 1:]
    else:
        width, height, channels = generated.shape

        for i in range(width):
            for j in range(height):
                if mask[i, j] == 1:
                    if hist_match == 1:
                        for channel in range(3):
                            generated[i, j, channel] = match_histograms(generated[i, j, channel], content[i, j, channel])
                    else:
                        generated[i, j, 1:] = content[i, j, 1:]

    generated = fromimage(toimage(generated, mode=mode), mode='RGB')  # Convert to RGB color space
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
parser.add_argument('--mask', default=None, type=str, help='Path to mask image')
parser.add_argument('--hist_match', type=int, default=0, help='Perform histogram matching for color matching')

args = parser.parse_args()

if args.hist_match == 1:
    image_suffix = "_histogram_color.png"
    mode = "RGB"
else:
    image_suffix = "_original_color.png"
    mode = "YCbCr"

image_path = os.path.splitext(args.generated_image)[0] + image_suffix

generated_image = imread(args.generated_image, mode="RGB")
img_width, img_height, _ = generated_image.shape

content_image = imread(args.content_image, mode=mode)
content_image = imresize(content_image, (img_width, img_height), interp='bicubic')

mask_transfer = args.mask is not None
if mask_transfer:
    mask_img = load_mask(args.mask, generated_image.shape)
else:
    mask_img = None

img = original_color_transform(content_image, generated_image, mask_img, args.hist_match, mode=mode)
imsave(image_path, img)

print("Image saved at path : %s" % image_path)


