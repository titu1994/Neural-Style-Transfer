import argparse
import os
from scipy.misc import imread, imresize, imsave, fromimage, toimage

# util function to preserve image color
def original_color_transform(content, generated):
    generated = fromimage(toimage(generated, mode='RGB'), mode='YCbCr')  # Convert to YCbCr color space
    generated[:, :, 1:] = content[:, :, 1:]  # Generated CbCr = Content CbCr
    generated = fromimage(toimage(generated, mode='YCbCr'), mode='RGB')  # Convert to RGB color space
    return generated

parser = argparse.ArgumentParser(description='Neural style transfer color preservation.')

parser.add_argument('content_image', type=str, help='Path to content image')
parser.add_argument('generated_image', type=str, help='Path to generated image')

args = parser.parse_args()

image_path = os.path.splitext(args.generated_image)[0] + "_original_color.png"

generated_image = imread(args.generated_image, mode="RGB")
img_width, img_height, _ = generated_image.shape

content_image = imread(args.content_image, mode='YCbCr')
content_image = imresize(content_image, (img_width, img_height), interp='bicubic')

img = original_color_transform(content_image, generated_image)
imsave(image_path, img)

print("Image saved at path : %s" % image_path)


