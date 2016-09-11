from scipy.misc import imread, imresize, imsave, fromimage, toimage
from scipy.optimize import fmin_l_bfgs_b
import numpy as np
import time
import argparse

from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, ZeroPadding2D, AveragePooling2D, MaxPooling2D
from keras import backend as K
from keras.utils.data_utils import get_file

"""
Neural Style Transfer with Keras 1.0.8

Based on:
https://github.com/fchollet/keras/blob/master/examples/neural_style_transfer.py

-----------------------------------------------------------------------------------------------------------------------
"""

THEANO_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_th_dim_ordering_th_kernels_notop.h5'
TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

parser = argparse.ArgumentParser(description='Neural style transfer with Keras.')
parser.add_argument('base_image_path', metavar='base', type=str,
                    help='Path to the image to transform.')
parser.add_argument('style_reference_image_path', metavar='ref', type=str,
                    help='Path to the style reference image.')
parser.add_argument('result_prefix', metavar='res_prefix', type=str,
                    help='Prefix for the saved results.')

parser.add_argument("--image_size", dest="img_size", default=400, type=int, help='Output Image size')
parser.add_argument("--content_weight", dest="content_weight", default=0.025, type=float,
                    help="Weight of content")  # 0.025
parser.add_argument("--style_weight", dest="style_weight", default=1, type=float, help="Weight of content")  # 1.0
parser.add_argument("--style_scale", dest="style_scale", default=1.0, type=float,
                    help="Scale the weightage of the style")  # 1, 0.5, 2
parser.add_argument("--total_variation_weight", dest="tv_weight", default=8.5e-5, type=float,
                    help="Total Variation in the Weights")  # 1.0
parser.add_argument("--num_iter", dest="num_iter", default=10, type=int, help="Number of iterations")
parser.add_argument("--rescale_image", dest="rescale_image", default="True", type=str,
                    help="Rescale image after execution to original dimentions")
parser.add_argument("--rescale_method", dest="rescale_method", default="bilinear", type=str,
                    help="Rescale image algorithm")
parser.add_argument("--maintain_aspect_ratio", dest="maintain_aspect_ratio", default="True", type=str,
                    help="Maintain aspect ratio of image")
parser.add_argument("--content_layer", dest="content_layer", default="conv5_2", type=str, help="Optional 'conv4_2'")
parser.add_argument("--init_image", dest="init_image", default="content", type=str,
                    help="Initial image used to generate the final image. Options are 'content' or 'noise")
parser.add_argument("--pool_type", dest="pool", default="max", type=str,
                    help='Pooling type. Can be "ave" for average pooling'
                         ' or "max" for max pooling ')
parser.add_argument('--preserve_color', dest='color', default="False", type=str,
                    help='Preserve original color in image')
parser.add_argument('--min_improvement', default=0.0, type=float,
                    help='Defines minimum improvement required to continue script')

args = parser.parse_args()
base_image_path = args.base_image_path
style_reference_image_path = args.style_reference_image_path
result_prefix = args.result_prefix


def str_to_bool(v):
    return v.lower() in ("true", "yes", "t", "1")


rescale_image = str_to_bool(args.rescale_image)
maintain_aspect_ratio = str_to_bool(args.maintain_aspect_ratio)
preserve_color = str_to_bool(args.color)

# these are the weights of the different loss components
total_variation_weight = args.tv_weight
style_weight = args.style_weight * args.style_scale
content_weight = args.content_weight

# dimensions of the generated picture.
img_width = img_height = args.img_size
assert img_height == img_width, 'Due to the use of the Gram matrix, width and height must match.'

img_WIDTH = img_HEIGHT = 0
aspect_ratio = 0


# util function to open, resize and format pictures into appropriate tensors
def preprocess_image(image_path, load_dims=False):
    global img_WIDTH, img_HEIGHT, aspect_ratio

    img = imread(image_path, mode="RGB")  # Prevents crashes due to PNG images (ARGB)
    if load_dims:
        img_WIDTH = img.shape[0]
        img_HEIGHT = img.shape[1]
        aspect_ratio = img_HEIGHT / img_WIDTH

    img = imresize(img, (img_width, img_height)).astype('float32')
    img[:, :, 0] -= 103.939
    img[:, :, 1] -= 116.779
    img[:, :, 2] -= 123.68

    img = img[:, :, ::-1]

    if K.image_dim_ordering() == "th":
        img = img.transpose((2, 0, 1)).astype('float32')

    img = np.expand_dims(img, axis=0)
    return img


# util function to convert a tensor into a valid image
def deprocess_image(x):
    if K.image_dim_ordering() == "th":
        x = x.reshape((3, img_width, img_height))
        x = x.transpose((1, 2, 0))
    else:
        x = x.reshape((img_width, img_height, 3))

    x = x[:, :, ::-1]
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = np.clip(x, 0, 255).astype('uint8')
    return x


# util function to preserve image color
def original_color_transform(content, generated):
    generated = fromimage(toimage(generated), mode='YCbCr')  # Convert to YCbCr color space
    generated[:, :, 1:] = content[:, :, 1:]  # Generated CbCr = Content CbCr
    generated = fromimage(toimage(generated, mode='YCbCr'), mode='RGB')  # Convert to RGB color space
    return generated


# Decide pooling function
pooltype = str(args.pool).lower()
assert pooltype in ["ave", "max"], 'Pooling argument is wrong. Needs to be either "ave" or "max".'

pooltype = 1 if pooltype == "ave" else 0


def pooling_func():
    if pooltype == 1:
        return AveragePooling2D((2, 2), strides=(2, 2))
    else:
        return MaxPooling2D((2, 2), strides=(2, 2))


# get tensor representations of our images
base_image = K.variable(preprocess_image(base_image_path, True))
style_reference_image = K.variable(preprocess_image(style_reference_image_path))

# this will contain our generated image
if K.image_dim_ordering() == 'th':
    combination_image = K.placeholder((1, 3, img_width, img_height))
else:
    combination_image = K.placeholder((1, img_width, img_height, 3))

# combine the 3 images into a single Keras tensor
input_tensor = K.concatenate([base_image,
                              style_reference_image,
                              combination_image], axis=0)

if K.image_dim_ordering() == "th":
    shape = (3, 3, img_width, img_height)
else:
    shape = (3, img_width, img_height, 3)

# build the VGG16 network with our 3 images as input
first_layer = Convolution2D(64, 3, 3, activation='relu', name='conv1_1')
first_layer.set_input(input_tensor, shape)

model = Sequential()
model.add(first_layer)
model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2', border_mode='same'))
model.add(pooling_func())

model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1', border_mode='same'))
model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2', border_mode='same'))
model.add(pooling_func())

model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1', border_mode='same'))
model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2', border_mode='same'))
model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3', border_mode='same'))
model.add(pooling_func())

model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1', border_mode='same'))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2', border_mode='same'))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3', border_mode='same'))
model.add(pooling_func())

model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1', border_mode='same'))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2', border_mode='same'))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3', border_mode='same'))
model.add(pooling_func())

if K.image_dim_ordering() == "th":
    weights = get_file('vgg16_weights_th_dim_ordering_th_kernels_notop.h5', THEANO_WEIGHTS_PATH_NO_TOP, cache_subdir='models')
else:
    weights = get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', TF_WEIGHTS_PATH_NO_TOP, cache_subdir='models')

model.load_weights(weights)
print('Model loaded.')

# get the symbolic outputs of each "key" layer (we gave them unique names).
outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])


# compute the neural style loss
# first we need to define 4 util functions

# the gram matrix of an image tensor (feature-wise outer product)
def gram_matrix(x):
    assert K.ndim(x) == 3
    if K.image_dim_ordering() == "th":
        features = K.batch_flatten(x)
    else:
        features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram


# the "style loss" is designed to maintain
# the style of the reference image in the generated image.
# It is based on the gram matrices (which capture style) of
# feature maps from the style reference image
# and from the generated image
def style_loss(style, combination):
    assert K.ndim(style) == 3
    assert K.ndim(combination) == 3
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_width * img_height
    return K.sum(K.square(S - C)) / (4. * (channels ** 2) * (size ** 2))


# an auxiliary loss function
# designed to maintain the "content" of the
# base image in the generated image
def content_loss(base, combination):
    return K.sum(K.square(combination - base))


# the 3rd loss function, total variation loss,
# designed to keep the generated image locally coherent
def total_variation_loss(x):
    assert K.ndim(x) == 4
    if K.image_dim_ordering() == 'th':
        a = K.square(x[:, :, :img_width - 1, :img_height - 1] - x[:, :, 1:, :img_height - 1])
        b = K.square(x[:, :, :img_width - 1, :img_height - 1] - x[:, :, :img_width - 1, 1:])
    else:
        a = K.square(x[:, :img_width - 1, :img_height - 1, :] - x[:, 1:, :img_height - 1, :])
        b = K.square(x[:, :img_width - 1, :img_height - 1, :] - x[:, :img_width - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))


# combine these loss functions into a single scalar
loss = K.variable(0.)
layer_features = outputs_dict[args.content_layer]  # 'conv5_2' or 'conv4_2'
base_image_features = layer_features[0, :, :, :]
combination_features = layer_features[2, :, :, :]
loss += content_weight * content_loss(base_image_features,
                                      combination_features)

feature_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
for layer_name in feature_layers:
    layer_features = outputs_dict[layer_name]
    style_reference_features = layer_features[1, :, :, :]
    combination_features = layer_features[2, :, :, :]
    sl = style_loss(style_reference_features, combination_features)
    loss += (style_weight / len(feature_layers)) * sl
loss += total_variation_weight * total_variation_loss(combination_image)

# get the gradients of the generated image wrt the loss
grads = K.gradients(loss, combination_image)

outputs = [loss]
if type(grads) in {list, tuple}:
    outputs += grads
else:
    outputs.append(grads)

f_outputs = K.function([combination_image], outputs)


def eval_loss_and_grads(x):
    if K.image_dim_ordering() == 'th':
        x = x.reshape((1, 3, img_width, img_height))
    else:
        x = x.reshape((1, img_width, img_height, 3))
    outs = f_outputs([x])
    loss_value = outs[0]
    if len(outs[1:]) == 1:
        grad_values = outs[1].flatten().astype('float64')
    else:
        grad_values = np.array(outs[1:]).flatten().astype('float64')
    return loss_value, grad_values


# this Evaluator class makes it possible
# to compute loss and gradients in one pass
# while retrieving them via two separate functions,
# "loss" and "grads". This is done because scipy.optimize
# requires separate functions for loss and gradients,
# but computing them separately would be inefficient.
class Evaluator(object):
    def __init__(self):
        self.loss_value = None
        self.grads_values = None

    def loss(self, x):
        assert self.loss_value is None
        loss_value, grad_values = eval_loss_and_grads(x)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values


evaluator = Evaluator()

# run scipy-based optimization (L-BFGS) over the pixels of the generated image
# so as to minimize the neural style loss

assert args.init_image in ["content", "noise"], "init_image must be one of ['content', 'noise']"
if "content" in args.init_image:
    x = preprocess_image(base_image_path, True)
else:
    x = np.random.uniform(0, 255, (1, img_width, img_height, 3)) - 128.

    if K.image_dim_ordering() == "th":
        x = x.transpose((0, 3, 1, 2))

# We require original image if we are to preserve color in YCbCr mode
if preserve_color:
    content = imread(base_image_path, mode="YCbCr")
    content = imresize(content, (img_width, img_height))

num_iter = args.num_iter
prev_min_val = np.inf

improvement_threshold = float(args.min_improvement)

for i in range(num_iter):
    print('Start of iteration', (i + 1))
    start_time = time.time()

    x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(), fprime=evaluator.grads, maxfun=20)

    improvement = (prev_min_val - min_val) / prev_min_val * 100

    print('Current loss value:', min_val, " Improvement : %0.3f" % improvement, "%")
    prev_min_val = min_val
    # save current generated image
    img = deprocess_image(x.copy())

    if preserve_color and content is not None:
        img = original_color_transform(content, img)

    if maintain_aspect_ratio & (not rescale_image):
        img_ht = int(img_width * aspect_ratio)
        print("Rescaling Image to (%d, %d)" % (img_width, img_ht))
        img = imresize(img, (img_width, img_ht), interp=args.rescale_method)

    if rescale_image:
        print("Rescaling Image to (%d, %d)" % (img_WIDTH, img_HEIGHT))
        img = imresize(img, (img_WIDTH, img_HEIGHT), interp=args.rescale_method)

    fname = result_prefix + '_at_iteration_%d.png' % (i + 1)
    imsave(fname, img)
    end_time = time.time()
    print('Image saved as', fname)
    print('Iteration %d completed in %ds' % (i + 1, end_time - start_time))

    if improvement_threshold is not 0.0:
        if improvement < improvement_threshold and improvement is not np.nan:
            print("Improvement (%f) is less than improvement threshold (%f). Early stopping script." % (
                improvement, improvement_threshold))
            exit()
