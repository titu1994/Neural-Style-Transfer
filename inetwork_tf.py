from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

# from scipy.misc import imread, imresize, imsave, fromimage, toimage
from utils import imread, imresize, imsave, fromimage, toimage

from scipy.optimize import fmin_l_bfgs_b
import numpy as np
import time
import argparse
import warnings

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Convolution2D, AveragePooling2D, MaxPooling2D
from tensorflow.keras import backend as K
from tensorflow.keras.utils import get_file
# from tensorflow.keras.utils import convert_all_kernels_in_model

from tf_bfgs import LBFGSOptimizer

"""
Neural Style Transfer with Keras 2.0.5

Based on:
https://github.com/keras-team/keras-io/blob/master/examples/generative/neural_style_transfer.py

Contains few improvements suggested in the paper Improving the Neural Algorithm of Artistic Style
(http://arxiv.org/abs/1605.04603).

-----------------------------------------------------------------------------------------------------------------------
"""

THEANO_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_th_dim_ordering_th_kernels_notop.h5'
TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

TH_19_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_th_dim_ordering_th_kernels_notop.h5'
TF_19_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'


parser = argparse.ArgumentParser(description='Neural style transfer with Keras.')
parser.add_argument('base_image_path', metavar='base', type=str,
                    help='Path to the image to transform.')

parser.add_argument('syle_image_paths', metavar='ref', nargs='+', type=str,
                    help='Path to the style reference image.')

parser.add_argument('result_prefix', metavar='res_prefix', type=str,
                    help='Prefix for the saved results.')

parser.add_argument("--style_masks", type=str, default=None, nargs='+',
                    help='Masks for style images')

parser.add_argument("--content_mask", type=str, default=None,
                    help='Masks for the content image')

parser.add_argument("--color_mask", type=str, default=None,
                    help='Mask for color preservation')

parser.add_argument("--image_size", dest="img_size", default=400, type=int,
                    help='Minimum image size')

parser.add_argument("--content_weight", dest="content_weight", default=0.025, type=float,
                    help="Weight of content")

parser.add_argument("--style_weight", dest="style_weight", nargs='+', default=[1], type=float,
                    help="Weight of style, can be multiple for multiple styles")

parser.add_argument("--style_scale", dest="style_scale", default=1.0, type=float,
                    help="Scale the weighing of the style")

parser.add_argument("--total_variation_weight", dest="tv_weight", default=8.5e-5, type=float,
                    help="Total Variation weight")

parser.add_argument("--num_iter", dest="num_iter", default=10, type=int,
                    help="Number of iterations")

parser.add_argument("--model", default="vgg16", type=str,
                    help="Choices are 'vgg16' and 'vgg19'")

parser.add_argument("--content_loss_type", default=0, type=int,
                    help='Can be one of 0, 1 or 2. Readme contains the required information of each mode.')

parser.add_argument("--rescale_image", dest="rescale_image", default="False", type=str,
                    help="Rescale image after execution to original dimentions")

parser.add_argument("--rescale_method", dest="rescale_method", default="bilinear", type=str,
                    help="Rescale image algorithm")

parser.add_argument("--maintain_aspect_ratio", dest="maintain_aspect_ratio", default="True", type=str,
                    help="Maintain aspect ratio of loaded images")

parser.add_argument("--content_layer", dest="content_layer", default="conv5_2", type=str,
                    help="Content layer used for content loss.")

parser.add_argument("--init_image", dest="init_image", default="content", type=str,
                    help="Initial image used to generate the final image. Options are 'content', 'noise', or 'gray'")

parser.add_argument("--pool_type", dest="pool", default="max", type=str,
                    help='Pooling type. Can be "ave" for average pooling or "max" for max pooling')

parser.add_argument('--preserve_color', dest='color', default="False", type=str,
                    help='Preserve original color in image')

parser.add_argument('--min_improvement', default=0.0, type=float,
                    help='Defines minimum improvement required to continue script')

parser.add_argument('--steps_per_epoch', dest='steps_per_epoch', type=int, default=100,
                    help='Number of adam updates per global step')

def str_to_bool(v):
    return v.lower() in ("true", "yes", "t", "1")

''' Arguments '''

args = parser.parse_args()
base_image_path = args.base_image_path
style_reference_image_paths = args.syle_image_paths
result_prefix = args.result_prefix

style_image_paths = []
for style_image_path in style_reference_image_paths:
    style_image_paths.append(style_image_path)

style_masks_present = args.style_masks is not None
mask_paths = []

if style_masks_present:
    for mask_path in args.style_masks:
        mask_paths.append(mask_path)

if style_masks_present:
    assert len(style_image_paths) == len(mask_paths), "Wrong number of style masks provided.\n" \
                                                      "Number of style images = %d, \n" \
                                                      "Number of style mask paths = %d." % \
                                                      (len(style_image_paths), len(style_masks_present))

content_mask_present = args.content_mask is not None
content_mask_path = args.content_mask


color_mask_present = args.color_mask is not None

rescale_image = str_to_bool(args.rescale_image)
maintain_aspect_ratio = str_to_bool(args.maintain_aspect_ratio)
preserve_color = str_to_bool(args.color)

# these are the weights of the different loss components
content_weight = args.content_weight
total_variation_weight = args.tv_weight

style_weights = []

if len(style_image_paths) != len(args.style_weight):
    print("Mismatch in number of style images provided and number of style weights provided. \n"
          "Found %d style images and %d style weights. \n"
          "Equally distributing weights to all other styles." % (len(style_image_paths), len(args.style_weight)))

    weight_sum = sum(args.style_weight) * args.style_scale
    count = len(style_image_paths)

    for i in range(len(style_image_paths)):
        style_weights.append(weight_sum / count)
else:
    for style_weight in args.style_weight:
        style_weights.append(style_weight * args.style_scale)

# Decide pooling function
pooltype = str(args.pool).lower()
assert pooltype in ["ave", "max"], 'Pooling argument is wrong. Needs to be either "ave" or "max".'

pooltype = 1 if pooltype == "ave" else 0

read_mode = "gray" if args.init_image == "gray" else "color"

# dimensions of the generated picture.
img_width = img_height = 0

img_WIDTH = img_HEIGHT = 0
aspect_ratio = 0

assert args.content_loss_type in [0, 1, 2], "Content Loss Type must be one of 0, 1 or 2"


# util function to open, resize and format pictures into appropriate tensors
def preprocess_image(image_path, load_dims=False, read_mode="color"):
    global img_width, img_height, img_WIDTH, img_HEIGHT, aspect_ratio

    mode = "RGB" if read_mode == "color" else "L"
    channels = 3 if mode == 'RGB' else 1
    img = tf.io.read_file(image_path)  # Prevents crashes due to PNG images (ARGB)
    img = tf.image.decode_image(img, channels)
    img = tf.image.convert_image_dtype(img, tf.float32)

    img *= 255.

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)

    if mode == "L":
        # Expand the 1 channel grayscale to 3 channel grayscale image
        temp = np.zeros(img.shape + (3,), dtype=np.uint8)
        temp[:, :, 0] = img
        temp[:, :, 1] = img.copy()
        temp[:, :, 2] = img.copy()

        img = temp

    if load_dims:
        img_WIDTH = img.shape[0]
        img_HEIGHT = img.shape[1]
        aspect_ratio = float(img_HEIGHT) / img_WIDTH

        img_width = args.img_size

        if maintain_aspect_ratio:
            img_height = int(img_width * aspect_ratio)
        else:
            img_height = args.img_size

    new_shape = tf.cast(tf.convert_to_tensor([img_width, img_height]), tf.int32)
    img = tf.image.resize(img, new_shape)

    img = img.numpy()

    # RGB -> BGR
    img = img[:, :, ::-1]

    img[:, :, 0] -= 103.939
    img[:, :, 1] -= 116.779
    img[:, :, 2] -= 123.68

    img = tf.convert_to_tensor(img, dtype=tf.float32)

    if K.image_data_format() == "channels_first":
        img = tf.cast(tf.transpose(img, (2, 0, 1)), tf.float32)

    img = tf.expand_dims(img, axis=0)
    return img


# util function to convert a tensor into a valid image
def deprocess_image(x):
    if K.image_data_format() == "channels_first":
        x = x.reshape((3, img_width, img_height))
        x = x.transpose((1, 2, 0))
    else:
        x = x.reshape((img_width, img_height, 3))

    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68

    # BGR -> RGB
    x = x[:, :, ::-1]

    x = np.clip(x, 0, 255).astype('uint8')
    return x


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


def load_mask(mask_path, shape, return_mask_img=False):
    if K.image_data_format() == "channels_first":
        _, channels, width, height = shape
    else:
        _, width, height, channels = shape

    mask = imread(mask_path, mode="L") # Grayscale mask load
    mask = imresize(mask, (width, height)).astype('float32')

    # Perform binarization of mask
    mask[mask <= 127] = 0
    mask[mask > 128] = 255

    max = np.amax(mask)
    mask /= max

    if return_mask_img: return mask

    mask_shape = shape[1:]

    mask_tensor = np.empty(mask_shape)

    for i in range(channels):
        if K.image_data_format() == "channels_first":
            mask_tensor[i, :, :] = mask
        else:
            mask_tensor[:, :, i] = mask

    return mask_tensor


def pooling_func(x):
    if pooltype == 1:
        return AveragePooling2D((2, 2), strides=(2, 2))(x)
    else:
        return MaxPooling2D((2, 2), strides=(2, 2))(x)


# get tensor representations of our images
base_image = K.variable(preprocess_image(base_image_path, True, read_mode=read_mode))

style_reference_images = []
for style_path in style_image_paths:
    style_reference_images.append(K.variable(preprocess_image(style_path)))

# this will contain our generated image
# if K.image_data_format() == "channels_first":
#     combination_image = K.placeholder((1, 3, img_width, img_height))
# else:
#     combination_image = K.placeholder((1, img_width, img_height, 3))
#
# image_tensors = [base_image]
# for style_image_tensor in style_reference_images:
#     image_tensors.append(style_image_tensor)
# image_tensors.append(combination_image)

# nb_tensors = len(image_tensors)
# nb_style_images = nb_tensors - 2 # Content and Output image not considered

# combine the various images into a single Keras tensor
# input_tensor = K.concatenate(image_tensors, axis=0)

nb_tensors = 1
if K.image_data_format() == "channels_first":
    shape = (nb_tensors, 3, img_width, img_height)
else:
    shape = (nb_tensors, img_width, img_height, 3)

ip = Input(batch_shape=shape)

# build the VGG16 network with our 3 images as input
x = Convolution2D(64, (3, 3), activation='relu', name='conv1_1', padding='same')(ip)
x = Convolution2D(64, (3, 3), activation='relu', name='conv1_2', padding='same')(x)
x = pooling_func(x)

x = Convolution2D(128, (3, 3), activation='relu', name='conv2_1', padding='same')(x)
x = Convolution2D(128, (3, 3), activation='relu', name='conv2_2', padding='same')(x)
x = pooling_func(x)

x = Convolution2D(256, (3, 3), activation='relu', name='conv3_1', padding='same')(x)
x = Convolution2D(256, (3, 3), activation='relu', name='conv3_2', padding='same')(x)
x = Convolution2D(256, (3, 3), activation='relu', name='conv3_3', padding='same')(x)
if args.model == "vgg19":
    x = Convolution2D(256, (3, 3), activation='relu', name='conv3_4', padding='same')(x)
x = pooling_func(x)

x = Convolution2D(512, (3, 3), activation='relu', name='conv4_1', padding='same')(x)
x = Convolution2D(512, (3, 3), activation='relu', name='conv4_2', padding='same')(x)
x = Convolution2D(512, (3, 3), activation='relu', name='conv4_3', padding='same')(x)
if args.model == "vgg19":
    x = Convolution2D(512, (3, 3), activation='relu', name='conv4_4', padding='same')(x)
x = pooling_func(x)

x = Convolution2D(512, (3, 3), activation='relu', name='conv5_1', padding='same')(x)
x = Convolution2D(512, (3, 3), activation='relu', name='conv5_2', padding='same')(x)
x = Convolution2D(512, (3, 3), activation='relu', name='conv5_3', padding='same')(x)
if args.model == "vgg19":
    x = Convolution2D(512, (3, 3), activation='relu', name='conv5_4', padding='same')(x)
x = pooling_func(x)

model = Model(ip, x)

if K.image_data_format() == "channels_first":
    if args.model == "vgg19":
        weights = get_file('vgg19_weights_th_dim_ordering_th_kernels_notop.h5', TH_19_WEIGHTS_PATH_NO_TOP, cache_subdir='models')
    else:
        weights = get_file('vgg16_weights_th_dim_ordering_th_kernels_notop.h5', THEANO_WEIGHTS_PATH_NO_TOP, cache_subdir='models')
else:
    if args.model == "vgg19":
        weights = get_file('vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5', TF_19_WEIGHTS_PATH_NO_TOP, cache_subdir='models')
    else:
        weights = get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', TF_WEIGHTS_PATH_NO_TOP, cache_subdir='models')

model.load_weights(weights)

if K.backend() == 'tensorflow' and K.image_data_format() == "channels_first":
    warnings.warn('You are using the TensorFlow backend, yet you '
                  'are using the Theano '
                  'image dimension ordering convention '
                  '(`image_dim_ordering="th"`). '
                  'For best performance, set '
                  '`image_dim_ordering="tf"` in '
                  'your Keras config '
                  'at ~/.keras/keras.json.')
    # convert_all_kernels_in_model(model)

print('Model loaded.')

# compute the neural style loss
# first we need to define 4 util functions

# def clip_0_1(image):
#   return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

# Improvement 1
# the gram matrix of an image tensor (feature-wise outer product) using shifted activations
def gram_matrix(x):
    result = tf.linalg.einsum('bijc,bijd->bcd', x, x)
    # input_shape = tf.shape(x)
    # num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    # gram = result / (num_locations)
    gram = result
    return gram


class StyleContentModel(tf.keras.Model):

    def __init__(self, model: tf.keras.Model, style_layers, content_layers, style_mask_path=None, content_mask_path=None):
        super(StyleContentModel, self).__init__()

        # get the symbolic outputs of each "key" layer (we gave them unique names).
        outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])
        self.shape_dict = dict([(layer.name, layer.output_shape) for layer in model.layers])

        style_activations = [outputs_dict[layer_name] for layer_name in style_layers]
        content_activations = [outputs_dict[layer_name] for layer_name in content_layers]

        activations = style_activations + content_activations

        self.vgg = tf.keras.Model(model.input, activations)

        self.style_layer_names = style_layers
        self.content_layer_names = content_layers

        self.num_style_layers = len(style_layers)
        self.num_content_layers = len(content_layers)

    def call(self, inputs, training=None, mask=None):
        outputs = self.vgg(inputs)
        style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                          outputs[self.num_style_layers:])

        style_outputs = [gram_matrix(style_output)
                         for style_output in style_outputs]

        content_dict = {content_name: value
                        for content_name, value
                        in zip(self.content_layer_names, content_outputs)}

        style_dict = {style_name: value
                      for style_name, value
                      in zip(self.style_layer_names, style_outputs)}

        return {'content': content_dict, 'style': style_dict}


# the "style loss" is designed to maintain
# the style of the reference image in the generated image.
# It is based on the gram matrices (which capture style) of
# feature maps from the style reference image
# and from the generated image
def style_loss(style, combination, mask_path=None, nb_channels=None):
    assert K.ndim(style) == 3
    assert K.ndim(combination) == 3

    if content_mask_path is not None:
        content_mask = K.variable(load_mask(content_mask_path, nb_channels))
        combination = combination * K.stop_gradient(content_mask)
        del content_mask

    if mask_path is not None:
        style_mask = K.variable(load_mask(mask_path, nb_channels))
        style = style * K.stop_gradient(style_mask)
        if content_mask_path is None:
            combination = combination * K.stop_gradient(style_mask)
        del style_mask

    channels = 3
    size = img_width * img_height
    return K.sum(K.square(style - combination)) / (4. * (channels ** 2) * (size ** 2))


# an auxiliary loss function
# designed to maintain the "content" of the
# base image in the generated image
def content_loss(base, combination):
    channel_dim = 0 if K.image_data_format() == "channels_first" else -1

    try:
        channels = K.int_shape(base)[channel_dim]
    except TypeError:
        channels = K.shape(base)[channel_dim]
    size = img_width * img_height

    if args.content_loss_type == 1:
        multiplier = 1. / (2. * (channels ** 0.5) * (size ** 0.5))
    elif args.content_loss_type == 2:
        multiplier = 1. / (channels * size)
    else:
        multiplier = 1.

    return multiplier * K.sum(K.square(combination - base))


# the 3rd loss function, total variation loss,
# designed to keep the generated image locally coherent
def total_variation_loss(x):
    assert K.ndim(x) == 4
    if K.image_data_format() == "channels_first":
        a = K.square(x[:, :, :img_width - 1, :img_height - 1] - x[:, :, 1:, :img_height - 1])
        b = K.square(x[:, :, :img_width - 1, :img_height - 1] - x[:, :, :img_width - 1, 1:])
    else:
        a = K.square(x[:, :img_width - 1, :img_height - 1, :] - x[:, 1:, :img_height - 1, :])
        b = K.square(x[:, :img_width - 1, :img_height - 1, :] - x[:, :img_width - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))

if args.model == "vgg19":
    feature_layers = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv3_3', 'conv3_4',
                      'conv4_1', 'conv4_2', 'conv4_3', 'conv4_4', 'conv5_1', 'conv5_2', 'conv5_3', 'conv5_4']
else:
    feature_layers = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv3_3',
                      'conv4_1', 'conv4_2', 'conv4_3', 'conv5_1', 'conv5_2', 'conv5_3']

extractor = StyleContentModel(model, style_layers=feature_layers, content_layers=[args.content_layer],
                              style_mask_path=None, content_mask_path=None)

content_targets = extractor(base_image)['content']

style_targets_list = []
for reference_style in style_reference_images:
    style_targets = extractor(reference_style)['style']
    style_targets_list.append(style_targets)

learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1.0,
                                                               decay_steps=100,
                                                               decay_rate=0.96)
optimizer = tf.optimizers.Adam(learning_rate=learning_rate, beta_1=0.99, epsilon=1e-1)
nb_layers = len(feature_layers) - 1


def compute_loss(input, outputs, shape_dict):
    style_combined_outputs = outputs['style']
    content_combined_outputs = outputs['content']

    # Content losses
    content_losses = content_weight * tf.add_n([content_loss(content_targets[name], content_combined_outputs[name])
                                                            for name in content_combined_outputs.keys()])

    num_style_references = len(style_reference_images)
    num_style_layers = len(feature_layers)

    if style_masks_present:
        style_masks = mask_paths  # If mask present, pass dictionary of masks to style loss
    else:
        style_masks = [None for _ in range(num_style_references)]  # If masks not present, pass None to the style loss

    # Style losses
    style_losses = []
    for style_img_id in range(num_style_references):
        style_features = style_targets_list[style_img_id]
        style_mask = style_masks[style_img_id]

        sl_i = 0.
        for feature_layer_id in range(num_style_layers - 1):
            target_feature_layer = style_features[feature_layers[feature_layer_id]]
            style_output = style_combined_outputs[feature_layers[feature_layer_id]]
            shape = shape_dict[feature_layers[feature_layer_id]]

            sl1 = style_loss(target_feature_layer, style_output, style_mask, shape)

            target_feature_layer = style_features[feature_layers[feature_layer_id + 1]]
            style_output = style_combined_outputs[feature_layers[feature_layer_id + 1]]
            shape = shape_dict[feature_layers[feature_layer_id + 1]]

            sl2 = style_loss(target_feature_layer, style_output, style_mask, shape)

            sl_i = sl_i + (sl1 - sl2) * (style_weights[style_img_id] / (2 ** (nb_layers - (feature_layer_id + 1))))

        style_losses.append(sl_i)

    style_losses = tf.add_n(style_losses)

    # Total Variation Losses
    tv_losses = total_variation_weight * total_variation_loss(input)

    return content_losses, style_losses, tv_losses


@tf.function
def train_step(image):
    with tf.GradientTape() as tape:
        outputs = extractor(image)
        content_losses, style_losses, tv_losses = compute_loss(image, outputs, shape_dict=extractor.shape_dict)
        loss = content_losses + style_losses + tv_losses

    grad = tape.gradient(loss, image)
    optimizer.apply_gradients([(grad, image)])
    # tf.print("C Loss ", content_losses, "| S Loss : ", style_losses, "| TV Loss : ", tv_losses)
    return loss


if "content" in args.init_image or "gray" in args.init_image:
    x = preprocess_image(base_image_path, True, read_mode=read_mode)
elif "noise" in args.init_image:
    x = np.random.uniform(0, 255, (1, img_width, img_height, 3)) - 128.
    x = x.astype('float32')

    if K.image_data_format() == "channels_first":
        x = x.transpose((0, 3, 1, 2))
else:
    print("Using initial image : ", args.init_image)
    x = preprocess_image(args.init_image, read_mode=read_mode)

# Make input trainable
x = tf.Variable(x)

# We require original image if we are to preserve color in YCbCr mode
if preserve_color:
    content = imread(base_image_path, mode="YCbCr")
    content = imresize(content, (img_width, img_height))

    if color_mask_present:
        if K.image_data_format() == "channels_first":
            color_mask_shape = (None, None, img_width, img_height)
        else:
            color_mask_shape = (None, img_width, img_height, None)

        color_mask = load_mask(args.color_mask, color_mask_shape, return_mask_img=True)
    else:
        color_mask = None
else:
    color_mask = None

num_iter = args.num_iter
steps_per_epoch = args.steps_per_epoch
prev_min_val = -1

improvement_threshold = float(args.min_improvement)

step_count = 0
total_steps = num_iter * steps_per_epoch

for i in range(num_iter):
    print("Starting iteration %d of %d (step %d / %d)" % ((i + 1), num_iter, step_count, total_steps))
    start_time = time.time()

    for step in range(steps_per_epoch):
        step_count += 1
        loss_val = train_step(x)

        if (step_count) % 500 == 0:
            print(f"Loss at step {step_count}: {loss_val.numpy():0.6f}")

    # x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(), fprime=evaluator.grads, maxfun=20)

    loss_val = loss_val.numpy()

    if prev_min_val == -1:
        prev_min_val = loss_val

    improvement = (prev_min_val - loss_val) / prev_min_val * 100

    print("Current loss value:", loss_val, " Improvement : %0.3f" % improvement, "%")
    prev_min_val = loss_val
    # save current generated image
    img = deprocess_image(x.numpy())

    if preserve_color and content is not None:
        img = original_color_transform(content, img, mask=color_mask)

    if not rescale_image:
        img_ht = int(img_width * aspect_ratio)
        print("Rescaling Image to (%d, %d)" % (img_width, img_ht))
        img = imresize(img, (img_width, img_ht), interp=args.rescale_method)

    if rescale_image:
        print("Rescaling Image to (%d, %d)" % (img_WIDTH, img_HEIGHT))
        img = imresize(img, (img_WIDTH, img_HEIGHT), interp=args.rescale_method)

    fname = result_prefix + "_at_iteration_%d.png" % (i + 1)
    imsave(fname, img)
    end_time = time.time()
    print("Image saved as", fname)
    print("Iteration %d completed in %ds" % (i + 1, end_time - start_time))

    if improvement_threshold != 0.0:
        if improvement < improvement_threshold and improvement != 0.0:
            print("Improvement (%f) is less than improvement threshold (%f). Early stopping script." %
                  (improvement, improvement_threshold))
            exit()

"""
python inetwork_tf.py "C:\\Users\\somsh\\OneDrive\\Pictures\\Album Art\\Ryogi-Shiki-small2.jpg" "C:\\Users\\somsh\\OneDrive\\Pictures\\
Art\\Blue Butterfly.jpg" "C:\\Users\\somsh\\Desktop\\Neural Art\\Ryogi" --image_size 400 --content_weight 0.025 --style_weight 1.0 --total_variation_weight 8.5E-05 --style_scale 1 --num_iter
 10 --rescale_image "False" --rescale_method "bicubic" --maintain_aspect_ratio "True" --content_layer "conv5_2" --init_image "content" --pool_type "max" --preserve_color "False" --min_
improvement 0 --model "vgg16" --content_loss_type 0
"""