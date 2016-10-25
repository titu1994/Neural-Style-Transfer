import time
import argparse
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import imread, imsave, imresize, fromimage, toimage, imfilter

from keras import backend as K
from keras.layers import Input, AveragePooling2D
from keras.models import Model
from keras.preprocessing.image import load_img, img_to_array
from keras.applications import vgg16

"""
Neural Doodle in Keras using Keras 1.0.8+

Based on the original script available at : https://github.com/fchollet/keras/blob/master/examples/neural_doodle.py

References:
[Dmitry Ulyanov's blog on fast-neural-doodle](http://dmitryulyanov.github.io/feed-forward-neural-doodle/)
[Torch code for fast-neural-doodle](https://github.com/DmitryUlyanov/fast-neural-doodle)
[Torch code for online-neural-doodle](https://github.com/DmitryUlyanov/online-neural-doodle)
[Paper Texture Networks: Feed-forward Synthesis of Textures and Stylized Images](http://arxiv.org/abs/1603.03417)
[Discussion on parameter tuning](https://github.com/fchollet/keras/issues/3705)

"""

# Command line arguments
parser = argparse.ArgumentParser(description='Keras neural doodle example')
parser.add_argument('--nlabels', type=int,help='number of semantic labels (regions in differnet colors)'
                         ' in style_mask/target_mask')
parser.add_argument('--style-image', type=str, help='path to image to learn style from')
parser.add_argument('--style-mask', type=str, help='path to semantic mask of style image')
parser.add_argument('--target-mask', type=str, help='path to semantic mask of target image')
parser.add_argument('--content-image', type=str, default=None, help='path to optional content image')
parser.add_argument('--target-image-prefix', type=str, help='path prefix for generated results')

parser.add_argument("--img_size", type=int, default=-1, help='Image size will be rescaled to these dimensions. '
                                                              'Use -1 for no rescaling of input images')
parser.add_argument("--num_iter", dest="num_iter", default=10, type=int, help="Number of iterations")
parser.add_argument('--preserve_color', dest='color', default="False", type=str,
                    help='Preserve original color in image')
parser.add_argument("--min_improvement", default=0.0, type=float,
                    help="Minimum improvement required to continue training")

parser.add_argument("--content_weight", dest="content_weight", default=0.1, type=float, help="Weight of content")
parser.add_argument("--style_weight", dest="style_weight", default=1, type=float, help="Weight of content")
parser.add_argument("--tv_weight", dest="tv_weight", default=8.5e-5, type=float,
                    help="Total Variation in the Weights")
parser.add_argument("--region_style_weight", dest="region_weight", default=1.0, type=float, help="Region Style Weight")


args = parser.parse_args()

def str_to_bool(v):
    return v.lower() in ("true", "yes", "t", "1")

style_img_path = args.style_image
style_mask_path = args.style_mask
target_mask_path = args.target_mask
content_img_path = args.content_image
target_img_prefix = args.target_image_prefix
use_content_img = content_img_path is not None

nb_labels = args.nlabels
nb_colors = 3  # RGB
# determine image sizes based on target_mask
ref_img = imread(target_mask_path)

if args.img_size != -1:
    aspect_ratio = float(ref_img.shape[1]) / float(ref_img.shape[0])
    ref_img = imresize(ref_img, (int(args.img_size), int(args.img_size * aspect_ratio)))

img_nrows, img_ncols = ref_img.shape[:2]

total_variation_weight = float(args.tv_weight)
style_weight = float(args.style_weight)
content_weight =float(args.content_weight) if use_content_img else 0
region_style_weight = float(args.region_weight)

content_feature_layers = ['block5_conv2']
# To get better generation qualities, use more conv layers for style features
style_feature_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                        'block4_conv1', 'block5_conv1']

preserve_color = str_to_bool(args.color)

# helper functions for reading/processing images
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(img_nrows, img_ncols))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg16.preprocess_input(img)
    return img


def deprocess_image(x):
    if K.image_dim_ordering() == 'th':
        x = x.reshape((3, img_nrows, img_ncols))
        x = x.transpose((1, 2, 0))
    else:
        x = x.reshape((img_nrows, img_ncols, 3))
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68

    # BGR to RGB
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x


# util function to preserve image color
def original_color_transform(content, generated):
    generated = fromimage(toimage(generated, mode='RGB'), mode='YCbCr')  # Convert to YCbCr color space
    generated[:, :, 1:] = content[:, :, 1:]  # Generated CbCr = Content CbCr
    generated = fromimage(toimage(generated, mode='YCbCr'), mode='RGB')  # Convert to RGB color space
    return generated


def kmeans(xs, k):
    assert xs.ndim == 2
    try:
        from sklearn.cluster import k_means
        _, labels, _ = k_means(xs.astype("float64"), k)
    except ImportError:
        from scipy.cluster.vq import kmeans2
        _, labels = kmeans2(xs, k, missing='raise')
    return labels


def load_mask_labels():
    '''Load both target and style masks.
    A mask image (nr x nc) with m labels/colors will be loaded
    as a 4D boolean tensor: (1, m, nr, nc) for 'th' or (1, nr, nc, m) for 'tf'
    '''
    target_mask_img = load_img(target_mask_path,
                               target_size=(img_nrows, img_ncols))
    target_mask_img = img_to_array(target_mask_img)
    style_mask_img = load_img(style_mask_path,
                              target_size=(img_nrows, img_ncols))
    style_mask_img = img_to_array(style_mask_img)
    if K.image_dim_ordering() == 'th':
        mask_vecs = np.vstack([style_mask_img.reshape((3, -1)).T,
                               target_mask_img.reshape((3, -1)).T])
    else:
        mask_vecs = np.vstack([style_mask_img.reshape((-1, 3)),
                               target_mask_img.reshape((-1, 3))])

    labels = kmeans(mask_vecs, nb_labels)
    style_mask_label = labels[:img_nrows *
                               img_ncols].reshape((img_nrows, img_ncols))
    target_mask_label = labels[img_nrows *
                               img_ncols:].reshape((img_nrows, img_ncols))

    stack_axis = 0 if K.image_dim_ordering() == 'th' else -1
    style_mask = np.stack([style_mask_label == r for r in range(nb_labels)],
                          axis=stack_axis)
    target_mask = np.stack([target_mask_label == r for r in range(nb_labels)],
                           axis=stack_axis)

    return (np.expand_dims(style_mask, axis=0),
            np.expand_dims(target_mask, axis=0))


# Create tensor variables for images
if K.image_dim_ordering() == 'th':
    shape = (1, nb_colors, img_nrows, img_ncols)
else:
    shape = (1, img_nrows, img_ncols, nb_colors)

style_image = K.variable(preprocess_image(style_img_path))
target_image = K.placeholder(shape=shape)
if use_content_img:
    content_image = K.variable(preprocess_image(content_img_path))
else:
    content_image = K.zeros(shape=shape)

images = K.concatenate([style_image, target_image, content_image], axis=0)

# Create tensor variables for masks
raw_style_mask, raw_target_mask = load_mask_labels()
style_mask = K.variable(raw_style_mask.astype("float32"))
target_mask = K.variable(raw_target_mask.astype("float32"))
masks = K.concatenate([style_mask, target_mask], axis=0)

# index constants for images and tasks variables
STYLE, TARGET, CONTENT = 0, 1, 2

# Build image model, mask model and use layer outputs as features
# image model as VGG19
image_model = vgg16.VGG16(include_top=False, input_tensor=images)

# mask model as a series of pooling
mask_input = Input(tensor=masks, shape=(None, None, None), name="mask_input")
x = mask_input
for layer in image_model.layers[1:]:
    name = 'mask_%s' % layer.name
    if 'conv' in layer.name:
        x = AveragePooling2D((3, 3), strides=(1, 1), name=name, border_mode="same")(x)
    elif 'pool' in layer.name:
        x = AveragePooling2D((2, 2), name=name)(x)
mask_model = Model(mask_input, x)

# Collect features from image_model and task_model
image_features = {}
mask_features = {}
for img_layer, mask_layer in zip(image_model.layers, mask_model.layers):
    if 'conv' in img_layer.name:
        assert 'mask_' + img_layer.name == mask_layer.name
        layer_name = img_layer.name
        img_feat, mask_feat = img_layer.output, mask_layer.output
        image_features[layer_name] = img_feat
        mask_features[layer_name] = mask_feat


# Define loss functions
def gram_matrix(x):
    assert K.ndim(x) == 3
    features = K.batch_flatten(x)
    gram = K.dot(features, K.transpose(features))
    return gram


def region_style_loss(style_image, target_image, style_mask, target_mask):
    '''Calculate style loss between style_image and target_image,
    for one common region specified by their (boolean) masks
    '''
    assert 3 == K.ndim(style_image) == K.ndim(target_image)
    assert 2 == K.ndim(style_mask) == K.ndim(target_mask)
    if K.image_dim_ordering() == 'th':
        masked_style = style_image * style_mask
        masked_target = target_image * target_mask
        nb_channels = K.shape(style_image)[0]
    else:
        masked_style = K.permute_dimensions(
            style_image, (2, 0, 1)) * style_mask
        masked_target = K.permute_dimensions(
            target_image, (2, 0, 1)) * target_mask
        nb_channels = K.shape(style_image)[-1]
    s = gram_matrix(masked_style) / K.mean(style_mask) / nb_channels
    c = gram_matrix(masked_target) / K.mean(target_mask) / nb_channels
    return K.mean(K.square(s - c))


def style_loss(style_image, target_image, style_masks, target_masks):
    '''Calculate style loss between style_image and target_image,
    in all regions.
    '''
    assert 3 == K.ndim(style_image) == K.ndim(target_image)
    assert 3 == K.ndim(style_masks) == K.ndim(target_masks)
    loss = K.variable(0)
    for i in range(nb_labels):
        if K.image_dim_ordering() == 'th':
            style_mask = style_masks[i, :, :]
            target_mask = target_masks[i, :, :]
        else:
            style_mask = style_masks[:, :, i]
            target_mask = target_masks[:, :, i]
        loss += region_style_weight * region_style_loss(style_image, target_image, style_mask, target_mask)
    return loss


def content_loss(content_image, target_image):
    return K.sum(K.square(target_image - content_image))


def total_variation_loss(x):
    assert 4 == K.ndim(x)
    if K.image_dim_ordering() == 'th':
        a = K.square(x[:, :, :img_nrows - 1, :img_ncols - 1] -
                     x[:, :, 1:, :img_ncols - 1])
        b = K.square(x[:, :, :img_nrows - 1, :img_ncols - 1] -
                     x[:, :, :img_nrows - 1, 1:])
    else:
        a = K.square(x[:, :img_nrows - 1, :img_ncols - 1, :] -
                     x[:, 1:, :img_ncols - 1, :])
        b = K.square(x[:, :img_nrows - 1, :img_ncols - 1, :] -
                     x[:, :img_nrows - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))


# Overall loss is the weighted sum of content_loss, style_loss and tv_loss
# Each individual loss uses features from image/mask models.
loss = K.variable(0)
for layer in content_feature_layers:
    content_feat = image_features[layer][CONTENT, :, :, :]
    target_feat = image_features[layer][TARGET, :, :, :]
    loss += content_weight * content_loss(content_feat, target_feat)

for layer in style_feature_layers:
    style_feat = image_features[layer][STYLE, :, :, :]
    target_feat = image_features[layer][TARGET, :, :, :]
    style_masks = mask_features[layer][STYLE, :, :, :]
    target_masks = mask_features[layer][TARGET, :, :, :]
    sl = style_loss(style_feat, target_feat, style_masks, target_masks)
    loss += (style_weight / len(style_feature_layers)) * sl

loss += total_variation_weight * total_variation_loss(target_image)
loss_grads = K.gradients(loss, target_image)

# Evaluator class for computing efficiency
outputs = [loss]
if type(loss_grads) in {list, tuple}:
    outputs += loss_grads
else:
    outputs.append(loss_grads)

f_outputs = K.function([target_image], outputs)


def eval_loss_and_grads(x):
    if K.image_dim_ordering() == 'th':
        x = x.reshape((1, 3, img_nrows, img_ncols))
    else:
        x = x.reshape((1, img_nrows, img_ncols, 3))
    outs = f_outputs([x])
    loss_value = outs[0]
    if len(outs[1:]) == 1:
        grad_values = outs[1].flatten().astype('float64')
    else:
        grad_values = np.array(outs[1:]).flatten().astype('float64')
    return loss_value, grad_values


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

# Generate images by iterative optimization
if use_content_img:
    x = preprocess_image(content_img_path)
else:
    if K.image_dim_ordering() == 'th':
        x = np.random.uniform(0, 255, (1, 3, img_nrows, img_ncols)) - 128.
    else:
        x = np.random.uniform(0, 255, (1, img_nrows, img_ncols, 3)) - 128.

# We require original image if we are to preserve color in YCbCr mode
if preserve_color and use_content_img:
    content = imread(content_img_path, mode="YCbCr")
    content = imresize(content, (img_nrows, img_ncols))

prev_min_val = 0.

for i in range(args.num_iter):
    print('Start of iteration', i + 1)
    start_time = time.time()
    x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(), fprime=evaluator.grads, maxfun=20)

    if prev_min_val == 0:
        improvement = 0
    else:
        improvement = (prev_min_val - min_val) / prev_min_val * 100

    print("Current loss value:", min_val, " Improvement : %0.3f" % improvement, "%")
    prev_min_val = min_val

    # save current generated image
    img = deprocess_image(x.copy())

    if not use_content_img:
        img = imfilter(img, ftype='smooth')
        img = imfilter(img, ftype='sharpen')

    if use_content_img and preserve_color and content is not None:
        img = original_color_transform(content, img)

    fname = target_img_prefix + '_at_iteration_%d.png' % (i + 1)
    imsave(fname, img)
    end_time = time.time()
    print('Image saved as', fname)
    print('Iteration %d completed in %ds' % (i + 1, end_time - start_time))

    if args.min_improvement != 0.0:
        if improvement < args.min_improvement and i > 1:
            print("Script is early stopping since improvement (%0.2f) < min improvement (%0.2f)" %
                  (improvement, args.min_improvement))
            output_image = target_img_prefix + '.png'
            imsave(output_image, img)
            exit()
