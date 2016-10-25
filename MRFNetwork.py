from scipy.misc import imread, imresize, imsave
from scipy.optimize import fmin_l_bfgs_b
from sklearn.feature_extraction.image import reconstruct_from_patches_2d, extract_patches_2d
import scipy.interpolate
import scipy.ndimage

import numpy as np
import time
import os
import argparse
import h5py

from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, ZeroPadding2D, AveragePooling2D
from keras import backend as K

"""
Neural Style Transfer with Keras 1.0.6

Uses the VGG-16 model as described in the Keras example below :
https://github.com/fchollet/keras/blob/master/examples/neural_style_transfer.py

Note:

Before running this script, download the weights for the VGG16 model at:
https://drive.google.com/file/d/0Bz7KyqmuGsilT0J5dmRCM0ROVHc/view?usp=sharing
(source: https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3)
and make sure the variable `weights_path` in this script matches the location of the file.

-----------------------------------------------------------------------------------------------------------------------

"""



def _calc_patch_grid_dims(shape, patch_size, patch_stride):
    x_w, x_h, x_c = shape
    num_rows = 1 + (x_h - patch_size) // patch_stride
    num_cols = 1 + (x_w - patch_size) // patch_stride
    return num_rows, num_cols


def make_patch_grid(x, patch_size, patch_stride=1):
    '''x shape: (num_channels, rows, cols)'''
    x = x.transpose(2, 1, 0)
    patches = extract_patches_2d(x, (patch_size, patch_size))
    x_w, x_h, x_c  = x.shape
    num_rows, num_cols = _calc_patch_grid_dims(x.shape, patch_size, patch_stride)
    patches = patches.reshape((num_rows, num_cols, patch_size, patch_size, x_c))
    patches = patches.transpose((0, 1, 4, 2, 3))
    #patches = np.rollaxis(patches, -1, 2)
    return patches


def combine_patches_grid(in_patches, out_shape):
    '''Reconstruct an image from these `patches`

    input shape: (rows, cols, channels, patch_row, patch_col)
    '''
    num_rows, num_cols = in_patches.shape[:2]
    num_channels = in_patches.shape[-3]
    patch_size = in_patches.shape[-1]
    num_patches = num_rows * num_cols
    in_patches = np.reshape(in_patches, (num_patches, num_channels, patch_size, patch_size))  # (patches, channels, pr, pc)
    in_patches = np.transpose(in_patches, (0, 2, 3, 1)) # (patches, p, p, channels)
    recon = reconstruct_from_patches_2d(in_patches, out_shape)
    return recon.transpose(2, 1, 0)


class PatchMatcher(object):
    '''A matcher of image patches inspired by the PatchMatch algorithm.

    image shape: (width, height, channels)
    '''
    def __init__(self, input_shape, target_img, patch_size=1, patch_stride=1, jump_size=0.5,
            num_propagation_steps=5, num_random_steps=5, random_max_radius=1.0, random_scale=0.5):
        self.input_shape = input_shape
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.jump_size = jump_size
        self.num_propagation_steps = num_propagation_steps
        self.num_random_steps = num_random_steps
        self.random_max_radius = random_max_radius
        self.random_scale = random_scale
        self.num_input_rows, self.num_input_cols = _calc_patch_grid_dims(input_shape, patch_size, patch_stride)
        self.target_patches = make_patch_grid(target_img, patch_size)
        self.target_patches_normed = self.normalize_patches(self.target_patches)
        self.coords = np.random.uniform(0.0, 1.0,  # TODO: switch to pixels
            (2, self.num_input_rows, self.num_input_cols))# * [[[self.num_input_rows]],[[self.num_input_cols]]]
        self.similarity = np.zeros(input_shape[:2:-1], dtype ='float32')
        self.min_propagration_row = 1.0 / self.num_input_rows
        self.min_propagration_col = 1.0 / self.num_input_cols
        self.delta_row = np.array([[[self.min_propagration_row]], [[0.0]]])
        self.delta_col = np.array([[[0.0]], [[self.min_propagration_col]]])

    def update(self, input_img, reverse_propagation=False):
        input_patches = self.get_patches_for(input_img)
        self.update_with_patches(self.normalize_patches(input_patches), reverse_propagation=reverse_propagation)

    def update_with_patches(self, input_patches, reverse_propagation=False):
        self._propagate(input_patches, reverse_propagation=reverse_propagation)
        self._random_update(input_patches)

    def get_patches_for(self, img):
        return make_patch_grid(img, self.patch_size)

    def normalize_patches(self, patches):
        norm = np.sqrt(np.sum(np.square(patches), axis=(2, 3, 4), keepdims=True))
        return patches / norm

    def _propagate(self, input_patches, reverse_propagation=False):
        if reverse_propagation:
            roll_direction = 1
        else:
            roll_direction = -1
        sign = float(roll_direction)
        for step_i in range(self.num_propagation_steps):
            new_coords = self.clip_coords(np.roll(self.coords, roll_direction, 1) + self.delta_row * sign)
            coords_row, similarity_row = self.eval_state(new_coords, input_patches)
            new_coords = self.clip_coords(np.roll(self.coords, roll_direction, 2) + self.delta_col * sign)
            coords_col, similarity_col = self.eval_state(new_coords, input_patches)
            self.coords, self.similarity = self.take_best(coords_row, similarity_row, coords_col, similarity_col)

    def _random_update(self, input_patches):
        for alpha in range(1, self.num_random_steps + 1):  # NOTE this should actually stop when the move is < 1
            new_coords = self.clip_coords(self.coords + np.random.uniform(-self.random_max_radius, self.random_max_radius, self.coords.shape) * self.random_scale ** alpha)
            self.coords, self.similarity = self.eval_state(new_coords, input_patches)

    def eval_state(self, new_coords, input_patches):
        new_similarity = self.patch_similarity(input_patches, new_coords)
        delta_similarity = new_similarity - self.similarity
        coords = np.where(delta_similarity > 0, new_coords, self.coords)
        best_similarity = np.where(delta_similarity > 0, new_similarity, self.similarity)
        return coords, best_similarity

    def take_best(self, coords_a, similarity_a, coords_b, similarity_b):
        delta_similarity = similarity_a - similarity_b
        best_coords = np.where(delta_similarity > 0, coords_a, coords_b)
        best_similarity = np.where(delta_similarity > 0, similarity_a, similarity_b)
        return best_coords, best_similarity

    def patch_similarity(self, source, coords):
        '''Check the similarity of the patches specified in coords.'''
        target_vals = self.lookup_coords(self.target_patches_normed, coords)
        err = source * target_vals
        return np.sum(err, axis=(2, 3, 4))

    def clip_coords(self, coords):
        # TODO: should this all be in pixel space?
        coords = np.clip(coords, 0.0, 1.0)
        return coords

    def lookup_coords(self, x, coords):
        x_shape = np.expand_dims(np.expand_dims(x.shape, -1), -1)
        i_coords = np.round(coords * (x_shape[:2] - 1)).astype('int32')
        return x[i_coords[0], i_coords[1]]

    def get_reconstruction(self, patches=None, combined=None):
        if combined is not None:
            patches = make_patch_grid(combined, self.patch_size)
        if patches is None:
            patches = self.target_patches
        patches = self.lookup_coords(patches, self.coords)
        recon = combine_patches_grid(patches, self.input_shape)
        return recon

    def scale(self, new_shape, new_target_img):
        '''Create a new matcher of the given shape and replace its
        state with a scaled up version of the current matcher's state.
        '''
        new_matcher = PatchMatcher(new_shape, new_target_img, patch_size=self.patch_size,
                patch_stride=self.patch_stride, jump_size=self.jump_size,
                num_propagation_steps=self.num_propagation_steps,
                num_random_steps=self.num_random_steps,
                random_max_radius=self.random_max_radius,
                random_scale=self.random_scale)
        new_matcher.coords = congrid(self.coords, new_matcher.coords.shape, method='neighbour')
        new_matcher.similarity = congrid(self.similarity, new_matcher.coords.shape, method='neighbour')
        return new_matcher


def congrid(a, newdims, method='linear', centre=False, minusone=False):
    '''Arbitrary resampling of source array to new dimension sizes.
    Currently only supports maintaining the same number of dimensions.
    To use 1-D arrays, first promote them to shape (x,1).

    Uses the same parameters and creates the same co-ordinate lookup points
    as IDL''s congrid routine, which apparently originally came from a VAX/VMS
    routine of the same name.

    method:
    neighbour - closest value from original data
    nearest and linear - uses n x 1-D interpolations using
                         scipy.interpolate.interp1d
    (see Numerical Recipes for validity of use of n 1-D interpolations)
    spline - uses ndimage.map_coordinates

    centre:
    True - interpolation points are at the centres of the bins
    False - points are at the front edge of the bin

    minusone:
    For example- inarray.shape = (i,j) & new dimensions = (x,y)
    False - inarray is resampled by factors of (i/x) * (j/y)
    True - inarray is resampled by(i-1)/(x-1) * (j-1)/(y-1)
    This prevents extrapolation one element beyond bounds of input array.
    '''
    if not a.dtype in [np.float64, np.float32]:
        a = np.cast[float](a)

    m1 = np.cast[int](minusone)
    ofs = np.cast[int](centre) * 0.5
    old = np.array( a.shape )
    ndims = len( a.shape )
    if len( newdims ) != ndims:
        print ("[congrid] dimensions error. "
              "This routine currently only support "
              "rebinning to the same number of dimensions.")
        return None
    newdims = np.asarray( newdims, dtype=float )
    dimlist = []

    if method == 'neighbour':
        for i in range( ndims ):
            base = np.indices(newdims)[i]
            dimlist.append( (old[i] - m1) / (newdims[i] - m1) \
                            * (base + ofs) - ofs )
        cd = np.array( dimlist ).round().astype(int)
        newa = a[list( cd )]
        return newa

    elif method in ['nearest','linear']:
        # calculate new dims
        for i in range( ndims ):
            base = np.arange( newdims[i] )
            dimlist.append( (old[i] - m1) / (newdims[i] - m1) \
                            * (base + ofs) - ofs )
        # specify old dims
        olddims = [np.arange(i, dtype = np.float) for i in list( a.shape )]

        # first interpolation - for ndims = any
        mint = scipy.interpolate.interp1d( olddims[-1], a, kind=method )
        newa = mint( dimlist[-1] )

        trorder = [ndims - 1] + range( ndims - 1 )
        for i in range( ndims - 2, -1, -1 ):
            newa = newa.transpose( trorder )

            mint = scipy.interpolate.interp1d( olddims[i], newa, kind=method )
            newa = mint( dimlist[i] )

        if ndims > 1:
            # need one more transpose to return to original dimensions
            newa = newa.transpose( trorder )

        return newa
    elif method in ['spline']:
        oslices = [ slice(0,j) for j in old ]
        oldcoords = np.ogrid[oslices]
        nslices = [ slice(0,j) for j in list(newdims) ]
        newcoords = np.mgrid[nslices]

        newcoords_dims = [i for i in range(np.rank(newcoords))]
        #make first index last
        newcoords_dims.append(newcoords_dims.pop(0))
        newcoords_tr = newcoords.transpose(newcoords_dims)
        # makes a view that affects newcoords

        newcoords_tr += ofs

        deltas = (np.asarray(old) - m1) / (newdims - m1)
        newcoords_tr *= deltas

        newcoords_tr -= ofs

        newa = scipy.ndimage.map_coordinates(a, newcoords)
        return newa
    else:
        print("Congrid error: Unrecognized interpolation type.\n",
              "Currently only \'neighbour\', \'nearest\',\'linear\',",
              "and \'spline\' are supported.")
        return None

class BaseModel(object):
    '''Model to be extended.'''
    def __init__(self, net, args):
        self.set_net(net)
        self.args = args

    def set_net(self, net):
        self.net = net
        self.net_input = net.layers[0].input
        self.layer_map = dict([(layer.name, layer) for layer in self.net.layers])
        self._f_layer_outputs = {}

    def build(self, a_image, ap_image, b_image, output_shape):
        self.output_shape = output_shape
        loss = self.build_loss(a_image, ap_image, b_image)
        # get the gradients of the generated image wrt the loss
        grads = K.gradients(loss, self.net_input)
        outputs = [loss]
        if type(grads) in {list, tuple}:
            outputs += grads
        else:
            outputs.append(grads)
        self.f_outputs = K.function([self.net_input], outputs)

    def build_loss(self, a_image, ap_image, b_image):
        '''Create an expression for the loss as a function of the image inputs.'''
        loss = K.variable(0.0)
        # get the symbolic outputs of each "key" layer (we gave them unique names).
        loss += self.args.tv_weight * total_variation_loss(self.net_input)
        return loss

    def precompute_static_features(self, a_image, ap_image, b_image):
        # figure out which layers we need to extract
        a_layers, ap_layers, b_layers = set(), set(), set()
        if self.args.analogy_weight:
            for layerset in (a_layers, ap_layers, b_layers):
                layerset.update(analogy_layers)
        if self.args.mrf_weight:
            ap_layers.update(mrf_layers)
        if self.args.b_bp_content_weight:
            b_layers.update(feature_layers)
        if self.args.style_weight:
            ap_layers.add(args.content_layer)
        # let's get those features
        all_a_features = self.get_features(a_image, a_layers)
        all_ap_image_features = self.get_features(ap_image, ap_layers)
        all_b_features = self.get_features(b_image, b_layers)
        return all_a_features, all_ap_image_features, all_b_features

    def get_features(self, x, layers):
        if not layers:
            return None
        f = K.function([self.net_input], [self.get_layer_output(layer_name) for layer_name in layers])
        feature_outputs = f([x])
        features = dict(zip(layers, feature_outputs))
        return features

    def get_f_layer(self, layer_name):
        return K.function([self.net_input], [self.get_layer_output(layer_name)])

    def get_layer_output(self, name):
        if not name in self._f_layer_outputs:
            layer = self.layer_map[name]
            self._f_layer_outputs[name] = layer.output
        return self._f_layer_outputs[name]

    def get_layer_output_shape(self, name):
        layer = self.layer_map[name]
        return layer.output_shape

    def eval_loss_and_grads(self, x):
        x = x.reshape(self.output_shape)
        outs = self.f_outputs([x])
        loss_value = outs[0]
        if len(outs[1:]) == 1:
            grad_values = outs[1].flatten().astype('float64')
        else:
            grad_values = np.array(outs[1:]).flatten().astype('float64')
        return loss_value, grad_values


class NNFModel(BaseModel):
    '''Faster model for image analogies.'''
    def build(self, a_image, ap_image, b_image, output_shape):
        self.output_shape = output_shape
        loss = self.build_loss(a_image, ap_image, b_image)
        # get the gradients of the generated image wrt the loss
        grads = K.gradients(loss, self.net_input)
        outputs = [loss]
        if type(grads) in {list, tuple}:
            outputs += grads
        else:
            outputs.append(grads)
        f_inputs = [self.net_input]
        for nnf in self.feature_nnfs:
            f_inputs.append(nnf.placeholder)
        self.f_outputs = K.function(f_inputs, outputs)

    def eval_loss_and_grads(self, x):
        x = x.reshape(self.output_shape)
        f_inputs = [x]
        # update the patch indexes
        start_t = time.time()
        for nnf in self.feature_nnfs:
            nnf.update(x, num_steps=self.args.mrf_nnf_steps)
            new_target = nnf.matcher.get_reconstruction()
            f_inputs.append(new_target)
        print('PatchMatch update in {:.2f} seconds'.format(time.time() - start_t))
        # run it through
        outs = self.f_outputs(f_inputs)
        loss_value = outs[0]
        if len(outs[1:]) == 1:
            grad_values = outs[1].flatten().astype('float64')
        else:
            grad_values = np.array(outs[1:]).flatten().astype('float64')
        return loss_value, grad_values

    def build_loss(self, a_image, ap_image, b_image):
        '''Create an expression for the loss as a function of the image inputs.'''
        print('Building loss...')
        loss = super(NNFModel, self).build_loss(a_image, ap_image, b_image)
        # Precompute static features for performance
        print('Precomputing static features...')
        all_a_features, all_ap_image_features, all_b_features = self.precompute_static_features(a_image, ap_image, b_image)
        print('Building and combining losses...')
        if self.args.analogy_weight:
            for layer_name in analogy_layers:
                a_features = all_a_features[layer_name][0]
                ap_image_features = all_ap_image_features[layer_name][0]
                b_features = all_b_features[layer_name][0]
                # current combined output
                layer_features = self.get_layer_output(layer_name)
                combination_features = layer_features[0, :, :, :]
                al = nnf_analogy_loss(
                    a_features, ap_image_features, b_features, combination_features,
                    num_steps=self.args.analogy_nnf_steps, patch_size=self.args.patch_size,
                    patch_stride=self.args.patch_stride, jump_size=1.0)
                loss += (self.args.analogy_weight / len(analogy_layers)) * al

        existing_feature_nnfs = getattr(self, 'feature_nnfs', [None] * len(mrf_layers))
        self.feature_nnfs = []
        if self.args.mrf_weight:
            for layer_name, existing_nnf in zip(mrf_layers, existing_feature_nnfs):
                ap_image_features = all_ap_image_features[layer_name][0]
                # current combined output
                layer_features = self.get_layer_output(layer_name)
                combination_features = layer_features[0, :, :, :]
                input_shape = self.get_layer_output_shape(layer_name)
                if existing_nnf and not self.args.randomize_mnf_nnf:
                    matcher = existing_nnf.matcher.scale((input_shape[3], input_shape[2], input_shape[1]), ap_image_features)
                else:
                    matcher = PatchMatcher(
                        (input_shape[3], input_shape[2], input_shape[1]), ap_image_features,
                        patch_size=self.args.patch_size, jump_size=1.0, patch_stride=self.args.patch_stride)
                nnf = NNFState(matcher, self.get_f_layer(layer_name))
                self.feature_nnfs.append(nnf)
                sl = content_loss(combination_features, nnf.placeholder)
                loss += (self.args.mrf_weight / len(mrf_layers)) * sl

        if self.args.content_weight:
            for layer_name in feature_layers:
                b_features = K.variable(all_b_features[layer_name][0])
                # current combined output
                bp_features = self.get_layer_output(layer_name)
                cl = content_loss(bp_features, b_features)
                loss += self.args.content_weight / len(feature_layers) * cl

        if self.args.style_weight != 0.0:
            #for layer_name in self.args.content_layer:
                layer_name = self.args.content_layer
                ap_image_features = K.variable(all_ap_image_features[layer_name][0])
                layer_features = self.get_layer_output(layer_name)
                layer_shape = self.get_layer_output_shape(layer_name)
                # current combined output
                combination_features = layer_features[0, :, :, :]
                nsl = style_loss(ap_image_features, combination_features)
                loss += (self.args.style_weight / len(self.args.content_layer)) * nsl

        return loss

class AnalogyModel(BaseModel):
    '''Brute Force Model for image analogies.'''

    def build_loss(self, a_image, ap_image, b_image):
        '''Create an expression for the loss as a function of the image inputs.'''
        print('Building loss...')
        loss = super(AnalogyModel, self).build_loss(a_image, ap_image, b_image)
        # Precompute static features for performance
        print('Precomputing static features...')
        all_a_features, all_ap_image_features, all_b_features = self.precompute_static_features(a_image, ap_image, b_image)
        print('Building and combining losses...')
        if self.args.analogy_weight != 0.0:
            for layer_name in analogy_layers:
                a_features = all_a_features[layer_name][0]
                ap_image_features = all_ap_image_features[layer_name][0]
                b_features = all_b_features[layer_name][0]
                # current combined output
                layer_features = self.get_layer_output(layer_name)
                combination_features = layer_features[0, :, :, :]
                al = analogy_loss(a_features, ap_image_features,
                    b_features, combination_features,
                    use_full_analogy=self.args.use_full_analogy,
                    patch_size=self.args.patch_size,
                    patch_stride=self.args.patch_stride)
                loss += (self.args.analogy_weight / len(self.args.analogy_layers)) * al

        if self.args.mrf_weight != 0.0:
            for layer_name in mrf_layers:
                ap_image_features = K.variable(all_ap_image_features[layer_name][0])
                layer_features = self.get_layer_output(layer_name)
                # current combined output
                combination_features = layer_features[0, :, :, :]
                sl = mrf_loss(ap_image_features, combination_features,
                    patch_size=self.args.patch_size,
                    patch_stride=self.args.patch_stride)
                loss += (self.args.mrf_weight / len(self.args.mrf_layers)) * sl

        if self.args.b_bp_content_weight != 0.0:
            for layer_name in feature_layers:
                b_features = K.variable(all_b_features[layer_name][0])
                # current combined output
                bp_features = self.get_layer_output(layer_name)
                cl = content_loss(bp_features, b_features)
                loss += self.args.content_weight / len(feature_layers) * cl

        if self.args.style_weight != 0.0:
            for layer_name in self.args.content_layer:
                ap_image_features = K.variable(all_ap_image_features[layer_name][0])
                layer_features = self.get_layer_output(layer_name)
                layer_shape = self.get_layer_output_shape(layer_name)
                # current combined output
                combination_features = layer_features[0, :, :, :]
                nsl = style_loss(ap_image_features, combination_features)
                loss += (self.args.style_weight / len(self.args.content_layer)) * nsl
        return loss


"""

This is the beginning of the program

"""

parser = argparse.ArgumentParser(description='Neural style transfer with Keras.')
parser.add_argument('style_image', metavar='base', type=str,
                    help='Path to the image to transform.')
parser.add_argument('style_image_mask_path', metavar='base_mask', type=str,
                    help='Path to the image mask.')
parser.add_argument('content_image_path', metavar='ref', type=str,
                    help='Path to the style reference image.')
parser.add_argument('result_prefix', metavar='res_prefix', type=str,
                    help='Prefix for the saved results.')

parser.add_argument("--image_size", dest="img_size", default=512, type=int, help='Output Image size')
parser.add_argument("--content_weight", dest="content_weight", default=0.025, type=float, help="Weight of content") # 0.025
parser.add_argument("--style_weight", dest="style_weight", default=1, type=float, help="Weight of content") # 1.0
parser.add_argument("--style_scale", dest="style_scale", default=1.0, type=float, help="Scale the weightage of the style") # 1, 0.5, 2
parser.add_argument("--total_variation_weight", dest="tv_weight", default=1e-3, type=float, help="Total Variation in the Weights") # 1.0
parser.add_argument("--num_iter", dest="num_iter", default=10, type=int, help="Number of iterations")
parser.add_argument("--rescale_image", dest="rescale_image", default="True", type=str, help="Rescale image after execution to original dimentions")
parser.add_argument("--rescale_method", dest="rescale_method", default="bilinear", type=str, help="Rescale image algorithm")
parser.add_argument("--maintain_aspect_ratio", dest="maintain_aspect_ratio", default="True", type=str, help="Maintain aspect ratio of image")
parser.add_argument("--content_layer", dest="content_layer", default="conv5_2", type=str, help="Optional 'conv4_2'")
parser.add_argument("--init_image", dest="init_image", default="content", type=str, help="Initial image used to generate the final image. Options are 'content' or 'noise")
parser.add_argument('--analogy-w', dest='analogy_weight', type=float, default=1.0, help='Weight for analogy loss.')
parser.add_argument('--analogy-layers', dest='analogy_layers', default=['conv3_1', 'conv4_1'], help='Comma-separated list of layer names to be used for the analogy loss')
parser.add_argument('--use-full-analogy', dest='use_full_analogy', action="store_true", help='Use the full set of analogy patches (slower/more memory but maybe more accurate)')
parser.add_argument('--mrf-w', dest='mrf_weight', type=float, default=1, help='Weight for MRF loss between A\' and B\'')
parser.add_argument('--mrf-layers', dest='mrf_layers', default=['conv3_1', 'conv4_1'], help='Comma-separated list of layer names to be used for the MRF loss')
parser.add_argument('--b-content-w', dest='b_bp_content_weight', type=float, default=4.0, help='Weight for content loss between B and B\'')
parser.add_argument('--mrf-nnf-steps', dest='mrf_nnf_steps', type=int, default=5, help='Number of patchmatch updates per iteration for local coherence loss.')
parser.add_argument('--analogy-nnf-steps', dest='analogy_nnf_steps', type=int, default=15, help='Number of patchmatch updates for the analogy loss (done once per scale).')
parser.add_argument('--patch-stride', dest='patch_stride', type=int, default=1, help='Patch stride used for matching. Currently required to be 1.')
parser.add_argument('--randomize-mrf-nnf', dest='randomize_mnf_nnf', action='store_true', help='Randomize the local coherence similarity matrix at the start of a new scale instead of scaling it up.')
parser.add_argument('--patch-size', dest='patch_size', type=int, default=1, help='Patch size used for matching.')

args = parser.parse_args()
style_image_path = args.style_image
style_image_mask_path = args.style_image_mask_path
content_image_path = args.content_image_path
result_prefix = args.result_prefix
weights_path = r"vgg16_weights.h5"

def strToBool(v):
    return v.lower() in ("true", "yes", "t", "1")

rescale_image = strToBool(args.rescale_image)
maintain_aspect_ratio = strToBool(args.maintain_aspect_ratio)

# these are the weights of the different loss components
total_variation_weight = args.tv_weight
style_weight = args.style_weight * args.style_scale
content_weight = args.content_weight

# dimensions of the generated picture.
img_width = img_height = args.img_size
assert img_height == img_width, 'Due to the use of the Gram matrix, width and height must match.'

analogy_layers  =['conv4_1'] # 'conv1_1','conv2_1','conv3_1', 'conv4_1'
mrf_layers = ['conv4_1', 'conv5_1'] # 'conv3_1', 'conv4_1'

img_WIDTH = img_HEIGHT = 0
aspect_ratio = 0
b_scale_ratio_width = b_scale_ratio_height = 0

# util function to open, resize and format pictures into appropriate tensors
def preprocess_image(image_path, load_dims=False, style_image=False):
    global img_WIDTH, img_HEIGHT, aspect_ratio, b_scale_ratio_height, b_scale_ratio_width

    img = imread(image_path, mode="RGB") # Prevents crashes due to PNG images (ARGB)
    if load_dims:
        img_WIDTH = img.shape[0]
        img_HEIGHT = img.shape[1]
        aspect_ratio = img_HEIGHT / img_WIDTH

    if style_image:
        b_scale_ratio_width = float(img.shape[0]) / img_WIDTH
        b_scale_ratio_height = float(img.shape[1]) / img_HEIGHT

    img = imresize(img, (img_width, img_height))
    img = img.transpose((2, 0, 1)).astype('float64')
    img = np.expand_dims(img, axis=0)
    return img

# util function to convert a tensor into a valid image
def deprocess_image(x):
    x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def make_patches(x, patch_size, patch_stride):
    '''Break image `x` up into a bunch of patches.'''
    from theano.tensor.nnet.neighbours import images2neibs
    x = K.expand_dims(x, 0)
    patches = images2neibs(x,
        (patch_size, patch_size), (patch_stride, patch_stride),
        mode='valid')
    # neibs are sorted per-channel
    patches = K.reshape(patches, (K.shape(x)[1], K.shape(patches)[0] // K.shape(x)[1], patch_size, patch_size))
    patches = K.permute_dimensions(patches, (1, 0, 2, 3))
    patches_norm = K.sqrt(K.sum(K.square(patches), axis=(1,2,3), keepdims=True))
    return patches, patches_norm

def combine_patches(patches, out_shape):
    '''Reconstruct an image from these `patches`'''
    patches = patches.transpose(0, 2, 3, 1)
    recon = reconstruct_from_patches_2d(patches, out_shape)
    return recon.transpose(2, 0, 1)


def find_patch_matches(a, a_norm, b):
    '''For each patch in A, find the best matching patch in B'''
    # we want cross-correlation here so flip the kernels
    convs = K.conv2d(a, b[:, :, ::-1, ::-1], border_mode='valid')
    argmax = K.argmax(convs / a_norm, axis=1)
    return argmax

def find_analogy_patches(a, a_prime, b, patch_size=3, patch_stride=1):
    '''This is for precalculating the analogy_loss

    Since A, A', and B never change we only need to calculate the patch matches once.
    '''
    # extract patches from feature maps
    a_patches, a_patches_norm = make_patches(K.variable(a), patch_size, patch_stride)
    a_prime_patches, a_prime_patches_norm = make_patches(K.variable(a_prime), patch_size, patch_stride)
    b_patches, b_patches_norm = make_patches(K.variable(b), patch_size, patch_stride)
    # find best patches and calculate loss
    p = find_patch_matches(b_patches, b_patches_norm, a_patches / a_patches_norm)
    #best_patches = a_prime_patches[p]
    best_patches = K.reshape(a_prime_patches[p], K.shape(b_patches))
    f = K.function([], best_patches)
    best_patches = f([])
    return best_patches

def make_patches_grid(x, patch_size, patch_stride):
    '''Break image `x` up into a grid of patches.

    input shape: (channels, rows, cols)
    output shape: (rows, cols, channels, patch_rows, patch_cols)
    '''
    from theano.tensor.nnet.neighbours import images2neibs  # TODO: all K, no T
    x = K.expand_dims(x, 0)
    xs = K.shape(x)
    num_rows = 1 + (xs[-2] - patch_size) // patch_stride
    num_cols = 1 + (xs[-1] - patch_size) // patch_stride
    num_channels = xs[-3]
    patches = images2neibs(x,
        (patch_size, patch_size), (patch_stride, patch_stride),
        mode='valid')
    # neibs are sorted per-channel
    patches = K.reshape(patches, (num_channels, K.shape(patches)[0] // num_channels, patch_size, patch_size))
    patches = K.permute_dimensions(patches, (1, 0, 2, 3))
    # arrange in a 2d-grid (rows, cols, channels, px, py)
    patches = K.reshape(patches, (num_rows, num_cols, num_channels, patch_size, patch_size))
    patches_norm = K.sqrt(K.sum(K.square(patches), axis=(2,3,4), keepdims=True))
    return patches, patches_norm


# get tensor representations of our images
content_img = preprocess_image(content_image_path, load_dims=True)
style_img = preprocess_image(style_image_path, style_image=True)
style_img_mask = preprocess_image(style_image_mask_path)

style_image_tensor = K.variable(style_img)
style_image_mask_tensor = K.variable(style_img_mask)
content_image_tensor = K.variable(content_img)

# this will contain our generated image
combination_image = K.placeholder((1, 3, img_width, img_height))

# combine the 4 images into a single Keras tensor
input_tensor = K.concatenate([style_image_tensor,
                              style_image_mask_tensor,
                              content_image_tensor,
                              combination_image], axis=0)

# build the VGG16 network with our 3 images as input
first_layer = ZeroPadding2D((1, 1), )
first_layer.set_input(input_tensor, shape=(4, 3, img_width, img_height))

model = Sequential()
model.add(first_layer)
model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
model.add(AveragePooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
model.add(AveragePooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
model.add(AveragePooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
model.add(AveragePooling2D((2, 2), strides=(2, 2)))

model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
model.add(ZeroPadding2D((1, 1)))
model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
model.add(AveragePooling2D((2, 2), strides=(2, 2)))

# load the weights of the VGG16 networks
# (trained on ImageNet, won the ILSVRC competition in 2014)
# note: when there is a complete match between your model definition
# and your weight savefile, you can simply call model.load_weights(filename)
assert os.path.exists(weights_path), 'Model weights not found (see "weights_path" variable in script).'
f = h5py.File(weights_path)
for k in range(f.attrs['nb_layers']):
    if k >= len(model.layers):
        # we don't look at the last (fully-connected) layers in the savefile
        break
    g = f['layer_{}'.format(k)]
    weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
    model.layers[k].set_weights(weights)
f.close()
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
    a = K.square(x[:, :, 1:, :img_width - 1] - x[:, :, :img_height - 1, :img_width - 1])
    b = K.square(x[:, :, :img_height - 1, 1:] - x[:, :, :img_width - 1, :img_height - 1])
    #a = K.square(x[:, :, :img_width-1, :img_height-1] - x[:, :, 1:, :img_height-1])
    #b = K.square(x[:, :, :img_width-1, :img_height-1] - x[:, :, :img_width-1, 1:])
    return K.sum(K.pow(a + b, 1.25))


def analogy_loss(a, a_prime, b, b_prime, patch_size=3, patch_stride=1, use_full_analogy=False):
    '''http://www.mrl.nyu.edu/projects/image-analogies/index.html'''
    best_a_prime_patches = find_analogy_patches(a, a_prime, b, patch_size=patch_size, patch_stride=patch_stride)
    if use_full_analogy:  # combine all the patches into a single image
        b_prime_patches, _ = make_patches(b_prime, patch_size, patch_stride)
        loss = content_loss(best_a_prime_patches, b_prime_patches) / patch_size ** 2
    else:
        bs = b.shape
        b_analogy = combine_patches(best_a_prime_patches, (bs[1], bs[2], bs[0]))
        loss = content_loss(np.expand_dims(b_analogy, 0), b_prime)
    return loss

def mrf_loss(source, combination, patch_size=3, patch_stride=1):
    '''CNNMRF http://arxiv.org/pdf/1601.04589v1.pdf'''
    # extract patches from feature maps
    combination_patches, combination_patches_norm = make_patches(combination, patch_size, patch_stride)
    source_patches, source_patches_norm = make_patches(source, patch_size, patch_stride)
    # find best patches and calculate loss
    patch_ids = find_patch_matches(combination_patches, combination_patches_norm, source_patches / source_patches_norm)
    best_source_patches = K.reshape(source_patches[patch_ids], K.shape(combination_patches))
    loss = K.sum(K.square(best_source_patches - combination_patches)) / patch_size ** 2
    return loss

def nnf_analogy_loss(a, a_prime, b, b_prime, num_steps=5, jump_size=1.0, patch_size=1, patch_stride=1):
    '''image shapes: (channels, rows, cols)
    '''
    bs = b.shape
    matcher = PatchMatcher((bs[2], bs[1], bs[0]), a, jump_size=jump_size, patch_size=patch_size, patch_stride=patch_stride)
    b_patches = matcher.get_patches_for(b)
    b_normed = matcher.normalize_patches(b_patches)
    for i in range(num_steps):
        matcher.update_with_patches(b_normed, reverse_propagation=bool(i % 2))
    target = matcher.get_reconstruction(combined=a_prime)
    loss = content_loss(target, b_prime)
    return loss

class NNFState(object):
    def __init__(self, matcher, f_layer):
        self.matcher = matcher
        mis = matcher.input_shape
        self.placeholder = K.placeholder(mis[::-1])
        self.f_layer = f_layer

    def update(self, x, num_steps=5):
        x_f = self.f_layer([x])[0]
        x_patches = self.matcher.get_patches_for(x_f[0])
        x_normed = self.matcher.normalize_patches(x_patches)
        for i in range(num_steps):
            self.matcher.update_with_patches(x_normed, reverse_propagation=bool(i % 2))


# combine these loss functions into a single scalar
loss = K.variable(0.)
layer_features = outputs_dict[args.content_layer] # 'conv5_2' or 'conv4_2'
base_image_features = layer_features[0, :, :, :]
combination_features = layer_features[3, :, :, :]
loss += content_weight * content_loss(base_image_features,
                                      combination_features)

feature_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
for layer_name in feature_layers:
    layer_features = outputs_dict[layer_name]
    style_reference_features = layer_features[1, :, :, :]
    combination_features = layer_features[3, :, :, :]
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
    x = x.reshape((1, 3, img_width, img_height))
    outs = f_outputs([x])
    loss_value = outs[0]
    if len(outs[1:]) == 1:
        grad_values = outs[1].flatten().astype('float64')
    else:
        grad_values = np.array(outs[1:]).flatten().astype('float64')
    return loss_value, grad_values
"""
"""
# this Evaluator class makes it possible
# to compute loss and gradients in one pass
# while retrieving them via two separate functions,
# "loss" and "grads". This is done because scipy.optimize
# requires separate functions for loss and gradients,
# but computing them separately would be inefficient.
class Evaluator(object):
    def __init__(self, model):
        self.loss_value = None
        self.grads_values = None
        self.model = model

    def loss(self, x):
        assert self.loss_value is None
        loss_value, grad_values = self.model.eval_loss_and_grads(x)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values


nnf = NNFModel(model, args)
nnf.build(style_img, style_img_mask, content_img, (1, 3, img_width, img_height))


#analogy = AnalogyModel(model, args)
#analogy.build(style_img, style_img_mask, content_img, (1, 3, img_width, img_height))
evaluator = Evaluator(nnf)

# run scipy-based optimization (L-BFGS) over the pixels of the generated image
# so as to minimize the neural style loss
assert args.init_image in ["content", "noise"] , "init_image must be one of ['original', 'noise']"
if "content" in args.init_image:
    x = preprocess_image(content_image_path, True, )
else:
    x = np.random.uniform(0, 255, (1, 3, img_width, img_height))


num_iter = args.num_iter
for i in range(num_iter):
    print('Start of iteration', (i+1))
    start_time = time.time()

    x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(),
                                     fprime=evaluator.grads, maxfun=20, maxiter=20, m=4)
    print('Current loss value:', min_val)
    # save current generated image
    img = deprocess_image(x.reshape((3, img_width, img_height)))

    if (maintain_aspect_ratio) & (not rescale_image):
        img_ht = int(img_width * aspect_ratio)
        print(img_width, img_ht)
        print("Rescaling Image to (%d, %d)" % (img_width, img_ht))
        img = imresize(img, (img_width, img_ht), interp=args.rescale_method)


    if rescale_image:
        print("Rescaling Image to (%d, %d)" % (img_WIDTH, img_HEIGHT))
        img = imresize(img, (img_WIDTH, img_HEIGHT), interp=args.rescale_method)

    fname = result_prefix + '_at_iteration_%d.png' % (i+1)
    imsave(fname, img)
    end_time = time.time()
    print('Image saved as', fname)
    print('Iteration %d completed in %ds' % (i+1, end_time - start_time))
