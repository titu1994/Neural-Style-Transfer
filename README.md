# Neural Style Transfer
Implementation of Neural Style Transfer from the paper <a href="http://arxiv.org/abs/1508.06576">A Neural Algorithm of Artistic Style</a> in Keras 1.0.8. 

INetwork implements and focuses on certain improvements suggested in <a href="http://arxiv.org/abs/1605.04603">Improving the Neural Algorithm of Artistic Style</a>. 

Uses the VGG-16 model as described in the Keras example below :
https://github.com/fchollet/keras/blob/master/examples/neural_style_transfer.py

Uses weights from Keras Deep Learning Models : https://github.com/fchollet/deep-learning-models

## Guide

See the <a href="https://github.com/titu1994/Neural-Style-Transfer/blob/master/Guide.md">guide</a> for details regarding how to use the script to acheive the best results

It also explains how to setup Theano (with GPU support) on both Windows and Linux. Theano on Windows is a long and tedious process, so the guide can speed up the process by simply letting you finish all the steps in the correct order, so as not to screw up the finicky Theano + Windows setup.

# Examples
<img src="https://raw.githubusercontent.com/titu1994/Neural_Style_Transfer/master/images/inputs/content/blue-moon-lake.jpg" width=45% height=300> <img src="https://raw.githubusercontent.com/titu1994/Neural_Style_Transfer/master/images/inputs/style/starry_night.jpg" width=45% height=300>
<br> Result after 50 iterations (Average Pooling) <br>
<img src="https://raw.githubusercontent.com/titu1994/Neural_Style_Transfer/master/images/output/Blue_Moon_Lake_iteration_50.jpg" width=90% height=450>
<br><br> For comparison, results after 50 iterations (Max Pooling) <br>
<img src="https://raw.githubusercontent.com/titu1994/Neural_Style_Transfer/master/images/output/Tsukiyomi_at_iteration_100-Max-Pooling.jpg" width=90% height=450>
<br><br> For comparison, results after 50 iterations using the INetwork. Notice that in comparison to Max Pooling results, it is far more detailed in the mountain peaks and colours are more natural<br>
<img src="https://github.com/titu1994/Neural-Style-Transfer/blob/master/images/output/Tsukiyomi_INetwork_at_iteration_50.jpg?raw=true" width=90% height=450>
<br><br> DeepArt.io result (1000 iterations and using improvements such as Markov Random Field Regularization) <br>
<img src="https://raw.githubusercontent.com/titu1994/Neural_Style_Transfer/master/images/output/DeepArt_Blue_Moon_Lake.png" width=90% height=450>
<br><br> For reference, the same image with Color Preservation 
<img src="https://github.com/titu1994/Neural-Style-Transfer/blob/master/images/output/Tsukiyomi_color_preserved.jpg?raw=true" width=90% height=450>
<br><br> As an example, here are two images of the Sagano Bamboo Forest with the "pattened-leaf" style, with and without color preservation <br>
<img src="https://github.com/titu1994/Neural-Style-Transfer/blob/master/images/inputs/content/sagano_bamboo_forest.jpg?raw=true" height=450 width=45%> <img src="https://github.com/titu1994/Neural-Style-Transfer/blob/master/images/inputs/style/patterned_leaves.jpg?raw=true" height=450 width=45%>
<br>
<img src="https://github.com/titu1994/Neural-Style-Transfer/blob/master/images/output/Bamboo-Fores.jpg?raw=true" height=450 width=45%> <img src="https://github.com/titu1994/Neural-Style-Transfer/blob/master/images/output/Bamboo-Forest-No-Color-Preservation.jpg?raw=true" height=450 width=45%>

## Weights (VGG 16)

Weights are now automatically downloaded and cached in the ~/.keras (Users/<username>/.keras for Windows) folder under the 'models' subdirectory. The weights are a smaller version which include only the Convolutional layers without Zero Padding Layers, thereby increasing the speed of execution.

Note: Requires the latest version of Keras (1.0.7+) due to use of new methods to get files and cache them into .keras directory.

## Modifications to original implementation :
- Uses 'conv5_2' output to measure content loss.
Original paper utilizes 'conv4_2' output

- Initial image used for image is the base image (instead of random noise image)
This method tends to create better output images, however parameters have to be well tuned.
Therefore their is a argument 'init_image' which can take the options 'content' or 'noise'

- Can use AveragePooling2D inplace of MaxPooling2D layers
The original paper uses AveragePooling for better results, but this can be changed to use MaxPooling2D layers via the argument `--pool_type="max"`. By default MaxPoooling is used, since if offers sharper images, but AveragePooling applys the style better in some cases (especially when style image is the "Starry Night" by Van Goph.

- Style weight scaling
- Rescaling of image to original dimensions, using lossy upscaling present
- Maintain aspect ratio of intermediate and final stage images, using lossy upscaling

## Improvements in INetwork
- Improvement 3.1 in paper : Geometric Layer weight adjustment for Style inference
- Improvement 3.2 in paper : Using all layers of VGG-16 for style inference
- Improvement 3.3 in paper : Activation Shift of gram matrix
- Improvement 3.5 in paper : Correlation Chain

These improvements are almost same as the Chain Blurred version, however a few differences exist : 
- Blurring of gram matrix G is not used, as in the paper the author concludes that the results are often not major, and convergence speed is greatly diminished due to very complex gradients.
- Only one layer for Content inference instead of using all the layers as suggested in the Chain Blurred version.
- Does not use CNN MRF network, but applies these modifications to the original algorithm.
- All of this is applied on the VGG-16 network, not on the VGG-19 network. It is trivial to extrapolate this to the VGG-19 network. Simply adding the layer names to the `feature_layers` list will be sufficient to apply these changes to the VGG-19 network. 

## Windows Helper
It is a C# program written to more easily generate the arguments for the python script Network.py or INetwork.py

<img src="https://github.com/titu1994/Neural-Style-Transfer/blob/master/images/Neural%20Art%20Windows.JPG?raw=true" height=600 width=90%>

- Upon first run, it will request the python path. Traverse your directory to locate the python.exe of your choice (Anaconda is tested)

### Benefits 
- Automatically executes the script based on the arguments.
- Easy selection of images (Content, Style, Output Prefix)
- Easy parameter selection
- Easily generate argument list, if command line execution is preferred. 

## Usage
Both Network.py and INetwork.py have similar usage styles, and share all parameters.

Network.py
```
python network.py "/path/to/content image" "path/to/style image" "result prefix or /path/to/result prefix"
```

INetwork.py
```
python inetwork.py "/path/to/content image" "path/to/style image" "result prefix or /path/to/result prefix"
```

There are various parameters discussed below which can be modified to alter the output image. Note that many parameters require the command to be enclosed in double quotes ( " " ).

Example:
```
python inetwork.py "/path/to/content image" "path/to/style image" "result prefix or /path/to/result prefix" --preserve_color "True" --pool_type "ave" --rescale_method "bicubic" --content_layer "conv4_2"
```

## Parameters
```
--image_size : Allows to set the Gram Matrix size. Default is 400 x 400, since it produces good results fast. 
--num_iter : Number of iterations. Default is 10. Test the output with 10 iterations, and increase to improve results.
--init_image : Can be "content" or "noise". Default is "content", since it reduces reproduction noise.
--pool_type : Pooling type. MaxPooling ("max") is default. For smoother images, use AveragePooling ("ave").
--preserve_color : Preserves the original color space of the content image, while applying only style. Post processing technique on final image, therefore does not harm quality of style.
--min_improvement : Sets the minimum improvement required to continue training. Default is 0.0, indicating no minimum threshold. Advised values are 0.05 or 0.01

--content_weight : Weightage given to content in relation to style. Default if 0.025
--style_weight : Weightage given to style in relation to content. Default is 1. 
--style_scale : Scales the style_weight. Default is 1. 
--total_variation_weight : Regularization factor. Smaller values tend to produce crisp images, but 0 is not useful. Default = 8.5E-5

--rescale_image : Rescale image to original dimensions after each iteration. (Bilinear upscaling)
--rescale_method : Rescaling algorithm. Default is bilinear. Options are nearest, bilinear, bicubic and cubic.
--maintain_aspect_ratio : Rescale the image just to the original aspect ratio. Size will be (gram_matrix_size, gram_matrix_size * aspect_ratio). Default is True
--content_layer : Selects the content layer. Paper suggests conv4_2, but better results can be obtained from conv5_2. Default is conv5_2.
```

# Network.py in action
![Alt Text](https://raw.githubusercontent.com/titu1994/Neural-Style-Transfer/master/images/Blue%20Moon%20Lake.gif)

# Requirements 
- Theano / Tensorflow
- Keras 
- CUDA (GPU) -- Recommended
- CUDNN (GPU) -- Recommended 
- Scipy + PIL
- Numpy

# Speed
On a 980M GPU, the time required for each epoch depends on mainly image size (gram matrix size) :

For a 400x400 gram matrix, each epoch takes approximately 8-10 seconds. <br>
For a 512x512 gram matrix, each epoch takes approximately 15-18 seconds. <br>
For a 600x600 gram matrix, each epoch takes approximately 24-28 seconds. <br>
  
# Issues
- Due to usage of content image as initial image, output depends heavily on parameter tuning. <br> Test to see if the image is appropriate in the first 10 epochs, and if it is correct, increase the number of iterations to smoothen and improve the quality of the output.
- Due to small gram sizes, the output image is usually small. 
<br> To correct this, use the implementations of this paper "Image Super-Resolution Using Deep Convolutional Networks" http://arxiv.org/abs/1501.00092 to upscale the images with minimal loss.
<br> Some implementations of the above paper for Windows : https://github.com/lltcggie/waifu2x-caffe/releases <br>(Download the waifu2x-caffe.zip and extract, program supports English)
- Implementation of Markov Random Field Regularization and Patch Match algorithm are currently being tested. MRFNetwork.py contains the basic code, which need to be integrated to use MRF and Patch Match as in Image Analogies paper <a href="http://arxiv.org/abs/1601.04589"> Combining Markov Random Fields and Convolutional Neural Networks for Image Synthesis </a>


