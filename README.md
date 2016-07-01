# Neural Style Transfer
Implementation of Neural Style Transfer from the paper A Neural Algorithm of Artistic Style(http://arxiv.org/abs/1508.06576) in Keras 1.0.4

Uses the VGG-16 model as described in the Keras example below :
https://github.com/fchollet/keras/blob/master/examples/neural_style_transfer.py

## Weights (VGG 16)

Before running this script, download the weights for the VGG16 model at:
https://drive.google.com/file/d/0Bz7KyqmuGsilT0J5dmRCM0ROVHc/view?usp=sharing
(source: https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3)
and make sure the variable `weights_path` in this script matches the location of the file.

- Save the weights in the root folder
- Save a copy of the weights in the root of the windows_helper folder (if you wish to use the Windows Helper tool to easily execute the script)

## Modifications to original implementation :
- Uses 'conv5_2' output to measure content loss.
Original paper utilizes 'conv4_2' output

- Initial image used for image is the base image (instead of random noise image)
This method tends to create better output images, however parameters have to be well tuned.
Therefore their is a argument 'init_image' which can take the options 'content' or 'noise'


- Uses AveragePooling2D inplace of MaxPooling2D layers
The original paper uses AveragePooling for better results, but this can be changed to use MaxPooling2D layers via the argument `--pool_type="max"`. By default AveragePooling is used, since if offers smoother images, but MaxPooling applys the style better in some cases (especially when style image is the "Starry Night" by Van Goph.

- Style weight scaling
- Rescaling of image to original dimensions, using lossy upscaling present in scipy.imresize()
- Maintain aspect ratio of intermediate and final stage images, using lossy upscaling

## Windows Helper
It is a C# program written to more easily generate the arguments for the python script Network.py

- Make sure to save the vgg16_weights.h5 weights file in the windows_helper folder.
- Upon first run, it will request the python path. Traverse your directory to locate the python.exe of your choice (Anaconda is tested)

### Benefits 
- Automatically executes the script based on the arguments.
- Easy selection of images (Content, Style, Output Prefix)
- Easy parameter selection
- Easily generate argument list, if command line execution is preferred. 

## Parameters
```
--image_size : Allows to set the Gram Matrix size. Default is 400 x 400, since it produces good results fast. 
--num_iter : Number of iterations. Default is 10. Test the output with 10 iterations, and increase to improve results.
--init_image : Can be "content" or "noise". Default is "content", since it reduces reproduction noise.
--pool_type : Pooling type. AveragePooling ("ave") is default, but smoothens the image too much. For sharper images, use MaxPooling ("max").

--content_weight : Weightage given to content in relation to style. Default if 0.025
--style_weight : Weightage given to style in relation to content. Default is 1. 
--style_scale : Scales the style_weight. Default is 1. 
--total_variation_weight : Regularization factor. Smaller values tend to produce crisp images, but 0 is not useful. Default = 1E-5

--rescale_image : Rescale image to original dimensions after each iteration. (Bilinear upscaling)
--rescale_method : Rescaling algorithm. Default is bilinear. Options are nearest, bilinear, bicubic and cubic.
--maintain_aspect_ratio : Rescale the image just to the original aspect ratio. Size will be (gram_matrix_size, gram_matrix_size * aspect_ratio). Default is True
--content_layer : Selects the content layer. Paper suggests conv4_2, but better results can be obtained from conv5_2. Default is conv5_2.
```

# Examples
<img src="https://raw.githubusercontent.com/titu1994/Neural_Style_Transfer/master/images/inputs/content/blue-moon-lake.jpg" width=45% height=300> <img src="https://raw.githubusercontent.com/titu1994/Neural_Style_Transfer/master/images/inputs/style/starry_night.jpg" width=45% height=300>
<br> Result after 50 iterations (Average Pooling) <br>
<img src="https://raw.githubusercontent.com/titu1994/Neural_Style_Transfer/master/images/output/Blue_Moon_Lake_iteration_50.jpg" width=90% height=450>
<br> For comparison, results after 50 iterations (Max Pooling) <br>
<img src="https://raw.githubusercontent.com/titu1994/Neural_Style_Transfer/master/images/output/Tsukiyomi_at_iteration_100-Max-Pooling.jpg" width=90% height=450>
<br> DeepArt.io result (1000 iterations and using improvements such as Markov Random Field Regularization) <br>
<img src="https://raw.githubusercontent.com/titu1994/Neural_Style_Transfer/master/images/output/DeepArt_Blue_Moon_Lake.png" width=90% height=450>


# Network.py in action
![Alt Text](https://raw.githubusercontent.com/titu1994/Neural-Style-Transfer/master/images/Blue%20Moon%20Lake.gif)

# Requirements 
- Theano
- Keras 
- CUDA (GPU)
- CUDNN (GPU)
- Scipy
- Numpy

# Speed
On a 980M GPU, the time required for each epoch depends on mainly image size (gram matrix size) :

For a 400x400 gram matrix, each epoch takes approximately 11-13 seconds. <br>
For a 512x512 gram matrix, each epoch takes approximately 18-22 seconds. <br>
For a 600x600 gram matrix, each epoch takes approximately 28-30 seconds. <br>
  
# Issues
- Due to usage of content image as initial image, output depends heavily on parameter tuning. <br> Test to see if the image is appropriate in the first 10 epochs, and if it is correct, increase the number of iterations to smoothen and improve the quality of the output.
- Generated image is seen to be visually better if a small image size (small gram matrix size) is used.
- Due to small gram sizes, the output image is usually small. 
<br> To correct this, use the implementations of this paper "Image Super-Resolution Using Deep Convolutional Networks" http://arxiv.org/abs/1501.00092 to upscale the images with minimal loss.
<br> Some implementations of the above paper for Windows : https://github.com/tanakamura/waifu2x-converter-cpp <br>
- Implementation of Markov Random Field Regularization and Patch Match algorithm are currently being tested. MRFNetwork.py contains the basic code, which need to be integrated to use MRF and Patch Match as in Image Analogies paper <a href="http://arxiv.org/abs/1601.04589"> Combining Markov Random Fields and Convolutional Neural Networks for Image Synthesis </a>

