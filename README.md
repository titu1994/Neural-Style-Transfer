# Neural Style Transfer & Neural Doodles
Implementation of Neural Style Transfer from the paper <a href="http://arxiv.org/abs/1508.06576">A Neural Algorithm of Artistic Style</a> in Keras 2.0+

INetwork implements and focuses on certain improvements suggested in <a href="http://arxiv.org/abs/1605.04603">Improving the Neural Algorithm of Artistic Style</a>. 

Color Preservation is based on the paper [Preserving Color in Neural Artistic Style Transfer](https://arxiv.org/abs/1606.05897).

Masked Style Transfer is based on the paper [Show, Divide and Neural: Weighted Style Transfer](http://cs231n.stanford.edu/reports/2016/pdfs/208_Report.pdf)

## Colaboratory Support

[This codebase can now be run directly from colaboratory using the following link](https://colab.research.google.com/github/titu1994/Neural-Style-Transfer/blob/master/NeuralStyleTransfer.ipynb), or by opening `NeuralStyleTransfer.ipynb` and visiting the Colab link.

Colab link supports almost all of the additional arguments, except of the masking ones. They will probably be added at a later date.

**NOTE :** Make sure you use a GPU in Colab or else the notebook will fail. To change Runtimes : `Runtime -> Change Runtime type ->`. Here select Python 3 and GPU as the hardware accelerator. 

## Guide

See the <a href="https://github.com/titu1994/Neural-Style-Transfer/blob/master/Guide.md">guide</a> for details regarding how to use the script to achieve the best results

It also explains how to setup Theano (with GPU support) on both Windows and Linux. Theano on Windows is a long and tedious process, so the guide can speed up the process by simply letting you finish all the steps in the correct order, so as not to screw up the finicky Theano + Windows setup.

The **Script Helper** program can be downloaded from the Releases tab of this repository, [Script Helper Releases](https://github.com/titu1994/Neural-Style-Transfer/releases). Extract it into any folder and run the `Neural Style Transfer.exe` program. On Linux, you will need to install Mono C# to run the script helper program.

# Examples
## Single Style Transfer
<img src="https://raw.githubusercontent.com/titu1994/Neural_Style_Transfer/master/images/inputs/content/blue-moon-lake.jpg" width=49% height=300 alt="blue moon lake"> <img src="https://raw.githubusercontent.com/titu1994/Neural_Style_Transfer/master/images/inputs/style/starry_night.jpg" width=49% height=300 alt="starry night">
<br><br> Results after 100 iterations using the INetwork<br>
<img src="https://github.com/titu1994/Neural-Style-Transfer/blob/master/images/output/Blue-Moon-Lake_at_iteration_100.jpg?raw=true" width=98% height=450 alt="blue moon lake style transfer">
<br><br> DeepArt.io result (1000 iterations and using improvements such as Markov Random Field Regularization) <br>
<img src="https://raw.githubusercontent.com/titu1994/Neural_Style_Transfer/master/images/output/DeepArt_Blue_Moon_Lake.jpg" width=98% height=450>

## Style Transfer with Color Preservation
An example of color preservation with Kinkaku-ji, a Buddhist temple, as the content image and Monet's "Water Lilies" as the art style: <br><br>
<img src="https://github.com/titu1994/Neural-Style-Transfer/blob/master/images/inputs/content/Kinkaku-ji.jpg?raw=true" height=300 width=49%> <img src="https://github.com/titu1994/Neural-Style-Transfer/blob/master/images/inputs/style/water-lilies-1919-2.jpg?raw=true" height=300 width=49%> <br>
<img src="https://github.com/titu1994/Neural-Style-Transfer/blob/master/images/output/Jukai_color_preservation.jpg?raw=true" height=300 width=49% alt="Kinkaku color preservation"> <img src="https://github.com/titu1994/Neural-Style-Transfer/blob/master/images/output/Jukai.jpg?raw=true" width=49% height=300 alt="kinkaku style transfer">
<br><br> As an example, here are two images of the Sagano Bamboo Forest with the "pattened-leaf" style, with and without color preservation <br>
<img src="https://github.com/titu1994/Neural-Style-Transfer/blob/master/images/inputs/content/sagano_bamboo_forest.jpg?raw=true" height=450 width=49%> <img src="https://github.com/titu1994/Neural-Style-Transfer/blob/master/images/inputs/style/patterned_leaves.jpg?raw=true" height=450 width=49%>
<br>
<img src="https://github.com/titu1994/Neural-Style-Transfer/blob/master/images/output/Bamboo-Fores.jpg?raw=true" height=450 width=49% alt="sagano bamboo forest style transfer color preservation"> <img src="https://github.com/titu1994/Neural-Style-Transfer/blob/master/images/output/Bamboo-Forest-No-Color-Preservation.jpg?raw=true" height=450 width=49% alt="sagano bamboo forest style transfer"> <br><br>

Color preservation can also be done using a mask. Using the `color_transfer.py` script and supplying a mask image, in which white regions will allow the content's colors to be transfered and black regions will keep the style-generated colors.

Below, the content image is "Sunlit Mountain", with the style image as "Seated Nude" by Picasso. Notice that the color preservation mask ensures that color transfer occurs only for the sky region, while the mountains are untouched.
<img src="https://github.com/titu1994/Neural-Style-Transfer/blob/master/images/inputs/content/Sunlit%20Mountains.jpg?raw=true" height=300 width=33%> <img src="https://github.com/titu1994/Neural-Style-Transfer/blob/master/images/inputs/mask/Sunlit%20Mountains%20Color%20Mask.jpg?raw=true" height=300 width=33%> <img src="https://github.com/titu1994/Neural-Style-Transfer/blob/master/images/inputs/style/seated-nude.jpg?raw=true" height=300 width=32%>

<img src="https://github.com/titu1994/Neural-Style-Transfer/blob/master/images/output/Sunlit-Mountain.jpg?raw=true" height=300 width=49%> <img src="https://github.com/titu1994/Neural-Style-Transfer/blob/master/images/output/Sunlit-Mountain_color_preservation.jpg?raw=true" height=300 width=50%>

## Style Interpolation
Style weight and Content weight can be manipulated to get drastically different results.

Leonid Afremov's "Misty Mood" is the style image and "Dipping Sun" is the content image : <br>
<img src="https://github.com/titu1994/Neural-Style-Transfer/blob/master/images/inputs/content/Dipping-Sun.jpg?raw=true" height=300 width=49%> <img src="https://github.com/titu1994/Neural-Style-Transfer/blob/master/images/inputs/style/misty-mood-leonid-afremov.jpg?raw=true" height=300 width=50%> 

<table>
<tr align='center'>
<td>Style=1, Content=1000</td>
<td>Style=1, Content=1</td>
<td>Style=1000, Content=1</td>
</tr>
<tr>
<td><img src="https://github.com/titu1994/Neural-Style-Transfer/blob/master/images/output/DippingSun3.jpg?raw=true" height=300></td>
<td><img src="https://github.com/titu1994/Neural-Style-Transfer/blob/master/images/output/DippingSun2.jpg?raw=true" height=300></td>
<td><img src="https://github.com/titu1994/Neural-Style-Transfer/blob/master/images/output/DippingSun1.jpg?raw=true" height=300></td>
</tr>
</table>


## Multiple Style Transfer
The next few images use the Blue Moon Lake as a content image and Vincent Van Gogh's "Starry Night" and Georgia O'Keeffe's "Red Canna" as the style images: <br>
<img src="https://raw.githubusercontent.com/titu1994/Neural_Style_Transfer/master/images/inputs/style/starry_night.jpg" width=49% height=300> <img src="https://github.com/titu1994/Neural-Style-Transfer/blob/master/images/inputs/style/red-canna.jpg?raw=true" height=300 width=49%>

The below are the results after 50 iterations using 3 different style weights : <br>
<table align='center'>
<tr align='center'>
<td>Starry Night : 1.0, Red Canna 0.2</td>
<td>Starry Night : 1.0, Red Canna 0.4</td>
<td>Starry Night : 1.0, Red Canna 1.0</td>
</tr>
<tr>
<td><img src="https://github.com/titu1994/Neural-Style-Transfer/blob/master/images/output/blue_moon_lake_1-0_2.jpg?raw=true" height=300></td>
<td><img src="https://github.com/titu1994/Neural-Style-Transfer/blob/master/images/output/blue_moon_lake_1-0_4.jpg?raw=true" height=300></td>
<td><img src="https://github.com/titu1994/Neural-Style-Transfer/blob/master/images/output/blue_moon_lake_1-1_at_iteration_50.jpg?raw=true" height=300></td>
</tr>
</table>

## Masked Style Transfer
Supplying an additional binary mask for each style, we can apply the style to a selected region and preserve the content in other regions.We can also use multiple masks to apply 2 different styles in 2 different regions of the same content image.

Note that with the `mask_transfer.py` script, a single content image can be masked with 1 mask to preserve content in blackend regions and preserve style transfer in whitened regions in the generated image. Currently, only content can be transfered in a post processed manner.

"The Starry Night" is used as the style image in the below images. The mask tries to preserve the woman's shape and color, while applying the style to all other regions. Results are very good, as "The Starry Night" has a tendency to overpower the content shape and color. <br>
<img src="https://github.com/titu1994/Neural-Style-Transfer/blob/master/images/inputs/content/Dawn%20Sky.jpg?raw=true" height=300 width=50% alt="dawn sky anime"> <img src="https://raw.githubusercontent.com/titu1994/Neural_Style_Transfer/master/images/inputs/style/starry_night.jpg" height=300 width=49%>

<img src="https://github.com/titu1994/Neural-Style-Transfer/blob/master/images/inputs/mask/Dawn-Sky-Mask.jpg?raw=true" height=300 width=50%> <img src="https://github.com/titu1994/Neural-Style-Transfer/blob/master/images/output/Dawn_Sky_masked.jpg?raw=true" height=300 width=49% alt="dawn sky style transfer anime">

<br>
Another example of masked style transfer is provided below. "Winter Wolf" is used as the content image and "Bamboo Forest" is used as the style image. The mask attempts to preserve the darkened cloudy sky, and apply the style only to the mountains and the wolf itself.

<img src="https://github.com/titu1994/Neural-Style-Transfer/blob/master/images/inputs/content/winter-wolf.jpg?raw=true" height=300 width=50%> <img src="https://github.com/titu1994/Neural-Style-Transfer/blob/master/images/inputs/style/bamboo_forest.jpg?raw=true" height=300 width=49%>

<img src="https://github.com/titu1994/Neural-Style-Transfer/blob/master/images/inputs/mask/winter-wolf-mask.jpg?raw=true" height=300 width=50%> <img src="https://github.com/titu1994/Neural-Style-Transfer/blob/master/images/output/WinterWolf-Masked.jpg?raw=true" height=300 width=49% alt="winter wolf style transfer">

<br>
These last few images use "Cherry Blossoms" as the content image, and uses two styles : "Candy Style" and Monet's "Water Lillies" using their respective masks to create an image with unique results. <br>

<img src="https://github.com/titu1994/Neural-Style-Transfer/blob/master/images/inputs/style/candy-style.jpg?raw=true" height=300 width=33%> <img src="https://github.com/titu1994/Neural-Style-Transfer/blob/master/images/inputs/style/water-lilies-1919-2.jpg?raw=true" height=300 width=33%> <img src="https://github.com/titu1994/Neural-Style-Transfer/blob/master/images/inputs/content/Japanese-cherry-widescreen-wallpaper-Picture-1366x768.jpg?raw=true" height=300 width=32%> 

<img src="https://github.com/titu1994/Neural-Style-Transfer/blob/master/images/inputs/mask/cherry-blossom-1.jpg?raw=true" height=300 width=33%> <img src="https://github.com/titu1994/Neural-Style-Transfer/blob/master/images/inputs/mask/cherry-blossom-2.jpg?raw=true" height=300 width=33%> <img src="https://github.com/titu1994/Neural-Style-Transfer/blob/master/images/output/Cherry-Blossoms.jpg?raw=true" height=300 width=32%>

### Silhouette Transfer
Using Masked Transfer, one can post process image silhouettes to generate from scratch artwork that is sharp, clear and manipulates the style to conform to the shape of the silhouette itself.

First we discuss the use of a silhouette of the content vs the content image itself. A silhouette offers a chance to generate new artwork in the artistic vein of the style, while conforming only to the shape of the content, and disregarding the content itself. Combined with post process masking, it is easy to generate artwork similar to the style image itself.

For this image, Starry Night was used as the Style Image.

<table align='center'>
<td>Content</td>
<td>Mask</td>
<td>Generated</td>
</tr>
<tr>
<td><img src="https://github.com/titu1994/Neural-Style-Transfer/blob/master/images/inputs/mask/Fai%20D%20Flowrite%20-%20Ring.jpg?raw=true" height=300></td>
<td><img src="https://github.com/titu1994/Neural-Style-Transfer/blob/master/images/inputs/mask/Fai%20D%20Flowrite%20-%20Ring%20-%20Inv.jpg?raw=true" height=300></td>
<td><img src="https://github.com/titu1994/Neural-Style-Transfer/blob/master/images/output/Fai-Silhuete.jpg?raw=true" height=300></td>
</tr>
</table>

For this example, we use "Blue Strokes" as the style image

<table align='center'>
<td>Content</td>
<td>Style</td>
</tr>
<tr>
<td><img src="https://github.com/titu1994/Neural-Style-Transfer/blob/master/images/inputs/mask/Sakura%20no%20Tsubasa.png?raw=true" height=300></td>
<td><img src="https://github.com/titu1994/Neural-Style-Transfer/blob/master/images/inputs/style/Blue%20Strokes.jpg?raw=true" height=300></td>
</tr>
<tr>
<td><img src="https://github.com/titu1994/Neural-Style-Transfer/blob/master/images/output/Wings-Silhuete.jpg?raw=true" height=300></td>
<td><img src="https://github.com/titu1994/Neural-Style-Transfer/blob/master/images/output/Wings-Silhuete%202.jpg?raw=true" height=300></td>
</tr>
</table>

## Texture Transfer
Utilizing a style image with a very distinctive texture, we can apply this texture to the content without any alterating in the algorithm. It is to be noted that the style image must possess a very strong texture to transfer correctly.

The below is an example of the content image "Aurea Luna", with the texture images which are available in the /style/metals directory, which are Silver and Gold. Color Preservation is applied to both images, and a mask is applied on the "Burnt Gold" image to style just the circle and not the entire square image.

<img src="https://github.com/titu1994/Neural-Style-Transfer/blob/master/images/inputs/content/Aurea-Luna.jpg?raw=true" width=33% alt="aurea luna golden moon clow reed"> <img src="https://github.com/titu1994/Neural-Style-Transfer/blob/master/images/inputs/style/metals/silver_plate.jpg?raw=true" width=33%> <img src="https://github.com/titu1994/Neural-Style-Transfer/blob/master/images/inputs/style/metals/burnt_gold.jpg?raw=true" width=33%>

<img src="https://github.com/titu1994/Neural-Style-Transfer/blob/master/images/output/molten-silver.jpg?raw=true" width=50% alt="molten silver moon"> <img src="https://github.com/titu1994/Neural-Style-Transfer/blob/master/images/output/burnt-gold.jpg?raw=true" width=49% alt="burnt gold moon">

## All Transfer Techniques
Each of these techniques can be used together, or in stages to generate stunning images. 

In the folowing image, I have used Masked style transfer in a multi scale style transfer technique - with scales of 192x192, 384x384, 768x768, applied a super resolution algorithm (4x and then downscaled to 1920x1080), applied color transfer and mask transfer again to sharpen the edges, used a simple sharpening algorithm and then finally denoise algorithm.

<img src="https://github.com/titu1994/Neural-Style-Transfer/blob/master/images/inputs/content/ancient_city.jpg?raw=true" width=33% alt="ancient city japanese" height=250> <img src="https://github.com/titu1994/Neural-Style-Transfer/blob/master/images/inputs/style/blue_swirls.jpg?raw=true" width=33% height=250> <img src="https://github.com/titu1994/Neural-Style-Transfer/blob/master/images/inputs/mask/ancient-city.jpg?raw=true" width=33% height=250> 

Result : <br>
<img src="https://github.com/titu1994/Neural-Style-Transfer/blob/master/images/output/ancient_city_multiscale.jpg?raw=true" width=99% alt="ancient city japanese">

## Various results with / without Color Preservation
Example of various styles (with and without color preservation). Images of the "Lost Grounds" from .Hack G.U.<br>
<img src="https://github.com/titu1994/Neural-Style-Transfer/blob/master/images/output/Lost-Grounds.jpg?raw=true" width=98%>

# Neural Doodle Examples
Renoit Style + Content Image <br>
<img src="https://github.com/titu1994/Neural-Style-Transfer/blob/master/images/neural_doodle/generated/renoit_new.png?raw=true" width=98%><br>
Monet Style + Doodle Creation <br>
<img src="https://github.com/titu1994/Neural-Style-Transfer/blob/master/images/neural_doodle/generated/monet_new.png?raw=true" width=98%>
<br>Van Gogh + Doodle Creation <br>
<img src="https://github.com/titu1994/Neural-Style-Transfer/blob/master/images/neural_doodle/generated/van%20gogh.png?raw=true" width=98%>

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
The original paper uses AveragePooling for better results, but this can be changed to use MaxPooling2D layers via the argument `--pool_type="max"`. By default MaxPooling is used, since if offers sharper images, but AveragePooling applies the style better in some cases (especially when style image is the "Starry Night" by Van Gogh).

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

## Script Helper
It is a C# program written to more easily generate the arguments for the python script Network.py or INetwork.py (Using Neural Style Transfer tab) and neural_doodle.py or improved_neural_doodle.py script (Using Neural Doodle Tab)

<img src="https://github.com/titu1994/Neural-Style-Transfer/blob/master/images/Neural%20Art%20Windows.JPG?raw=true" height=600 width=98%>

- Upon first run, it will request the python path. Traverse your directory to locate the python.exe of your choice (Anaconda is tested)
- The script helper program code is available at: https://github.com/titu1994/Neural-Style-Transfer-Windows The program runs on Linux using Mono

### Benefits 
- Allows Style Transfer, Neural Doodles, Color Transfer and Masked Style Transfer easily
- Automatically executes the script based on the arguments.
- Easy selection of images (Content, Style (Multiple Selection allowed), Output Prefix)
- Easy parameter selection
- Easily generate argument list, if command line execution is preferred. 
- Creates log folders for each execution so settings can be preserved
- Runs on Windows (Native) and Linux (Using Mono)

To use multiple style images, when the image choice window opens, select all style images as needed. Pass multiple style weights by using a space between each style weight in the parameters section.

## Usage
### Neural Style Transfer
Both Network.py and INetwork.py have similar usage styles, and share all parameters.

Network.py / INetwork.py
```
python network.py/inetwork.py "/path/to/content image" "path/to/style image" "result prefix or /path/to/result prefix"
```

To pass multiple style images, after passing the content image path, seperate each style path with a space
```
python inetwork.py "/path/to/content image" "path/to/style image 1" "path/to/style image 2" ... "result prefix or /path/to/result prefix" --style_weight 1.0 1.0 ... 
```

There are various parameters discussed below which can be modified to alter the output image. Note that many parameters require the command to be enclosed in double quotes ( " " ).

Example:
```
python inetwork.py "/path/to/content image" "path/to/style image" "result prefix or /path/to/result prefix" --preserve_color "True" --pool_type "ave" --rescale_method "bicubic" --content_layer "conv4_2"
```

To perform color preservation on an already generated image, use the `color_transform.py` as below. It will save the image in the same folder as the generated image with "_original_color" suffix.
```
python color_transfer.py "path/to/content/image" "path/to/generated/image"
```

A mask can also be supplied to color preservation script, using the `--mask` argument, where the white region signifies that color preservation should be done there, and black regions signify the color should not be preserved here.
```
python color_transfer.py "path/to/content/image" "path/to/generated/image" --mask "/path/to/mask/image"
```

A note on mask images: 
- They should be binary images (only black and white)
- White represents parts of the image that you want style transfer to occur
- Black represents parts of the image that you want to preserve the content
- Be careful of the order in which mask images are presented in Multi Style Multi Mask generation. They have a 1 : 1 mapping between style images and style masks.
- When using the Script Helper program, it may happen that the masks are being ordered incorrectly due to name-wise sorting. Therefore, rename the masks in alphabetic order to correct this flaw.

As a general example, here is the list of parameters to generate a multi style multi mask image:
```
python network.py "Japanese-cherry-widescreen-wallpaper-Picture-1366x768.jpg" "candy-style.jpg" "water-lilies-1919-2.jpg" \
"Cherry Blossom" --style_masks "cherry-blossom-1.jpg" "cherry-blossom-2.jpg" --content_weight 5 --style_weight 1.0 1.0 \
--num_iter 20 --model "vgg16" --content_loss_type 0
```

Like Color Transfer, single mask style transfer can also be applied as a post processing step instead of directly doing so in the style transfer script. You can preserve some portion of the content image in the generated image using the post processing script `mask_transfer.py`.

Example:
```
python mask_transfer.py "path/to/content/image" "path/to/generated/image" "path/to/content/mask"
```

### Neural Doodles
Both the neural_doodle.py and improved_neural_doodle.py script share similar usage styles.

neural_doodle.py & improved_neural_doodle.py
```
python neural_doodle.py --nlabels -style-image --style-mask --target-mask --content-image --target-image-prefix
```
 
Example 1 : Doodle using a style image, style mask and target mask (from keras examples)
```
python neural_doodle.py --nlabels 4 --style-image Monet/style.png \
    --style-mask Monet/style_mask.png --target-mask Monet/target_mask.png \
    --target-image-prefix generated/monet
```

Example 2:  Doodle using a style image, style mask, target mask and an optional content image.
```
 python neural_doodle.py --nlabels 4 --style-image Renoir/style.png \
    --style-mask Renoir/style_mask.png --target-mask Renoir/target_mask.png \
    --content-image Renoir/creek.jpg \
    --target-image-prefix generated/renoir
```

Multiple phases Example : Doodle using a style image, style mask, target mask and using it multiple times to acheive better results.
- Assume that an image has a size (400 x 600). 
- Divide the image size by 4 (100 x 125)
- Create 1st doodle according to the below script #1 (--img_size 100)
- Create 2nd doodle according to the below script #2 (Note that we pass 1st doodle as content image here) (--img_size 200)
- Create 3rd and last doodle acc to below script #3 (Note we pass 2nd doodle as content image here) (Do not put img_size parameter)

```
# Script 1
python improved_neural_doodle.py --nlabels 4 --style-image srcl.jpg --style-mask srcl-m.png --target-mask dst-m.png  --target-image-prefix ./doodle3-100 --num_iter 50 --img_size 100 --min_improvement 5.0

# Script 2
python improved_neural_doodle.py --nlabels 4 --style-image srcl.jpg --style-mask srcl-m.png --target-mask dst-m.png  --target-image-prefix ./doodle3-200 --num_iter 50 --content-image ./doodle3-100_at_iteration_XXXX.png --img_size 200 --min_improvement 2.5

############# Replace XXXX by last iteration number ################

# Script 3 
python improved_neural_doodle.py --nlabels 4 --style-image srcl.jpg --style-mask srcl-m.png --target-mask dst-m.png  --target-image-prefix ./doodle3-500 --num_iter 50 --content-image ./doodle3-200_at_iteration_XXXX.png

############# Replace XXXX by last iteration number ################
```

### Color Transfer (Post Processing)
Color transfer can be performed after the stylized image has already been generated. This can be done via the `color_transfer.py` script or via the Color Transfer tab in the Script Helper. Note that the script will save the image in the same folder as the generated image with "_original_color" suffix.

Example:
```
python color_transfer.py "path/to/content/image" "path/to/generated/image"
```

A mask can also be supplied to color preservation script, using the `--mask` argument, where the white region signifies that color preservation should be done there, and black regions signify the color should not be preserved here.
```
python color_transfer.py "path/to/content/image" "path/to/generated/image" --mask "/path/to/mask/image"
```

Using the `--hist_match` parameter set to 1, it will perform histogram color matching instead of direct color transfer
```
python color_transfer.py "path/to/content/image" "path/to/generated/image" --hist_match 1
```

Please note that for masks for color preservation and for style transfer have different representations. Color preservations will preserve white areas as content colors, and mask transfer will preserve black areas as content image.

### Masked Style Transfer (Post Processing)
If the general requirement is to preserve some portions of the content in the stylized image, then it can simply be done as a post processing step using the `mask_transfer.py` script or the Mask Transfer tab of the Script Helper.

For now, only the content can be preserved (by coloring the area **black** in the mask). To perform multi style multi mask style transfer, you must supply the styles and masks to the neural style script and let it run for several iterations. This cannot be done as a post processing step. 

Example:
```
python mask_transfer.py "path/to/content/image" "path/to/generated/image" "path/to/content/mask"
```

## Parameters (Neural Style)
```
--style_masks : Multiple style masks may be provided for masking certain regions of an image for style transfer. Number of 
  style_weight parameters must match number of style masks.
--color_mask : A single color mask, which defines the region where the color must be preserved. 

--image_size : Allows to set the Gram Matrix size. Default is 400 x 400, since it produces good results fast. 
--num_iter : Number of iterations. Default is 10. Test the output with 10 iterations, and increase to improve results.
--init_image : Can be "content", "noise" or "gray". Default is "content", since it reduces reproduction noise. "gray" is useful when you want only the color of the style to be used in the image.
--pool_type : Pooling type. MaxPooling ("max") is default. For smoother images, use AveragePooling ("ave").

--model : Can be "vgg16" or "vgg19". Changes between use of VGG 16 or VGG 19 model.
--content_loss_type : Can be 0, 1 or 2. 
                      0 does not add any scaling of the loss. 
                      1 = 1 / (2 * sqrt(channels) * sqrt(width * height))
                      2 = 1 / (channels * width * height)
--preserve_color : Preserves the original color space of the content image, while applying only style. Post processing technique on final image, therefore does not harm quality of style.
--min_improvement : Sets the minimum improvement required to continue training. Default is 0.0, indicating no minimum threshold. Advised values are 0.05 or 0.01

--content_weight : Weightage given to content in relation to style. Default if 0.025
--style_weight : Weightage given to style. Default is 1. When using multiple styles, seperate each style weight with a space
--style_scale : Scales the style_weight. Default is 1. 
--total_variation_weight : Regularization factor. Smaller values tend to produce crisp images, but 0 is not useful. Default = 8.5E-5

--rescale_image : Rescale image to original dimensions after each iteration. (Bilinear upscaling)
--rescale_method : Rescaling algorithm. Default is bilinear. Options are nearest, bilinear, bicubic and cubic.
--maintain_aspect_ratio : Rescale the image just to the original aspect ratio. Size will be (gram_matrix_size, gram_matrix_size * aspect_ratio). Default is True
--content_layer : Selects the content layer. Paper suggests conv4_2, but better results can be obtained from conv5_2. Default is conv5_2.
```

## Parameters (Neural Doodle)
```
--nlabels : Number of colors or labels in mask image
--image_size : Allows to set the Gram Matrix size. Default is -1, which means that it uses style image size automatically. 
--num_iter : Number of iterations. Default is 10. Test the output with 10 iterations, and increase to improve results.
--preserve_color : Preserves the original color space of the content image, while applying only style. Post processing technique on final image, therefore does not harm quality of style. Works only when using content image for guided style transfer
--min_improvement : Minimum improvement in percentage required to continue training. Set to 0.0 to disable.

--content_weight : Weightage given to content in relation to style. Default if 0.1
--style_weight : Weightage given to style in relation to content. Default is 1. 
--total_variation_weight : Regularization factor. Smaller values tend to produce crisp images, but 0 is not useful. Default = 8.5E-5
--region_style_weight : Weight for region style regularization. Keep it set to 1.0 unless testing for experimental purposes.
```

## Parameters (Color Transfer)
```
--masks : Optional, performs masked color transfer
--hist_match : Performs histogram color matching if set to 1. Default is 0.
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
- h5py

# Speed
On a 980M GPU, the time required for each epoch depends on mainly image size (gram matrix size) :

For a 400x400 gram matrix, each epoch takes approximately 8-10 seconds. <br>
For a 512x512 gram matrix, each epoch takes approximately 15-18 seconds. <br>
For a 600x600 gram matrix, each epoch takes approximately 24-28 seconds. <br>

For Masked Style Transfer, the speed is now same as if using no mask. This was acheived by preventing gradient computation of the mask multiplied with the style and content features.

For Multiple Style Transfer, INetwork.py requires slightly more time (~2x single style transfer as shown above for 2 styles, ~3x for 3 styles and so on). Results are better with INetwork.py in multiple style transfer.

For Multi Style Multi Mask Style Transfer, the speed is now same as if using multiple styles only. It was acheived by preventing gradient computation of the mask multiplied with the style and content features.

- For multi style multi mask network, Network.py requires roughly 24 (previously 72) seconds per iteration, whereas INetwork.py requires 87 (previously 248) seconds per iteration
  
# Issues
- Due to usage of content image as initial image, output depends heavily on parameter tuning. <br> Test to see if the image is appropriate in the first 10 epochs, and if it is correct, increase the number of iterations to smoothen and improve the quality of the output.
- Due to small gram sizes, the output image is usually small. 
<br> To correct this, use the implementations of this paper "Image Super-Resolution Using Deep Convolutional Networks" http://arxiv.org/abs/1501.00092 to upscale the images with minimal loss.
<br> Some implementations of the above paper for Windows : https://github.com/lltcggie/waifu2x-caffe/releases <br>(Download the waifu2x-caffe.zip and extract, program supports English)
- Implementation of Markov Random Field Regularization and Patch Match algorithm are currently being tested. MRFNetwork.py contains the basic code, which need to be integrated to use MRF and Patch Match as in Image Analogies paper <a href="http://arxiv.org/abs/1601.04589"> Combining Markov Random Fields and Convolutional Neural Networks for Image Synthesis </a>


