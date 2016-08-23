# Guide To Get Good Results

There are various parameters in both Network.py and INetwork.py scripts that can be modified to achieve different results.

## General Tips
- Max number of epochs should be around 100. The original paper suggests 1000 epochs, but this script provides a very good result much faster.
- Pooling type matters a lot. Generally, I find that using Max Pooling delivers better results, even though the paper suggests to use Average Pooling
- Large image sizes require a lot more time per epoch, and very large images may not even fit in GPU memory causing the script to crash.
- Use of Convolution 5_2 layer as content layer is highly recommended, since it delivers far better results.
- Due to use of Convolution 5_2 layer as content layer, the ratio of content weight : style weight needs to be carefully decided. I have found that changing the content : style weight ratio to 1 : 0.1 or 1 : 0.05 or sometimes 1 : 0.01 is very effective in styling the image without destroying the content.
- Always use init_image as "content" and not "noise". "noise" will produce a very grainy image
- Style scale simply multiplies the scale with the style weight. Keeping it constant at 1 and modifying the style weight is sufficient in achieving good results.

## Tips for Total Variation Regularization
Total Variation Weight has a subtle but important role to play. The implementation in keras examples states to use tv_weight as 1, but I found that the images are smoothed to an extreme degree, and the results are not appealing. After several tests, I have found a few values which are very suitable to certain cases :

1) If the content and style have similar colour schemes, use tv_weight = 1E-03<br>
2) If the content and style have at least one major color similar, use tv_weight = 5E-04<br>
3) If the content and style have the same background color, use tv_weight = 5E-03<br>
4) If the content and style do not share same color palette at all, use tv_weight = 5E-05<br>
5) If you want relatively crisp images without worrying about color similarity, use tv_weight = 1E-05. It works well almost 80 % of the time.<br>
6) If style image is "The Starry Night", use tv_weight = 1E-03 or 1E-05. Other values produce distortions and unpleasant visual artifacts in most cases.<br>
7) I have tried turing off tv_weight to 0, however the results were very poor. Image produced is sharp, but lacks continuity and is not visually pleasing. If you want very crisp images at the cost of some continuity in the final image, use tv_weight = 5E-08.<br>

## Improvements in INetwork.py 

The following improvements from the paper <a href="http://arxiv.org/abs/1605.04603">Improving the Neural Algorithm of Artistic Style</a> have been implemented in INetwork.py :

- Improvement 3.1 in paper : Geometric Layer weight adjustment for Style inference
- Improvement 3.2 in paper : Using all layers of VGG-16 for style inference
- Improvement 3.3 in paper : Activation Shift of gram matrix
- Improvement 3.5 in paper : Correlation Chain

## Windows Helper Program

Bundled together with the script is the windows_helper directory, which contains a Windows Forms application in C# that allows for rapid testing of the script. It provides a quick way to launch the script or copy the arguments to the clipboard for use in the command line.

Benefits

- Automatically executes the script based on the arguments.
- Easy selection of images (Content, Style, Output Prefix)
- Easy parameter selection
- Easily generate argument list, if command line execution is preferred.

## Need for cuDNN

Both scripts utilize the VGG-16 network, which consists of stacks of Convolutional Layers. Since VGG-16 is such a dense network, it is advisable to run this script only on GPU. Even on GPU, if the INetwork.py script is to be used, then it is absolutely recommended to install cuDNN 5.0 as well. 

This is because cuDNN 4+ has special support for VGG networks, without which the INetwork script will require several hundred seconds even on GPU. INetwork utilizes all VGG layers in orfer to compute style loss, and even uses Chained Inference between adjacent layers thus requiring a vast amount of time without cuDNN.

