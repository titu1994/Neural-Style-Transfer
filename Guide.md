# Guide To Get Good Results

There are various parameters in both Network.py and INetwork.py scripts that can be modified to achieve different results.

<b>Note: </b> As of Keras 1.1.0, Tensorflow is the default backend for Keras. However, if you are on Windows, Tensorflow is not available. Therefore, Windows users should go to their C:/Users/{UserName}/.keras directory and configure their keras.json file as below:

```
{
    "image_dim_ordering": "th",
    "floatx": "float32",
    "backend": "theano",
    "epsilon": 1e-07
}
```

# Acknowledgements

Uses the VGG-16 model as described in the Keras example below :
https://github.com/fchollet/keras/blob/master/examples/neural_style_transfer.py

Uses weights from Keras Deep Learning Models : https://github.com/fchollet/deep-learning-models

Neural Doodles is a modification of the example script available at Keras : https://github.com/fchollet/keras/blob/master/examples/neural_doodle.py

## General Tips
- Max number of epochs should be around 100. The original paper suggests 1000 epochs, but this script provides a very good result much faster.
- Pooling type matters a lot. Generally, I find that using Max Pooling delivers better results, even though the paper suggests to use Average Pooling
- Large image sizes require a lot more time per epoch, and very large images may not even fit in GPU memory causing the script to crash.
- Use of Convolution 5_2 layer as content layer is highly recommended, since it delivers far better results.
- Due to use of Convolution 5_2 layer as content layer, the ratio of content weight : style weight needs to be carefully decided. I have found that changing the content : style weight ratio to 1 : 0.1 or 1 : 0.05 or sometimes 1 : 0.01 is very effective in styling the image without destroying the content.
- Always use init_image as "content" and not "noise". "noise" will produce a very grainy image
- Style scale simply multiplies the scale with the style weight. Keeping it constant at 1 and modifying the style weight is sufficient in achieving good results.

## Examples 
A folder with several example stylized images as  well as some style images which are useful to produce images in less than 10 epochs : <a href="https://goo.gl/photos/joqqjcquRMu8uuZp6">Google Photos - Neural Style Transfer Folder </a>

<a href="https://goo.gl/photos/joqqjcquRMu8uuZp6"><img src="https://lh3.googleusercontent.com/pyjUXvpjoUn8bMmOgiPYt6UBnyMLduJ_H2lDEA2pBqAhPRo2WtargtJZf7AS_2BkD4u1_o84POAnoiIGekD7JeIZbyWjEvmmcEUPiisDdqYHRuumyfPwbjNJMIRBSkCVOa8e2Zevici-pDyO2Pojx62tIYrTDhSzZkQUAHZV8URVDAVZGKNf4OMP9VDimaZoYkmZEWr3LrL3347v1gOcYWOpOX5XthXBXFpxf33aFNoCUFP7pRKflsic2_tXccQ2A9HkUzN_OkLxfgM1vJcuiszvrn2HFtmJE_gLr6WZlbgIjT5hI-MzmcyGWtbK1SN_g-988499lC-ojj5ZEnmLRfXhCl2oYcLzqjmRyQGDdeu0OFhc4EdztN5Ral5N_2OxanK2uXYsE1ZwCBumAsd01UCw_DdPc-LKbZs4noP5GLskcr2KshO8ydIXpM9X_OZIv41AalePJB06a1Oiq2lQsoT6HpIqpGQZ_OCuWXQBM5yRO-gY2EWwO5xlK5ilqXbC9IK2T-RVoc1ug8PMRiH_jAcRPGLVSiT70nz3f-taJRv4FeNaxzKEvEktB8gn2Q28nHyWoV7PBiGihYBvi14c38l5wiR8xFcRxKTRWQi01FHLTEfiCA=w711-h400-no" width=90%></a>

<a href="https://goo.gl/photos/joqqjcquRMu8uuZp6"><img src="https://lh3.googleusercontent.com/3qmUFEHa74d0qhtsbwxhHnbJNxAqgZAOHdhKTggbrsQ_xSyeKhZAtmccyUEM1R-BDwXX8rMOnnNGtxej-DWYgvg6_qN18PR5i8lZBKrRgsKQwIvvZ9dNHAJ5jU9RY6bMpUU3GXXwff_db2D6yqA6csrzd_TXV_zGFwhAAWYGDWYm3vf2GkwUOhuV_Pw8eXskOPKpChZHNHkBC4GAhrgFMtu6uodaYdyQg5uLgT-UU-iNWURPgfzeNn9yWpH4HbFnAeYd2qStxW3Z1aobRBcgurdE_KsODYxPQnA2vyBBAni6Gq3G9AEZChgmNWB1OM2jZX3YN9JDbC_kza5JPV04ICEm3cwco2GgGScg6Jjy_1vzxusl-u4HfqzyKVAe7a44mrDd-FWhpeu9C7GFI_yNlS5bTH5vsOMDY76Bta7_j4l9sXEzy89ZQTR_2IWBvh-3a_UIxeTnlpWgwdYDXweuC9LQVnr-gspFHVnQl6de5OP0lotjAJZ9_GYVEIrHEy6YBRZ-urciffEuHBCtzjJpdapFdMI-uct77uIEL--vQ31WTZ9y6tLa8EFQpc66Rmx5fP3Z2C2kj_1dma79jBGnVff0WiIUq0bDKvvw1F-Sy69pz7Dy3w=w600-h400-no" width=90%></a>

Example of various styles (with and without color preservation). Images of the "Lost Grounds" from .Hack G.U.<br>
<img src="https://github.com/titu1994/Neural-Style-Transfer/blob/master/images/output/Lost-Grounds.jpg?raw=true" width=90%>


## Tips for Total Variation Regularization
Total Variation Weight has a subtle but important role to play. The implementation in keras examples states to use tv_weight as 1, but I found that the images are smoothed to an extreme degree, and the results are not appealing. After several tests, I have found a few values which are very suitable to certain cases :

1) If the content and style have similar colour schemes, use tv_weight = 1E-5<br>
2) If the content and style have at least one major color similar, use tv_weight = 5E-05<br>
3) If the content and style have the same background color, use tv_weight = 8E-05<br>
4) If the content and style do not share same color palette at all, use tv_weight = 5E-05<br>
5) If you want relatively crisp images without worrying about color similarity, use tv_weight = 8.5E-05. It works well almost 90 % of the time.<br>
6) If style image is "The Starry Night", use tv_weight = 1E-04 or 1E-05. Other values produce distortions and unpleasant visual artifacts in most cases.<br>
7) I have tried turing off tv_weight to 0, however the results were very poor. Image produced is sharp, but lacks continuity and is not visually pleasing. If you want very crisp images at the cost of some continuity in the final image, use tv_weight = 5E-08.<br>

## Improvements in INetwork.py 

The following improvements from the paper <a href="http://arxiv.org/abs/1605.04603">Improving the Neural Algorithm of Artistic Style</a> have been implemented in INetwork.py :

- Improvement 3.1 in paper : Geometric Layer weight adjustment for Style inference
- Improvement 3.2 in paper : Using all layers of VGG-16 for style inference
- Improvement 3.3 in paper : Activation Shift of gram matrix
- Improvement 3.5 in paper : Correlation Chain

## Script Helper Program

Bundled together with the script is the script_helper directory, which contains a C# Forms application that allows for rapid testing of the script. It provides a quick way to launch the script or copy the arguments to the clipboard for use in the command line.

Benefits

- Automatically executes the script based on the arguments.
- Easy selection of images (Content, Style, Output Prefix)
- Easy parameter selection
- Easily generate argument list, if command line execution is preferred.
- Works on Linux using Mono

### Setting Up Script Helper on Windows
Setting up the script helper on Windows is extremely easy. Minimum requirements are .NET 4.5 which is preinstalled on all Windows 7 and above.

Steps:
- Follow below steps to setup Anaconda and Theano on Windows
- Download the latest Script Helper release from https://github.com/titu1994/Neural-Style-Transfer/releases
- Extract to some location
- Run "Neural Style Transfer.exe" program and set the content, style, output prefix and mask if necessary
- Set other parameters as needed. Note that the default parameters are generally the best for most styles.
- Note : The first time you execute the script, it will open a pop up asking for the location of Anaconda (Python) with all the necessary packages installed. 

### Setting Up Script Helper on Linux
Setting up script helper on Linux is a 2 step process, but still very easy.

Steps:
- Download and install `mono-complete` for your linux distro. For Ubuntu (tested) : `sudo apt-get mono-complete`
- Follow below steps to setup Anaconda and Theano on Linux
- Follow remaining steps for Windows Script Helper (above). It's that easy.

## Need for cuDNN

Both scripts utilize the VGG-16 network, which consists of stacks of Convolutional Layers. Since VGG-16 is such a dense network, it is advisable to run this script only on GPU. Even on GPU, if the INetwork.py script is to be used, then it is absolutely recommended to install cuDNN 5.1 as well. 

This is because cuDNN 4+ has special support for VGG networks, without which the INetwork script will require several hundred seconds even on GPU. INetwork utilizes all VGG layers in orfer to compute style loss, and even uses Chained Inference between adjacent layers thus requiring a vast amount of time without cuDNN.

# Setting Up Theano for GPU (on Windows)

Setting up Theano for GPU compute on Windows is a huge undertaking, with large number of extra files that need to be installed. Be prepared cause some files may take hours to download, so have some coffee prepared. 

## Steps

We begin with the downloads, then we will set them all up in one go.

- First and foremost, install <a href="https://beta.visualstudio.com/downloads/">Microsoft Visual Studio 2013</a>. Be sure to download the ISO version, simply because if you switch laptops, at least you wont have to download this again.
- Next, download <a href="https://developer.nvidia.com/cuda-downloads">CUDA 7.5</a>. 
- Then we move onto cuDNN. This will require an NVidia Dev account, for which you will have to register and wait to be authorized. After you are authorized, you can log in and go to the <a href="https://developer.nvidia.com/cudnn">downloads section</a>. Fill up the survey, and then continue onto downloading cuDNN.
- Install <a href="https://repo.continuum.io/archive/index.html">Anaconda 2.2.0</a> (Its called <b>Anaconda3-2.2.0-Windows-x86_64.exe</b>). I know that Anaconda 4.1.1+ has already come out, but its fatal problem is its support for Python 3.5, which Theano does <b>NOT</b> currently support. This is a slightly convoluted, but still easier way to get Python 3.4 stack with Anaconda. (Another slightly less convoluted way is installing the latest Anaconda, force downgrade python to 3.4.4 using `conda install python==3.4.4` and then downgrading all the packages with `conda update --all`. This will reinstall the entire conda stack with 3.4.4 packages. Bummer. Or just go with Miniconda and downgrade to python 3.4.4 and `conda install` all the required packages as required)

The next few are optional, but still recommended if you want Theano to run fast on CPU as well:
- Download OpenBLAS Windows binaries from <a href="https://sourceforge.net/projects/openblas/files/v0.2.14/">here</a>. I downloaded the  OpenBLAS-v0.2.14-Win64-int32.zip. version.
- Download the mingw64_dll.zip from <a href="http://sourceforge.net/projects/openblas/files/v0.2.14/mingw64_dll.zip/download">here</a>
- Install TDM GCC 64 from <a href="http://tdm-gcc.tdragon.net/">here</a>

We are now down with the downloads. Onto actually installing all these things:
- MS VS 2013 should auto install when downloaded.
- Next, install CUDA. 
- Now, extract cuDNN into a folder. Inside it should be 3 sub folders which must be copied inside the CUDA directory. It should overwrite several files. If it doesn't overwrite the files, it's in the wrong place.
- Next install Anaconda 2.2.0, and add it as default python interpreter. This adds python and conda commands to path automatically.
- Next, extract OpenBLAS into C: drive. Also extract the mingw64 dlls into the bin folder.
- Lastly, extract TDM-GCC into a seperate folder in C: drive

Now, we have to setup some environment variables. To do so, go to Control Panal -> System and Security -> System -> "Advanced System Settings" on the left -> Advanced tab -> Environment Variables
- Add system variable CUDA_PATH = C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5
- Add system variable CUDNN_PATH = C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5
- Add system variable VS120COMNTOOLS = C:\Program Files (x86)\Microsoft Visual Studio 12.0\Common7\Tools\ # See if this exists first, but it should.

Next, we will add a few things to the PATH variable. In Windows 10, you can easily add new paths using the buttons, but for older Windows versions, append the path to the variable and end with a semicolon. <b>Note that the order must be exactly similar to this, cause if you have MINGW or CYGWIN installed seperately then TDM-GCC has to come before them in the path string. </b>

- C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\bin
- C:\TDM-GCC-64\bin 
- Optional (if already installed): C:\MinGW\bin
- C:\openblas\bin # Assuming path of openblas is in C: drive in folder openblas

Aaand you're almost done!. A few last steps to take, so open a command prompt (Win + R, cmd.exe):
- `conda update --all`
- `conda install mingw libpython` # Two modules which are absolutely needed, else you will see raw code when running without a doubt.
- `conda install h5py` # Ideally comes installed with Keras, but just to be sure.
- `pip install pillow` # Dependency for scipy.misc (all image related stuff)

Next go to this <a href="http://www.lfd.uci.edu/~gohlke/pythonlibs/">site</a>, which contains pre built windows binaries for several python libraries which are used in Python Scientific Stack.
- Download these files : scipy, numpy, statsmodels
- Open command prompt at location where these files are located (Deselect everything, then Shift + Right click empty space, there will be a new version called "Open Command windows here")
- Use : `pip install X.whl`, replacing X with the full name of each of the downloaded files. Note that the .whl extention is necessary.
- Finally install the all important libraries : `pip install theano keras`

There is one last step, and that is setting up the all important .theanorc.txt file in the user directory. 
The path will be <b>C:\Users\YOUR_USERNAME\\.theanorc.txt </b> Create a file with the name ".theanorc.txt" in this path.

It's contents need to be : 
```
[global]
device=gpu 
floatX=float32
optimizer=fast_run
optimizer_including=cudnn
allow_gc=True

[cuda]
root = C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5

[nvcc]
fastmath = True
compiler_bindir=C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\bin

[blas]
ldflags= -LC:\openblas\bin -lopenblas

[lib]
cnmem=0.8
```

# Setting up Theano on Linux (100x Simpler than on Windows, the Windows Helper program runs with mono)

These steps should work for most versions of Ubuntu (14.04 / 15.04 / 16.04):
- Install Anaconda2, whatever version is most recent and add to path as default python interpreter. Drawback is you have to program in python 2.7. Otherwise use steps as above to downgrade Anaconda 4.1+ with 3.5 to python 3.4 and then update all packages, or install <a href="https://repo.continuum.io/archive/index.html">Anaconda 2.2.0</a> (linux version) directly and update all packages.
- Install CUDA
- Install cuDNN for CUDA using steps as above
- Add CUDA path to PATH variable (either in ~/.bashrc or ~/.profile)
- `pip install theano keras` That's it!

Finally setup a minimal ~/.theanorc file with the following content :
```
[global]
device=gpu
floatX=float32
optimizer=fast_run
optimizer_including=cudnn

[lib]
cnmem=0.8
```





