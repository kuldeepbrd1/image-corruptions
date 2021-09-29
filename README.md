# Image corruption and augmentations in Python

### Get Started
1. Make a new conda environment 
    ```shell
    conda create -n ip_env python=3.8
    ```

2. Install Wand package for Ubuntu
    ```shell
    sudo apt-get install libmagickwand-dev
    ```

3. Install python packages from requirements.txt
    ```shell
    pip install -r requirements.txt
    ```

### Usage

To use the scripts for augmenting a dataset of images, follow the steps below:

1. Make a JSON for randomization and effects customization. Copy the [`/configs/base.json`](/configs/base.json)
    ```json
    {
    "corruption_methods": {
        "random_erase": false,
        "color_jitter": true,
        "gaussian_noise": true,
        "shot_noise": true,
        "impulse_noise": true,
        "speckle_noise": true,
        "gaussian_blur": true,
        "glass_blur": true,
        "defocus_blur": true,
        "motion_blur": true,
        "zoom_blur": true,
        "fog": true,
        "spatter": true,
        "earth_bg": false
    },

    "ordered_randomization_layers":{
        "background" : {"effects": ["earth_bg"], "p": 0, "level": null},
        "color" : {"effects": ["color_jitter"], "p": 0.3, "level": 1},
        "noise" : {"effects": ["gaussian_noise", "shot_noise", "impulse_noise", "speckle_noise"], "p": 0.4, "level": 1},
        "blur" :  {"effects": ["gaussian_blur", "defocus_blur", "motion_blur", "zoom_blur", "spatter"], "p": 0.3, "level": 1},
        "erase" : {"effects": ["random_erase"], "p": 0, "level": null}
    },
    
    "extensions": {
        "original": "png",
        "background":"png",
        "target": "png"
    },

    "opts":{
        "earth_augmentation":{
            "ann_file": "C:\\Users\\apoca\\Desktop\\data\\Envisat_ICTR\\json_data\\bbox\\train_bbox.json",
            "earth_bg_images_dir": "C:\\Users\\apoca\\Google Drive (k.r.barad@student.tudelft.nl)\\Envisat\\data\\Himawari-8\\01_June_2020_0000_1300"
        },
        "skip_existing_target_images": false 
    }
    }
    ```

    2. Save it in the configs dir. 

    3. To run the augmentations on a set of images use:
        ```shell 
        python augment_image_files.py --src-dir=data/src/ --dst-dir=data/dst/ --config=configs/{YOURCONFIG}.json  
        ```


    ### Explanation of Intesity levels of individual effects
- **Gaussian Noise**:  Gaussian noise is added by perturbing the normalized pixel intensities in the image with pixel intensity drawn from a normal distribution. The severity is specified by the standard deviation of such a normal distribution. The Level-1 standard deviation is set to 0.08 and the Level-2 standard deviation is set to 0.12. 
    ```python
    x = np . array ( img ) / 255.
    corrup_img = np . clip (x + np . random . normal ( size =x. shape , scale = severity ) , 0,1) * 255
    ```

- **Shot Noise**: Shot noise is added with the pixel intensity drawn from a Poisson distribution. The severity is used to specify the variance of the distribution, from which the the noise for that pixel would be generated. The Level-1 severity factor is set to 60 and the Level-2 severity factor is set to 25.
    ```python
    x = np . array ( img ) / 255.
    corrup_img = np . clip ( np . random . poisson (x* severity ) / severity , 0, 1) *255
    ```

- **Impulse Noise** : Impulse noise is added by replacing a proportion of pixels in the image with hot pixels (normalized pixel intensity = 1). The severity specified the proportion of the total image pixels to be replaced. The Level-1 severity factor is set to 0.015 and the Level-2 severity factor is set to 0.06
    ```python 
    x = np . array ( img ) / 255.
    corrup_img = np . clip ( sk . util . random_noise ( x , mode =’s&p ’, amount = c) ,0 , 1) *255
    ```

- **Speckle Noise**: Speckle noise is added by perturbing the image pixels by an amount obtained by multiplying pixel intensities with a random value drawn from a Gaussian distribution. The severity specifies
the standard deviation of the Gaussian distribution. The Level-1 severity factor is set to 0.15 and the Level-2 severity factor is set to 0.2
    ```python
    x = np . array ( img ) / 255.
    corrup_img =( x + x * np . random . normal ( size =x . shape , scale = severity ) , 0, 1) *255
    ```

- **Gaussian Blur**: Gaussian Blur is added by using a convolution operation and modifying the pixel intensity value using a Gaussian kernel. The severity specifies the standard deviation of the Gaussian kernel, which is convolved. The Level-1 severity is set to 1 and the Level-2 severity is set to 2
    ```python
    x = np . array ( img ) / 255.
    corrup_img = skimage . filters . gaussian (x , sigma = severity , multichannel = True ) ,0, 1) * 255
    ```

- **Defocus Blur**: Defocus Blur is added by constructing a kernel that represents an aliasing disk with Gaussian blur. The image is then convolved with the disk kernel. The severity specifies a 2-tuple with the radius of the aliasing disk and the standard deviation of the gaussian blur used for the disk. The Level-1 severity factor is set to (3,0.1) and the Level-2 severity factor is set to (4,0.5)
    ```python
    x = np . array ( img ) / 255.
    kernel = AugmentationHelpers . disk ( radius =c [0] , alias_blur = c [1])
    channels = []
    for d in range (3) :
        channels . append ( cv2 . filter2D (x [: , :, d] , -1 , kernel ))
    channels = np . array ( channels ). transpose ((1 , 2, 0) ) # 3 x224x224 -> 224 x224x3

    corrup_img = np . clip ( channels , 0, 1) * 255
    ```

- **Motion Blur**: Motion Blur is added using the Wand Library 2 . The severity specifies a 2-tuple with the radius of the aliasing disk and the standard deviation of the blur used with the disk. The Level-1 severity factor is set to (7,3) and the Level-2 severity factor is set to (15,5) 

- **Zoom Blur**: Motion Blur is added by adding two overlays of the zooming on an image array. First overlay zooms in by a large zoom factor and the second overlay zooms out by a small zoom factor. The zoomed image pixel values after each zooming operation are determined by spline interpolation. The severity specifies a 2-tuple with the zoom factor for the two operations. The Level-1 severity factor is set to (1.11,0.01) and the Level-2 severity factor is set to (1.16,0.02)

- **Spatter**: Spatter is added by simulating liquid droplets. The description of magnitude is not trivial, and the reader is referred to the original code here 3.