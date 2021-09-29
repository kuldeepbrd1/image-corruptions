import os
from PIL import Image
import cv2
import os.path
import time
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.utils.data as data
import numpy as np

import skimage as sk
from skimage.filters import gaussian
from io import BytesIO
from wand.image import Image as WandImage
from wand.api import library as wandlibrary
import wand.color as WandColor
import ctypes
from PIL import Image as PILImage
import cv2
from scipy.ndimage import zoom as scizoom
from scipy.ndimage.interpolation import map_coordinates
import warnings

## Adapted from  https://github.com/hendrycks/robustness
# For more info read https://arxiv.org/abs/1903.12261

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']
warnings.simplefilter("ignore", UserWarning)

# Tell Python about the C method
wandlibrary.MagickMotionBlurImage.argtypes = (ctypes.c_void_p,  # wand
                                            ctypes.c_double,  # radius
                                            ctypes.c_double,  # sigma
                                            ctypes.c_double)  # angle

# Extend wand.image.Image class to include method signature
class MotionImage(WandImage):
    def motion_blur(self, radius=0.0, sigma=0.0, angle=0.0):
        wandlibrary.MagickMotionBlurImage(self.wand, radius, sigma, angle)


class FileHelpers:
    def is_image_file(filename):
        """Checks if a file is an image.
        Args:
            filename (string): path to a file
        Returns:
            bool: True if the filename ends with a known image extension
        """
        filename_lower = filename.lower()
        return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)

    def find_subfolders(dir):
        subfolders = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        subfolders.sort()
        subfolder_to_idx = {subfolders[i]: i for i in range(len(subfolders))}
        return subfolders, subfolder_to_idx


    def make_dataset(dir, subfolder_to_idx):
        images = []
        dir = os.path.expanduser(dir)
        if subfolder_to_idx:
            for target in sorted(os.listdir(dir)):
                d = os.path.join(dir, target)
                if not os.path.isdir(d):
                    continue

                for root, _, fnames in sorted(os.walk(d)):
                    for fname in sorted(fnames):
                        if FileHelpers.is_image_file(fname):
                            path = os.path.join(root, fname)
                            item = (path, fname, subfolder_to_idx[target])
                            images.append(item)
        else:
            for root, _, fnames in sorted(os.walk(dir)):
                for fname in sorted(fnames):
                    if FileHelpers.is_image_file(fname):
                        path = os.path.join(root, fname)
                        item = (path, fname,"")
                        images.append(item)


        return images

class AugmentationHelpers:
    # -------------- Helpers ------------------

    def disk(radius, alias_blur=0.1, dtype=np.float32):
        if radius <= 8:
            L = np.arange(-8, 8 + 1)
            ksize = (3, 3)
        else:
            L = np.arange(-radius, radius + 1)
            ksize = (5, 5)
        X, Y = np.meshgrid(L, L)
        aliased_disk = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)
        aliased_disk /= np.sum(aliased_disk)

        # supersample disk to antialias
        return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)

    # Extend wand.image.Image class to include method signature
    class MotionImage(WandImage):
        def motion_blur(self, radius=0.0, sigma=0.0, angle=0.0):
            wandlibrary.MagickMotionBlurImage(self.wand, radius, sigma, angle)

    # modification of https://github.com/FLHerne/mapgen/blob/master/diamondsquare.py
    def plasma_fractal(mapsize=256, wibbledecay=3):
        """
        Generate a heightmap using diamond-square algorithm.
        Return square 2d array, side length 'mapsize', of floats in range 0-255.
        'mapsize' must be a power of two.
        """
        assert (mapsize & (mapsize - 1) == 0)
        maparray = np.empty((mapsize, mapsize), dtype=np.float_)
        maparray[0, 0] = 0
        stepsize = mapsize
        wibble = 100

        def wibbledmean(array):
            return array / 4 + wibble * np.random.uniform(-wibble, wibble, array.shape)

        def fillsquares():
            """For each square of points stepsize apart,
            calculate middle value as mean of points + wibble"""
            cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
            squareaccum = cornerref + np.roll(cornerref, shift=-1, axis=0)
            squareaccum += np.roll(squareaccum, shift=-1, axis=1)
            maparray[stepsize // 2:mapsize:stepsize,
            stepsize // 2:mapsize:stepsize] = wibbledmean(squareaccum)

        def filldiamonds():
            """For each diamond of points stepsize apart,
            calculate middle value as mean of points + wibble"""
            mapsize = maparray.shape[0]
            drgrid = maparray[stepsize // 2:mapsize:stepsize, stepsize // 2:mapsize:stepsize]
            ulgrid = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
            ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
            lulsum = ulgrid + np.roll(ulgrid, -1, axis=1)
            ltsum = ldrsum + lulsum
            maparray[0:mapsize:stepsize, stepsize // 2:mapsize:stepsize] = wibbledmean(ltsum)
            tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
            tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
            ttsum = tdrsum + tulsum
            maparray[stepsize // 2:mapsize:stepsize, 0:mapsize:stepsize] = wibbledmean(ttsum)

        while stepsize >= 2:
            fillsquares()
            filldiamonds()
            stepsize //= 2
            wibble /= wibbledecay

        maparray -= maparray.min()
        return maparray / maparray.max()


    def clipped_zoom(img, zoom_factor):
        h = img.shape[0]
        # ceil crop height(= crop width)
        ch = int(np.ceil(h / zoom_factor))

        top = (h - ch) // 2
        img = scizoom(img[top:top + ch, top:top + ch], (zoom_factor, zoom_factor, 1), order=1)
        # trim off any extra pixels
        trim_top = (img.shape[0] - h) // 2

        return img[trim_top:trim_top + h, trim_top:trim_top + h]


class Augmentations:


    # --------------------Distortions ------------------------
    order = {
        0 : ['random_erase'],
        1 : ['color_jitter'],
        2 : [ 'gaussian_noise', 'shot_noise', 'impulse_noise', 'speckle_noise'],
        3:  ['gaussian_blur', 'defocus_blur', 'motion_blur', 'zoom_blur', 'spatter']
    }

    def get_list():
        l = []
        for key in list(Augmentations.order):
            l += Augmentations.order[key]

        return l

    # Custom - Hendrycks 2019
    def gaussian_noise(x, severity=1):
        c = [.08, .12, 0.18, 0.26, 0.38][severity - 1]

        x = np.array(x) / 255.
        return np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1) * 255


    def shot_noise(x, severity=1):
        c = [60, 25, 12, 5, 3][severity - 1]

        x = np.array(x) / 255.
        return np.clip(np.random.poisson(x * c) / c, 0, 1) * 255


    def impulse_noise(x, severity=1):
        c = [.015, .06, .09, 0.17, 0.27][severity - 1]

        x = sk.util.random_noise(np.array(x) / 255., mode='s&p', amount=c)
        return np.clip(x, 0, 1) * 255


    def speckle_noise(x, severity=1):
        c = [.15, .2, 0.35, 0.45, 0.6][severity - 1]

        x = np.array(x) / 255.
        return np.clip(x + x * np.random.normal(size=x.shape, scale=c), 0, 1) * 255

    def gaussian_blur(x, severity=1):
        c = [1, 2, 3, 4, 6][severity - 1]

        x = gaussian(np.array(x) / 255., sigma=c, multichannel=True)
        return np.clip(x, 0, 1) * 255


    def glass_blur(x, severity=1):
        # sigma, max_delta, iterations
        c = [(0.7, 1, 2), (0.9, 2, 1), (1, 2, 3), (1.1, 3, 2), (1.5, 4, 2)][severity - 1]

        x = np.uint8(gaussian(np.array(x) / 255., sigma=c[0], multichannel=True) * 255)

        # locally shuffle pixels
        for i in range(c[2]):
            for h in range(224 - c[1], c[1], -1):
                for w in range(224 - c[1], c[1], -1):
                    dx, dy = np.random.randint(-c[1], c[1], size=(2,))
                    h_prime, w_prime = h + dy, w + dx
                    # swap
                    x[h, w], x[h_prime, w_prime] = x[h_prime, w_prime], x[h, w]

        return np.clip(gaussian(x / 255., sigma=c[0], multichannel=True), 0, 1) * 255


    def defocus_blur(x, severity=1):
        c = [(3, 0.1), (4, 0.5), (6, 0.5), (8, 0.5), (10, 0.5)][severity - 1]

        x = np.array(x) / 255.
        kernel = AugmentationHelpers.disk(radius=c[0], alias_blur=c[1])

        channels = []
        for d in range(3):
            channels.append(cv2.filter2D(x[:, :, d], -1, kernel))
        channels = np.array(channels).transpose((1, 2, 0))  # 3x224x224 -> 224x224x3

        return np.clip(channels, 0, 1) * 255


    def motion_blur(x, severity=1):
        c = [(7, 3), (15, 5), (15, 8), (15, 12), (20, 15)][severity - 1]

        output = BytesIO()
        x= Image.fromarray(np.uint8(x))
        x.save(output, format='PNG')
        x = MotionImage(blob=output.getvalue())

        x.motion_blur(radius=c[0], sigma=c[1], angle=np.random.uniform(-45, 45))

        x = cv2.imdecode(np.fromstring(x.make_blob(), np.uint8),
                        cv2.IMREAD_UNCHANGED)

        if x.shape != (224, 224):
            return np.clip(x[..., [2, 1, 0]], 0, 255)  # BGR to RGB
        else:  # greyscale to RGB
            return np.clip(np.array([x, x, x]).transpose((1, 2, 0)), 0, 255)


    def zoom_blur(x, severity=1):
        c = [np.arange(1, 1.11, 0.01),
            np.arange(1, 1.16, 0.01),
            np.arange(1, 1.21, 0.02),
            np.arange(1, 1.26, 0.02),
            np.arange(1, 1.31, 0.03)][severity - 1]

        x = (np.array(x) / 255.).astype(np.float32)
        out = np.zeros_like(x)
        for zoom_factor in c:
            out += AugmentationHelpers.clipped_zoom(x, zoom_factor)

        x = (x + out) / (len(c) + 1)
        return np.clip(x, 0, 1) * 255

    def fog(x, severity=1):
        c = [(1.5, 2), (2, 2), (2.5, 1.7), (2.5, 1.5), (3, 1.4)][severity - 1]
        size = x.shape[0]
        x = np.array(x) / 255.
        max_val = x.max()
        x += c[0] * AugmentationHelpers.plasma_fractal(mapsize=size, wibbledecay=c[1])[:size, :size][..., np.newaxis]
        return np.clip(x * max_val / (max_val + c[0]), 0, 1) * 255

    def spatter(x, severity=1):
        c = [(0.65, 0.3, 4, 0.69, 0.6, 0),
            (0.65, 0.3, 3, 0.68, 0.6, 0),
            (0.65, 0.3, 2, 0.68, 0.5, 0),
            (0.65, 0.3, 1, 0.65, 1.5, 1),
            (0.67, 0.4, 1, 0.65, 1.5, 1)][severity - 1]
        x = np.array(x, dtype=np.float32) / 255.

        liquid_layer = np.random.normal(size=x.shape[:2], loc=c[0], scale=c[1])

        liquid_layer = gaussian(liquid_layer, sigma=c[2])
        liquid_layer[liquid_layer < c[3]] = 0
        if c[5] == 0:
            liquid_layer = (liquid_layer * 255).astype(np.uint8)
            dist = 255 - cv2.Canny(liquid_layer, 50, 150)
            dist = cv2.distanceTransform(dist, cv2.DIST_L2, 5)
            _, dist = cv2.threshold(dist, 20, 20, cv2.THRESH_TRUNC)
            dist = cv2.blur(dist, (3, 3)).astype(np.uint8)
            dist = cv2.equalizeHist(dist)
            #     ker = np.array([[-1,-2,-3],[-2,0,0],[-3,0,1]], dtype=np.float32)
            #     ker -= np.mean(ker)
            ker = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
            dist = cv2.filter2D(dist, cv2.CV_8U, ker)
            dist = cv2.blur(dist, (3, 3)).astype(np.float32)

            m = cv2.cvtColor(liquid_layer * dist, cv2.COLOR_GRAY2BGRA)
            m /= np.max(m, axis=(0, 1))
            m *= c[4]

            # water is pale turqouise
            color = np.concatenate((175 / 255. * np.ones_like(m[..., :1]),
                                    238 / 255. * np.ones_like(m[..., :1]),
                                    238 / 255. * np.ones_like(m[..., :1])), axis=2)

            color = cv2.cvtColor(color, cv2.COLOR_BGR2BGRA)
            x = cv2.cvtColor(x, cv2.COLOR_BGR2BGRA)

            return cv2.cvtColor(np.clip(x + m * color, 0, 1), cv2.COLOR_BGRA2BGR) * 255
        else:
            m = np.where(liquid_layer > c[3], 1, 0)
            m = gaussian(m.astype(np.float32), sigma=c[4])
            m[m < 0.8] = 0
            #         m = np.abs(m) ** (1/c[4])

            # mud brown
            color = np.concatenate((63 / 255. * np.ones_like(x[..., :1]),
                                    42 / 255. * np.ones_like(x[..., :1]),
                                    20 / 255. * np.ones_like(x[..., :1])), axis=2)

            color *= m[..., np.newaxis]
            x *= (1 - m[..., np.newaxis])

            return np.clip(x + color, 0, 1) * 255


    def contrast(x, severity=1):
        c = [0.4, .3, .2, .1, .05][severity - 1]

        x = np.array(x) / 255.
        means = np.mean(x, axis=(0, 1), keepdims=True)
        return np.clip((x - means) * c + means, 0, 1) * 255

    def saturate(x, severity=1):
        c = [(0.3, 0), (0.1, 0), (2, 0), (5, 0.1), (20, 0.2)][severity - 1]

        x = np.array(x) / 255.
        x = sk.color.rgb2hsv(x)
        x[:, :, 1] = np.clip(x[:, :, 1] * c[0] + c[1], 0, 1)
        x = sk.color.hsv2rgb(x)

        return np.clip(x, 0, 1) * 255

    # Pytorch torchvision transforms

    def brightness(x, severity=1):
        img =  transforms.functional.adjust_brightness(Image.fromarray(x), severity*0.5)
        return np.array(img)

    #TODO: Make three discrete layers
    def color_jitter(x, severity=1):
        b0 = 0.5
        c0 = 0.5
        cl0 = 0.5
        transform = transforms.ColorJitter(brightness=b0*severity, contrast = c0*severity, saturation= cl0*severity)
        img = transform(Image.fromarray(x))
        return np.array(img)

    #TODO: Add options to the config 
    def random_erase(x, severity=1):
        transf2tensor = transforms.ToTensor()
        transf2PIL = transforms.ToPILImage()
        transform = transforms.RandomErasing(p=0.8,scale=(0.02,0.02))
        img = transform(transf2tensor(Image.fromarray(x)))

        return np.array(transf2PIL(img))


    methods = {
        'random_erase': random_erase,
        'color_jitter': color_jitter,
        'gaussian_noise': gaussian_noise,
        'shot_noise': shot_noise,
        'impulse_noise': impulse_noise,
        'speckle_noise': speckle_noise,
        'gaussian_blur': gaussian_blur,
        'glass_blur': glass_blur,
        'defocus_blur': defocus_blur,
        'motion_blur':motion_blur,
        'zoom_blur':zoom_blur,
        'fog':fog,
        'spatter': spatter
    }
