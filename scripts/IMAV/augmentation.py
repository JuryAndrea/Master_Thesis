from genericpath import exists
import os
from os.path import join
import pathlib
import sys
import numpy as np
import glob

#matplotlib
from matplotlib import pyplot as plt
# %matplotlib inline      # must be done before matplotlib import
# %matplotlib notebook  # must be done before matplotlib import
# images
import cv2
import imgaug.augmenters as iaa
#Pillow
from PIL import Image as Image
from IPython.display import display as display
# WAND
from wand.image import Image as WImage
from wand.display import display as Wdisplay
# torch
import torch

def augment_dataset_and_save():

    path = '/home/lamberti/work/dataset/imav-dataset-aug/z_50/acquisition_around_empty_bg/images/'
    range_img=10500
    path = '/home/lamberti/work/dataset/imav-dataset-aug/z_50/acquisition_around_random_bg/images'
    range_img=10500
    path = '/home/lamberti/work/dataset/imav-dataset-aug/z_50/acquisition_random/images/'
    range_img=10000
    path = '/home/lamberti/work/dataset/imav-dataset-aug/z_50/acquisition_random_tr_ccw/images/'
    range_img=5000
    path = '/home/lamberti/work/dataset/imav-dataset-aug/z_50/acquisition_random_tr_cw/images/'
    range_img=5000

    image_paths = [path + f'img_{i}.png' for i in range(range_img)]


    images = {path: cv2.imread(path) for path in image_paths}

    save_folder = 'aug_samples'
    os.makedirs(folder, exist_ok=True)

    for i in range(0,100):
        image = next(iter(images.values()))
        name = 'debug_%d.png' %  i
        cv2.imwrite(join(folder, name), image)

        img_out, params_out = himax_augment(image)
        # img_out = cv2.cvtColor(img_out, cv2.COLOR_BGR2GRAY)
        name = 'aug_debug_%d.png' %  i
        cv2.imwrite(join(folder, name), img_out)




def debug():
    image=image_original
    folder = 'aug_samples'

    name = 'aug_debug_orig.png'
    cv2.imwrite(join(folder, name), image_original)

    image, self._params['motion_blur'] = motion_blur(image, self._params['motion_blur'])
    name = 'aug_debug_aug1.png'
    cv2.imwrite(join(folder, name), image)

    image, self._params['gaussian_blur'] = gaussian_blur(image, self._params['gaussian_blur'])
    name = 'aug_debug_aug2.png'
    cv2.imwrite(join(folder, name), image)

    image, self._params['vignette'] = vignette(image, self._params['vignette'])
    name = 'aug_debug_aug3.png'
    cv2.imwrite(join(folder, name), image)

    image, self._params['noise'] = gaussian_noise(image, self._params['noise'])
    name = 'aug_debug_aug4.png'
    cv2.imwrite(join(folder, name), image)

    image, self._params['exposure'] = exposure(image, self._params['exposure'])
    name = 'aug_debug_aug5.png'
    cv2.imwrite(join(folder, name), image)

    image = cv2.resize(image, (162, 162), cv2.INTER_AREA)
    name = 'aug_debug_aug6.png'
    cv2.imwrite(join(folder, name), image)


    image_paths = [f'/home/lamberti/work/IMAV2022/himax_augment/himax_augment/images/sim/img_{i}.png' for i in range(16)]
    images = {path: cv2.imread(path) for path in image_paths}
    image_right = next(iter(images.values()))
    img_out, params_out = himax_augment(image_right)
    # img_out = cv2.cvtColor(img_out, cv2.COLOR_BGR2GRAY)
    name = 'aug_debug_righttt.png'
    cv2.imwrite(join(folder, name), img_out)


def motion_blur(image, params={}):
    _params = dict(
        # TODO: extract sampled params from MotionBlur
        kernel=[3, 11] # Kernel size is sampled uniformly in the range [3, 15] pixels.
    )
    _params.update(params)

    # Docs: https://imgaug.readthedocs.io/en/latest/_modules/imgaug/augmenters/blur.html#MotionBlur
    aug = iaa.MotionBlur(_params['kernel'])
    image = aug.augment_image(image)

    return image, _params


def _inv_gamma(alpha, beta, **kwargs):
    # Source: https://distribution-explorer.github.io/continuous/inverse_gamma.html
    return 1 / np.random.gamma(alpha, 1/beta, **kwargs)

def gaussian_blur(image, params={}):
    _params = dict(
        # Using the inverse gamma as it's a coniugate prior distribution for the variance of a Gaussian,
        # i.e. a good distribution to represent the value of a variance which we don't know precisely.
        # Rooted because sigma is a standard deviation. Expected value: beta / (alpha - 1) = 1
        sigma=np.sqrt(_inv_gamma(alpha=5.0, beta=4))
    )
    _params.update(params)

    image = cv2.GaussianBlur(image, (0, 0), _params['sigma'])

    return image, _params

def gaussian_noise(image, params={}):
    _params = dict(
        mu=0.0,
        # Keeping sigma pretty low because it already has a lot of impact on BW images
        sigma=np.random.uniform(5, 10)
    )
    _params.update(params)

    noise = np.random.normal(_params['mu'], _params['sigma'], image.shape).astype(np.int8)
    image = cv2.add(image, noise, dtype=8)

    return image, _params

def _vignette_mask(height, width, sigma, min_alpha, max_alpha):
    # Generate vignette mask using Gaussian kernels
    kernel_x = cv2.getGaussianKernel(width, sigma)
    kernel_y = cv2.getGaussianKernel(height, sigma)
    kernel = kernel_y * kernel_x.T

    kmin, kmax = kernel.min(), kernel.max()
    mask = (kernel - kmin) / (kmax - kmin)
    mask = mask * (max_alpha - min_alpha) + min_alpha

    return mask


def vignette(image, params={}):
    _params = dict(
        sigma=np.random.uniform(80, 150),
        min_alpha=0.3,
        max_alpha=1.0
    )
    _params.update(params)

    h, w = image.shape[:2]
    mask = _vignette_mask(w, w, _params['sigma'], _params['min_alpha'], _params['max_alpha'])

    if image.ndim == 3:
        mask = mask[..., np.newaxis]

    offset = (w - h) // 2
    image = image * mask[offset:(offset+h), 0:w]
    image = image.astype(np.uint8)

    return image, _params

def _gamma_lut(gamma=1.0):
    invGamma = 1.0 / gamma
    lut = ((np.arange(256) / 255.0) ** invGamma * 255.0).astype(np.uint8)

    return lut

def exposure(image, params={}):
    choice = np.random.randint(3)

    _params = {}

    if choice == 0:
        _params['gain'] = 1.0
        _params['gain_jitter'] = np.random.normal(0.0, 0.3)
    elif choice == 1:
        _params['gamma'] = np.random.uniform(0.4, 2.0)
    elif choice == 2:
        _params['dr_low'] = np.random.uniform(0.0, 0.1)
        _params['dr_high'] = np.random.uniform(0.5, 1.0)

    _params.update(params)

    if 'gain_variance' in params.keys():
        a = params['gain_variance'][0]
        b = params['gain_variance'][1]
        _params['gain'] =  params['gain'] + np.random.uniform(a,b)

    # Gain
    image = cv2.multiply(
        image, np.ones_like(image),
        scale=_params.get('gain', 1.0) + _params.get('gain_jitter', 0.0)
    )

    # Gamma
    if 'gamma' in _params:
        lut = _gamma_lut(_params['gamma'])
        image = cv2.LUT(image, lut)

    # Dynamic range
    if 'dr_low' in _params or 'dr_high' in _params:
        image = image / 255.0
        image = np.interp(image, [0, _params.get('dr_low', 0.0), _params.get('dr_high', 1.0), 1], [0, 0, 1, 1])
        image = (image * 255.0).astype(np.uint8)

    return image, _params


def distort(image, params={}):
    # Attempt to recreate the radial distortion of the Himax camera

    # Source: https://stackoverflow.com/a/60612056/1821284
    # Additional info:
    #  - https://docs.wand-py.org/en/0.6.7/guide/distortion.html#barrel
    #  - https://answers.opencv.org/question/148670/re-distorting-a-set-of-points-after-camera-calibration/

    # Calibration params for AI-Deck #1 by Stefano Lambertenghi (camera matrix not needed here)
    # himax_mtx = np.array([
    #     [174.79236613, 0., 159.11692316],
    #     [0., 175.91204939, 168.92298902],
    #     [0., 0., 1.]
    # ], dtype=np.float32)
    himax_dist = np.array([-0.08778807, 0.0646409, -0.00081547, -0.00024427, -0.0534228], dtype=np.float32)

    _params = dict(
        # The distortion vector produced by OpenCV is meant for _correcting_ lens distortion,
        # while here we need to _simulate_ it, so invert the sign of the parameters.
        # Ignoring dist[2] and dist[3] because they don't match the parameters expected by wand.
        # Slightly perturb the distortion params to improve robustness
        k1 = -himax_dist[0] * np.random.normal(1.0, 1.0),
        k2 = -himax_dist[1] * np.random.normal(1.0, 1.0),
        k3 = -himax_dist[4] * np.random.normal(1.0, 1.0),
    )
    _params.update(params)

    color = image.ndim == 3 and image.shape[2] == 3
    image = WImage.from_array(image, 'bgr' if color else 'r')
    image.virtual_pixel = 'random' # Fill missing regions with random values
    image.distort('barrel', (_params['k1'], _params['k2'], _params['k3']))
    image = np.array(image)

    if color:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return image, _params

def bgr_to_bayer_bg(image):
    # Convert a 3-channel BGR image to the BG Bayer pattern of the Himax

    height, width = image.shape[:2]
    b, g, r = cv2.split(image)
    out = np.empty((height, width), np.uint8)

    # strided slicing for the BG (i.e. RGGB) pattern:
    #   R G
    #   G B
    out[0::2, 0::2] = r[0::2, 0::2] # top left
    out[0::2, 1::2] = g[0::2, 1::2] # top right
    out[1::2, 0::2] = g[1::2, 0::2] # bottom left
    out[1::2, 1::2] = b[1::2, 1::2] # bottom right

    return out

def himax_augment(image, params={}):
    # Apply a sequence of transformations that map a simulated image to a real-looking Himax image

    _params = dict(
        # Simulate images produced by color Himax cameras
        color=False,

        # Leave default values
        motion_blur={},
        gaussian_blur={},
        distort={},
        vignette={},
        noise={},
        exposure=dict(gain=2.0),  # Images from the simulator are a bit dark, hardcode a 2x gain

        # Set to 162x162 to simulate binning
        height=162,
        width=162,
    )
    _params.update(params)

    if not _params['color']:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image, _params['motion_blur'] = motion_blur(image, _params['motion_blur'])
    image, _params['gaussian_blur'] = gaussian_blur(image, _params['gaussian_blur'])
    # image, _params['distort'] = distort(image, _params['distort'])
    image, _params['vignette'] = vignette(image, _params['vignette'])
    image, _params['noise'] = gaussian_noise(image, _params['noise'])
    image, _params['exposure'] = exposure(image, _params['exposure'])

    if _params['color']:
        image = bgr_to_bayer_bg(image)

    image = cv2.resize(image, (_params['width'], _params['height']), cv2.INTER_AREA)



class ImgAugTransform:
    def __init__(self, train=True, aug='all', params={}):
        # self.aug = iaa.Sequential([
        #     iaa.Scale((224, 224)),
        #     iaa.Sometimes(0.25, iaa.GaussianBlur(sigma=(0, 3.0))),
        #     iaa.Fliplr(0.5),
        #     iaa.Affine(rotate=(-20, 20), mode='symmetric'),
        #     iaa.Sometimes(0.25,
        #                   iaa.OneOf([iaa.Dropout(p=(0, 0.1)),
        #                              iaa.CoarseDropout(0.1, size_percent=0.5)])),
        #     iaa.AddToHueAndSaturation(value=(-10, 10), per_channel=True)
        # ])
        self.train=train
        self.aug=aug
        self.aug_enable = dict(
            blur     = 1            if (self.aug=='all' or self.aug=='blur'     or self.aug=='blur_noise'    or self.aug=='blur_exposure' ) else 0,
            noise    = 1            if (self.aug=='all' or self.aug=='noise'    or self.aug=='blur_noise'    or self.aug=='noise_exposure') else 0,
            exposure = 1            if (self.aug=='all' or self.aug=='exposure' or self.aug=='blur_exposure' or self.aug=='noise_exposure') else 0,
            gain_variance = [-1,1]  if (self.aug=='all' or self.aug=='exposure' or self.aug=='blur_exposure' or self.aug=='noise_exposure') else [0,0],
            # gain_variance = [-1.3,2] if (self.aug=='all' or self.aug=='exposure') else [0,0],
            # gain_variance = [-2,4]   if (self.aug=='all' or self.aug=='exposure') else [0,0],
        )

        print("Augmentations active:", f"Blur: {self.aug_enable['blur']}, Noise: {self.aug_enable['noise']}, Exposure: {self.aug_enable['exposure']}"
)
        self._params = dict(
            # Simulate images produced by color Himax cameras
            color=False,
            # Leave default values
            motion_blur={},
            gaussian_blur={},
            distort={},
            vignette={},
            noise={},
            exposure=dict(gain=2.0,
                          gain_variance=self.aug_enable['gain_variance']
                        #   gain_variance=[-1.3,2]
                        #   gain_variance=[-2,4]
                          ),  # Images from the simulator are a bit dark, hardcode a 2x gain

            # Set to 162x162 to simulate binning
            # height=324,
            # width=324,
        )
        self._params_last = dict(
            # Simulate images produced by color Himax cameras
            color=False,
            # Leave default values
            motion_blur={},
            gaussian_blur={},
            distort={},
            vignette={},
            noise={},
            exposure={},  # Images from the simulator are a bit dark, hardcode a 2x gain
            # Set to 162x162 to simulate binning
            # height=324,
            # width=324,
        )
        self._params.update(params)

        print(self._params)

    def __call__(self, image):
        image = np.array(image)
        # return self.aug.augment_image(img)

        if not self._params['color']:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        #select augmentation

        if self.train:
            if self.aug_enable['blur']:
                image, self._params_last['motion_blur'] = motion_blur(image, self._params['motion_blur'])
                image, self._params_last['gaussian_blur'] = gaussian_blur(image, self._params['gaussian_blur'])
            # image, self._params['distort'] = distort(image, self._params['distort'])
            if self.aug_enable['exposure']:
                image, self._params_last['vignette'] = vignette(image, self._params['vignette'])
            if self.aug_enable['noise']:
                image, self._params_last['noise'] = gaussian_noise(image, self._params['noise'])
            if self.aug_enable['exposure']:
                image, self._params_last['exposure'] = exposure(image, self._params['exposure'])

            if self._params['color']:
                image = bgr_to_bayer_bg(image)
            # debug: save to file
            # img_np = (image*255).detach().cpu().numpy().astype(np.uint8).squeeze()
            # cv2.imwrite('aug_debug2.png', img_np)

        # image = cv2.resize(image, (self._params['width'], self._params['height']), cv2.INTER_AREA)

        if not self._is_pil_image(image):
            image = Image.fromarray(np.uint8(image))

        return image#, self._params_last

    def _is_pil_image(self, img):
        return isinstance(img, Image.Image)


import torchvision.transforms as transforms
class Brightness:
    def __init__(self, brightness_factor=1.):
        self.brightness_factor = brightness_factor
    def __call__(self, image):
        image = transforms.functional.adjust_brightness(image, self.brightness_factor)
        # image.save('debug.png')
        return image


def test_himax_augment():
    image_paths = [f'/home/lamberti/work/IMAV2022/himax_augment/himax_augment/images/sim/img_{i}.png' for i in range(16)]
    images = {path: cv2.imread(path) for path in image_paths}

    folder = 'aug_samples'
    os.makedirs(folder, exist_ok=True)

    for i in range(0,100):
        image = next(iter(images.values()))
        name = 'debug_%d.png' %  i
        cv2.imwrite(join(folder, name), image)

        img_out, params_out = himax_augment(image)
        # img_out = cv2.cvtColor(img_out, cv2.COLOR_BGR2GRAY)
        name = 'aug_debug_%d.png' %  i
        cv2.imwrite(join(folder, name), img_out)

def next_aug_image(dataset_loader):
    ### USE DATASET TO ITERATE
    # dataset_iterator = iter(train_dataset)
    # img = next(dataset_iterator)[0]

    ### USE DATALOADER TO ITERATE
    loader_iterator = iter(dataset_loader) # i use dataset loader to shuffle testing images
    data = next(loader_iterator)
    img, label, filename = data[0], data[1], data[2]
    img = torch.squeeze(img ,0)

    # image back to numpy array
    img_np = (img*255).detach().cpu().numpy().astype(np.uint8).squeeze()
    return img_np

def test_ImgAugTransform():
    from utility import DronetDatasetV3
    from models import Dataset
    import torchvision.transforms as transforms
    import torch

    # data path
    data_path     = '/home/lamberti/work/dataset/imav-dataset'

    image_size= '162x162'
    # transformations
    transf_list =[]
    if image_size== '162x162': transf_list += [transforms.Resize(162)]
    transf_list += [ImgAugTransform()]
    transf_list += [
        # transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
        ]
    transformations = transforms.Compose(transf_list)

    dataset = Dataset(data_path)
    dataset.initialize_from_filesystem()

    # load training set
    train_dataset = DronetDatasetV3(
        transform=transformations,
        dataset=dataset,
        dataset_type='imav',
        selected_partition='train',
        # classification=args.yaw_rate_as_classification
        )

    # dataloader
    dataset_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=1,
    shuffle=True,
    num_workers=1)

    folder = 'aug_samples'
    os.makedirs(folder, exist_ok=True)

    for i in range(0,100):
        img_np = next_aug_image(dataset_loader)
        # save
        name = 'aug_debug_%d.png' %  i
        cv2.imwrite(join(folder, name), img_np)


if __name__ == '__main__':

    test_ImgAugTransform()
    # test_himax_augment()

    print('end')



