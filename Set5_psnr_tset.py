import numpy as np
import cv2
from os.path import join
from os import listdir
from math import log10
import matlab_imresize


class Resize(object):
    def __init__(self, scale_factor):
        self.scale_factor = scale_factor

    def __call__(self, img):
        dsize = (int(np.shape(img)[1] / self.scale_factor), int(np.shape(img)[0] / self.scale_factor))
        return matlab_imresize.imresize(img, output_shape=dsize)
        #return cv2.resize(img, dsize=dsize, interpolation=cv2.INTER_CUBIC)


def get_psnr(img1, img2, min_value=0, max_value=255):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = max_value - min_value
    return 10 * log10((PIXEL_MAX ** 2) / mse)

# I get set5 in this site (http://vllab.ucmerced.edu/wlai24/LapSRN/)
origin_test_dir = 'Set5'
origin_tests = [join(origin_test_dir, x) for x in sorted(listdir(origin_test_dir))]

resize = Resize(4)
resize_inverse = Resize(1 / 4)

psnrs = []

for origin_test in origin_tests:
    img_hr = cv2.imread(origin_test, cv2.IMREAD_COLOR)

    img_hr = cv2.cvtColor(img_hr, cv2.COLOR_BGR2YCrCb)

    img_hr = img_hr.astype(np.float64)

    img_lr = resize(img_hr)
    img_hr_new = resize_inverse(img_lr)

    img_hr = img_hr[4:-4, 4:-4, 0]
    img_hr_new = img_hr_new[4:-4, 4:-4, 0]

    img_hr = img_hr.clip(0, 255)
    img_hr_new = img_hr_new.clip(0, 255)

    img_hr = img_hr.astype(np.uint8)
    img_hr_new = img_hr_new.astype(np.uint8)

    psnr = get_psnr(img_hr.astype(np.float64), img_hr_new.astype(np.float64), 0, 255)
    psnrs.append(psnr)

print('mean psnr :', np.mean(psnrs))