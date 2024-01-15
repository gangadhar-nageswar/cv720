import numpy as np

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.filters
import skimage.morphology
import skimage.segmentation

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from skimage import data
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb

from skimage import data, io
from matplotlib import pyplot as plt

import cv2



# takes a color image
# returns a list of bounding boxes and black_and_white image
def findLetters(image):
    bboxes = []
    bw = None
    # insert processing in here
    # one idea estimate noise -> denoise -> greyscale -> threshold -> morphology -> label -> skip small boxes 
    # this can be 10 to 15 lines of code using skimage functions

    ##########################
    ##### your code here #####
    ##########################

    # gaussian blur
    image = skimage.filters.gaussian(image, sigma=1.5)
    grey_image = skimage.color.rgb2gray(image)
    thresh = threshold_otsu(grey_image)
    grey_image = grey_image > thresh

    bw = closing(grey_image < thresh, square(11))
    # grey_image[grey_image < thresh] = 0

    cleared = clear_border(bw)
    label_image = label(cleared)

    # plt.imshow(skimage.util.invert(cleared), cmap='gray')
    # plt.show()

    for region in regionprops(label_image):
        # take regions with large enough areas
        if region.area >= 100:
            bb = region.bbox
            bboxes.append(bb)
            

    # bw = 1-bw
    # bw = skimage.util.invert(bw)

    bw = skimage.morphology.binary_dilation(bw)

    return bboxes, bw