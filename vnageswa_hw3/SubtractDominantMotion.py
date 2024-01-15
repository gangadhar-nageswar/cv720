from cmath import sqrt
from operator import mod
import numpy as np
import cv2
from scipy import ndimage
from numpy import linalg as LA
from LucasKanadeAffine import LucasKanadeAffine
from InverseCompositionAffine import InverseCompositionAffine
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import affine_transform


def SubtractDominantMotion(image1, image2, threshold, num_iters, tolerance):
    """
    :param image1: Images at time t
    :param image2: Images at time t+1
    :param threshold: used for LucasKanadeAffine
    :param num_iters: used for LucasKanadeAffine
    :param tolerance: binary threshold of intensity difference when computing the mask
    :return: mask: [nxm]
    """
    
    # put your implementation here
    mask = np.ones(image2.shape, dtype=bool)
    
    # M = LucasKanadeAffine(image1, image2, threshold, num_iters)
    M = InverseCompositionAffine(image1, image2, threshold, num_iters)
    
    M = np.vstack((M, np.array([0, 0, 1])))
    print(f"M: {M}")
    
    It1 = affine_transform(image1, M)

    diff_img = abs(It1 - image2)

    mask = np.where(diff_img < tolerance, diff_img, 255)
    mask[0:5, :] = 0
    mask[-5:, :] = 0
    mask[:, 0:5] = 0
    mask[:, -5:] = 0
    

    # for i in range(rowsDelete.shape[0]):
    #     x_ind, y_ind = int(OrigMesh[i,0]), int(OrigMesh[i,1])
    #     mask[x_ind][y_ind] = 0


    # erode the image
    # kernel = np.ones((3, 3), np.uint8)

    # mask = cv2.erode(mask, np.ones((1, 1), np.uint8))
    # mask = cv2.dilate(mask, np.ones((3, 3), np.uint8))
    # mask = cv2.erode(mask, np.ones((5, 5), np.uint8))
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8))

    
    return mask





'''
Prev Implementation 1:

def SubtractDominantMotion(image1, image2, threshold, num_iters, tolerance):
    """
    :param image1: Images at time t
    :param image2: Images at time t+1
    :param threshold: used for LucasKanadeAffine
    :param num_iters: used for LucasKanadeAffine
    :param tolerance: binary threshold of intensity difference when computing the mask
    :return: mask: [nxm]
    """
    
    # put your implementation here
    mask = np.ones(image2.shape, dtype=bool)
    
    M = LucasKanadeAffine(image1, image2, threshold, num_iters)
    It1 = affine_transform(image1, M)
    
    # x_grid = np.arange(image1.shape[1])
    # y_grid = np.arange(image1.shape[0])
    # x_grid, y_grid = np.meshgrid(x_grid, y_grid)

    # OrigMesh = np.stack([x_grid, y_grid], axis=-1)
    # OrigMesh = OrigMesh.reshape(-1, 2)
    # OrigMesh = np.concatenate((OrigMesh, np.ones(shape=(OrigMesh.shape[0],1))), axis=1)

    # Xnew = np.matmul(OrigMesh, np.transpose(M))
    # x_new = Xnew[:,0]
    # y_new = Xnew[:,1]

    # interpolator = RectBivariateSpline(np.arange(image1.shape[0]), np.arange(image1.shape[1]), image1)
    # It1  = interpolator.ev(y_new, x_new)
    # It1 = np.reshape(It1, (image1.shape))

    diff_img = It1 - image2
    diff_img = diff_img * diff_img

    mask = np.where(diff_img < tolerance, diff_img, 255)

    # erode the image
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel)
    
    
    return mask


'''