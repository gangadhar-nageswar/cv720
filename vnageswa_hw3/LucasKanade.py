import numpy as np
from numpy.linalg import inv
from numpy import linalg as LA
import scipy
import scipy.ndimage
from scipy.interpolate import RectBivariateSpline

import cv2

def LucasKanade(It, It1, rect, threshold, num_iters, p0=np.zeros(2)):
    """
    :param It: template image
    :param It1: Current image
    :param rect: Current position of the car (top left, bot right coordinates)
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :param p0: Initial movement vector [dp_x0, dp_y0]
    :return: p: movement vector [dp_x, dp_y]
    """
    
    orig_It = np.copy(It)
    orig_It1 = np.copy(It1)

    # Put your implementation here
    p = np.copy(p0)

    # crop the required part of the image
    # the code assumes that the images are grayscale
    tc1,tr1 = int(rect[0]), int(rect[1])
    tc2,tr2 = int(rect[2]+1), int(rect[3]+1)

    It  = It[tr1:tr2,tc1:tc2]

    # initialise interpolator for obtaining image intensity after warping
    interpolator = RectBivariateSpline(np.arange(orig_It1.shape[0]), np.arange(orig_It1.shape[1]), orig_It1)

    for itr in range(num_iters):
        # Warp the current image
        # Obtain the new rect using the template offset delta_p
        # x and y indices of the new warped rectangle
        x_new = [i+p[0] for i in range(tc1, tc2)]
        y_new = [j+p[1] for j in range(tr1, tr2)]

        It1  = interpolator(y_new, x_new)
        # cv2.imshow('Original', It1)
        # cv2.waitKey(10)

        # Error
        b = It - It1
        b = b.flatten()

        # Compute the gradient x and gradient y -- > use DOG
        # shape -> Nx2 array
        dogIx = scipy.ndimage.gaussian_filter(It1, sigma=[0,1], order=1)
        dogIx = dogIx.flatten()
        dogIx = np.expand_dims(dogIx, 1)

        dogIy = scipy.ndimage.gaussian_filter(It1, sigma=[1,0], order=1)
        dogIy = dogIy.flatten()
        dogIy = np.expand_dims(dogIy, 1)

        It1_grad = np.concatenate((dogIx, dogIy), axis=1)

        # compute jacobian
        # In this case it is equal to Identity
        J = np.identity(2)

        # compute hessian
        A = np.matmul(It1_grad, J)
        H = np.matmul(np.transpose(A), A)

        # print(f"It1_grad: {It1_grad} \nA = {A} \nH = {H}")
        rhs = np.matmul(np.transpose(A), b)
        delta_p = np.matmul(inv(H), rhs)

        # p += delta_p
        p = np.ravel(p + delta_p.transpose())

        if LA.norm(delta_p, 2) <= threshold:
            break

    # warp function has to return integers
    p = np.round_(p)
    print(f"num of itrs: {itr} || p value: {p}")

    return p
