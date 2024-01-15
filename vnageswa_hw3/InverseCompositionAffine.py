from textwrap import indent
import numpy as np
import cv2
from numpy.linalg import inv
from numpy import linalg as LA
import scipy
from scipy.interpolate import RectBivariateSpline
import scipy.ndimage
from scipy.ndimage import affine_transform
import multiprocessing

def InverseCompositionAffine(It, It1, threshold, num_iters):
    """
    :param It: template image
    :param It1: Current image
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :return: M: the Affine warp matrix [2x3 numpy array]
    """

    orig_It = np.copy(It)
    orig_It1 = np.copy(It1)
    
    # put your implementation here
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    p0 = np.zeros(6)
    p = np.copy(p0)

    # gradient of the template
    It_dogIx_orig = cv2.Sobel(orig_It, cv2.CV_64F, 0, 1)
    It_dogIy_orig = cv2.Sobel(orig_It, cv2.CV_64F, 1, 0)
    
    H = np.zeros(shape=(6,6))
    A = np.zeros(shape=(orig_It.shape[0], orig_It.shape[1], 1, 6))
    
    # Compute Hessian
    for xind in range(orig_It.shape[1]):
        for yind in range(orig_It.shape[0]):
            
            grad_It = np.array([[It_dogIx_orig[yind][xind], It_dogIy_orig[yind][xind]]])  # are the indices order correct? cuz it is from cv2
            J = np.array([[xind, yind, 1, 0, 0, 0], [0, 0, 0, xind, yind, 1]])

            Aterm = np.matmul(grad_It, J)
            H += np.matmul(np.transpose(Aterm), Aterm)

            A[yind, xind, :, :] = Aterm


    for itr in range(num_iters):
        warped_It1 = affine_transform(orig_It1, M)

        rhs = np.zeros(shape=(6,1))
        for xind in range(orig_It.shape[1]):
            for yind in range(orig_It.shape[0]):

                Aterm = A[yind, xind, :, :]
                rhs += Aterm.transpose()*(orig_It[yind][xind] - warped_It1[yind][xind])

        delta_p = -np.matmul(inv(H), rhs).ravel()
        delta_M = np.array([[1+delta_p[0], delta_p[1], delta_p[2]], [delta_p[3], 1+delta_p[4], delta_p[5]], [0, 0, 1]])
        M = np.matmul(M, np.linalg.pinv(delta_M))
        M[2,:] = [0, 0, 1]

        if LA.norm(delta_p) <= threshold:
            break
    
    Minv = inv(M)
    Minv = np.delete(Minv, Minv.shape[0] - 1, axis=0)

    return Minv