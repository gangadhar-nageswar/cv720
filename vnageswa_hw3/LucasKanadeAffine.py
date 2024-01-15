from textwrap import indent
import numpy as np
from numpy.linalg import inv
from numpy import linalg as LA
import cv2
import scipy
from scipy.interpolate import RectBivariateSpline
import scipy.ndimage
from scipy.ndimage import affine_transform



def LucasKanadeAffine(It, It1, threshold, num_iters):
    """
    :param It: template image
    :param It1: Current image
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :return: M: the Affine warp matrix [2x3 numpy array] put your implementation here
    """

    orig_It = np.copy(It)
    orig_It1 = np.copy(It1)
    
    # Put your implementation here
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    p = np.zeros(shape=(6,1))

    for itr in range(num_iters):
        # print(f"itr: {itr}")

        It = affine_transform(orig_It1, M)

        # Compute gradient
        # dogIx = scipy.ndimage.gaussian_filter(orig_It1, sigma=[0,0.2], order=1)
        # dogIy = scipy.ndimage.gaussian_filter(orig_It1, sigma=[0.2,0], order=1)
        
        # dogIx = scipy.ndimage.sobel(orig_It1, 1)
        # dogIy = scipy.ndimage.sobel(orig_It1, 0)

        dogIx = cv2.Sobel(orig_It1, cv2.CV_64F, 0, 1)
        dogIy = cv2.Sobel(orig_It1, cv2.CV_64F, 1, 0)

        # warp the gradient
        dogIx = affine_transform(dogIx, M)
        dogIy = affine_transform(dogIy, M)

        H = np.zeros(shape=(6,6))
        rhs = 0

        for xind in range(5,orig_It.shape[1]-5):
            for yind in range(5,orig_It.shape[0]-5):
                It_grad = np.array([[dogIx[yind][xind], dogIy[yind][xind]]])
                J = np.array([[xind, yind, 1, 0, 0, 0], [0, 0, 0, xind, yind, 1]])

                # print(f"shape J: {J.shape} || It1_grad: {It1_grad.shape}")
                A = np.matmul(It_grad, J)
                H += np.matmul(np.transpose(A), A)
                
                b = orig_It[yind][xind] - It[yind][xind]
                rhs += b*np.transpose(A)
        

        delta_p = np.matmul(inv(H), rhs)
        p += delta_p

        # print(f"delta p: {delta_p}")

        M = np.array([[1+p[0][0], p[1][0], p[2][0]], [p[3][0], 1+p[4][0], p[5][0]], [0, 0, 1]])
        if LA.norm(delta_p, 2) <= threshold:
            break
    
    # print(LA.norm(p, 2))
    # warp function has to return integers
    # p = np.round_(p)
    # M = np.array([[1+p[0][0], p[1][0], p[2][0]], [p[3][0], 1+p[4][0], p[5][0]], [0, 0, 1]])

    Minv = inv(M)
    Minv = np.delete(Minv, Minv.shape[0] - 1, axis=0)

    return Minv





























'''
Previous Implementation:
Date/Time used: 25th Oct, 12:23 am


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


def compute_H_and_RHS_term(ind, Xorig_trim, Xnew_trim, warped_It1_dogIx, warped_It1_dogIy, It, warped_It1):
    # print(f"........ {ind} .......")
    xind, yind = Xorig_trim[ind][0], Xorig_trim[ind][1]
    It_grad = np.array([[warped_It1_dogIx[ind], warped_It1_dogIy[ind]]])
    J = np.array([[xind, yind, 1, 0, 0, 0], [0, 0, 0, xind, yind, 1]])

    # print(f"shape J: {J.shape} || It1_grad: {It1_grad.shape}")
    A = np.matmul(It_grad, J)
    
    H = np.matmul(np.transpose(A), A)

    b = It[ind] - warped_It1[ind]
    rhs = b*np.transpose(A)

    return [H, rhs]


def LucasKanadeAffine(It, It1, threshold, num_iters):
    """
    :param It: template image
    :param It1: Current image
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :return: M: the Affine warp matrix [2x3 numpy array] put your implementation here
    """

    # multiprocessing
    n_cpu = multiprocessing.cpu_count()

    orig_It = np.copy(It)
    orig_It1 = np.copy(It1)
    
    # Put your implementation here
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    p0 = np.zeros(shape=(6,1))
    p = np.copy(p0)

    ####### this implementation has been replaced with affine_transform function ####################
    x_grid = np.arange(orig_It1.shape[1])
    y_grid = np.arange(orig_It1.shape[0])
    x_grid, y_grid = np.meshgrid(x_grid, y_grid)

    OrigMesh = np.stack([x_grid, y_grid], axis=-1)
    OrigMesh = OrigMesh.reshape(-1, 2)
    print(f"OrigMesh shape: {OrigMesh.shape} || ones: {np.ones(shape=(OrigMesh.shape[0],1)).shape}")
    OrigMesh = np.concatenate((OrigMesh, np.ones(shape=(OrigMesh.shape[0],1))), axis=1)

    # initialise interpolator for obtaining image intensity after warping
    It1_interpolator = RectBivariateSpline(np.arange(orig_It1.shape[0]), np.arange(orig_It1.shape[1]), orig_It1)
    
    It1_dogIx_orig = cv2.Sobel(orig_It1, cv2.CV_64F, 0, 1)
    It1_dogIy_orig = cv2.Sobel(orig_It1, cv2.CV_64F, 1, 0)

    # It1_dogIx_orig = scipy.ndimage.gaussian_filter(orig_It1, sigma=[0,1], order=1)
    # It1_dogIy_orig = scipy.ndimage.gaussian_filter(orig_It1, sigma=[1,0], order=1)


    # It1_dogIx_orig = scipy.ndimage.gaussian_filter(orig_It1, sigma=[0,1], order=1)
    # It1_dogIy_orig = scipy.ndimage.gaussian_filter(orig_It1, sigma=[1,0], order=1)

    It1_dogIx_interpolator = RectBivariateSpline(np.arange(It1_dogIx_orig.shape[0]), np.arange(It1_dogIx_orig.shape[1]), It1_dogIx_orig)
    It1_dogIy_interpolator = RectBivariateSpline(np.arange(It1_dogIy_orig.shape[0]), np.arange(It1_dogIy_orig.shape[1]), It1_dogIy_orig)


    for itr in range(num_iters):
        print(f"itr: {itr}")

        Xorig = np.copy(OrigMesh)
        Xnew = np.matmul(Xorig, np.transpose(inv(M)))

        # remove points out of boundary
        x_values = Xnew[:, 0]
        y_values = Xnew[:, 1]
        rowsDelete = np.where((x_values > orig_It.shape[1]) | (x_values < 0) | (y_values > orig_It.shape[0]) | (y_values < 0))[0]

        Xnew_trim = np.delete(Xnew, rowsDelete, axis=0)
        Xorig_trim = np.delete(Xorig, rowsDelete, axis=0)
        It = np.delete(orig_It.flatten(), rowsDelete, axis=0)

        x_new = Xnew_trim[:,0]
        y_new = Xnew_trim[:,1]

        warped_It1  = It1_interpolator.ev(y_new, x_new)
        warped_It1_dogIx = It1_dogIx_interpolator.ev(y_new, x_new)
        warped_It1_dogIy = It1_dogIy_interpolator.ev(y_new, x_new)


        img_args = [(ind, Xorig_trim, Xnew_trim, warped_It1_dogIx, warped_It1_dogIy, It, warped_It1) for ind in range(Xnew_trim.shape[0])]

        pool = multiprocessing.Pool(n_cpu)
        termValues = pool.starmap(compute_H_and_RHS_term, img_args)
        print(termValues)

        H = sum(termValues[:][0])
        rhs = sum(termValues[:][1])

        print(f"H : {H} || rhs: {rhs}")


        # # initialise to zeros
        # H = np.zeros(shape=(6,6))
        # rhs = 0

        # for ind in range(Xnew_trim.shape[0]):
        #     xind, yind = Xorig_trim[ind][0], Xorig_trim[ind][1]
        #     It_grad = np.array([[warped_It1_dogIx[ind], warped_It1_dogIy[ind]]])
        #     J = np.array([[xind, yind, 1, 0, 0, 0], [0, 0, 0, xind, yind, 1]])

        #     # print(f"shape J: {J.shape} || It1_grad: {It1_grad.shape}")
        #     A = np.matmul(It_grad, J)
        #     H += np.matmul(np.transpose(A), A)
            
        #     b = It[ind] - warped_It1[ind]
        #     rhs += b*np.transpose(A)
        
        print(f"itr: {itr} || H: {H} || rhs: {rhs}")
        delta_p = np.matmul(inv(H), rhs)
        p += delta_p

        print(f"shape p: {p.shape} || delta p: {delta_p.shape}")

        M = np.array([[1+p[0][0], p[1][0], p[2][0]], [p[3][0], 1+p[4][0], p[5][0]], [0, 0, 1]])
        if LA.norm(delta_p, 2) <= threshold:
            break
    
    print(LA.norm(p, 2))
    # warp function has to return integers
    # p = np.round_(p)
    # M = np.array([[1+p[0][0], p[1][0], p[2][0]], [p[3][0], 1+p[4][0], p[5][0]], [0, 0, 1]])

    # print(f"num of itrs: {itr} || Matrix M: {M}")

    return M

'''









'''
Previous Implementation:

def LucasKanadeAffine(It, It1, threshold, num_iters):
    """
    :param It: template image
    :param It1: Current image
    :param threshold: if the length of dp is smaller than the threshold, terminate the optimization
    :param num_iters: number of iterations of the optimization
    :return: M: the Affine warp matrix [2x3 numpy array] put your implementation here
    """


    orig_It = np.copy(It)
    orig_It1 = np.copy(It1)

    # x_orig = np.arange(orig_It.shape[1])
    # y_orig = np.arange(orig_It.shape[0])
    # Xorig = np.transpose(np.concatenate((x_orig, y_orig), axis=1))
    # Xorig = np.concatenate((Xorig, np.ones(shape=())), axis=0)

    ######## this implementation has been replaced with affine_transform function ####################
    # x_grid = np.arange(orig_It.shape[1])
    # y_grid = np.arange(orig_It.shape[0])
    # x_grid, y_grid = np.meshgrid(x_grid, y_grid)

    # OrigMesh = np.stack([x_grid, y_grid], axis=-1)
    # OrigMesh = OrigMesh.reshape(-1, 2)
    # print(f"OrigMesh shape: {OrigMesh.shape} || ones: {np.ones(shape=(OrigMesh.shape[0],1)).shape}")
    # OrigMesh = np.concatenate((OrigMesh, np.ones(shape=(OrigMesh.shape[0],1))), axis=1)

    ## initialise interpolator for obtaining image intensity after warping
    # interpolator = RectBivariateSpline(np.arange(orig_It1.shape[0]), np.arange(orig_It1.shape[1]), orig_It1)
    ###################################################################################################

    # Put your implementation here
    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    p0 = np.zeros(shape=(6,1))
    p = np.copy(p0)

    for itr in range(num_iters):
        # Warp the current image
        # Obtain the new rect using the template offset delta_p
        # x and y indices of the new warped rectangle
        
        ## should use it ???? Xnew = affine_transform(It, M) ???

        # Xnew = np.matmul(OrigMesh, np.transpose(M))
        # x_new = Xnew[:,0]
        # y_new = Xnew[:,1]

        # It1  = interpolator.ev(y_new, x_new)
        # It1 = np.reshape(It1, (orig_It1.shape))

        It = affine_transform(orig_It1, inv(M))

        # Compute the gradient x and gradient y -- > use DOG
        # shape -> Nx2 array
        dogIx = scipy.ndimage.gaussian_filter(It1, sigma=[0,1], order=1)
        dogIy = scipy.ndimage.gaussian_filter(It1, sigma=[1,0], order=1)

        # compute jacobian
        # In this case it is equal to Identity
        
        H = np.zeros(shape=(6,6))
        rhs = 0

        for xind in range(orig_It1.shape[1]):
            for yind in range(orig_It1.shape[0]):
                It1_grad = np.array([[dogIx[yind][xind], dogIy[yind][xind]]])
                J = np.array([[xind, yind, 1, 0, 0, 0], [0, 0, 0, xind, yind, 1]])

                # print(f"shape J: {J.shape} || It1_grad: {It1_grad.shape}")
                A = np.matmul(It1_grad, J)
                H += np.matmul(np.transpose(A), A)
                
                b = It[yind][xind] - It1[yind][xind]
                rhs += b*np.transpose(A)
        

        delta_p = np.matmul(inv(H), rhs)
        p += delta_p

        M = np.array([[1+p[0][0], p[1][0], p[2][0]], [p[3][0], 1+p[4][0], p[5][0]], [0, 0, 1]])
        if LA.norm(delta_p, 2) <= threshold:
            break
    
    print(LA.norm(p, 2))
    # warp function has to return integers
    p = np.round_(p)
    M = np.array([[1+p[0][0], p[1][0], p[2][0]], [p[3][0], 1+p[4][0], p[5][0]], [0, 0, 1]])

    print(f"num of itrs: {itr} || Matrix M: {M}")

    return M



'''