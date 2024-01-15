# ##################################################################### #
# 16720: Computer Vision Homework 6
# Carnegie Mellon University
# April 20, 2020
# ##################################################################### #

import numpy as np
from q1 import loadData, estimateAlbedosNormals, displayAlbedosNormals
from q1 import estimateShape, plotSurface 
from utils import enforceIntegrability
from matplotlib import pyplot as plt
from matplotlib import cm

def estimatePseudonormalsUncalibrated(I):

    """
    Question 2 (b)

    Estimate pseudonormals without the help of light source directions. 

    Parameters
    ----------
    I : numpy.ndarray
        The 7 x P matrix of loaded images

    Returns
    -------
    B : numpy.ndarray
        The 3 x P matrix of pesudonormals

    """

    U, S, V = np.linalg.svd(I, full_matrices=False)
    
    S[3:] = 0
    S = S[:3]
    S = np.diag(S)

    B = V[:3, :]
    U = U[:, :3]    
    L = U@S

    return B, L.T



if __name__ == "__main__":

    # Put your main code here
    # Q2b

    I, L_orig, s = loadData()
    B, L = estimatePseudonormalsUncalibrated(I)

    print(f"original L: {L_orig}")
    print(f"\n\nestimated L: {L}")


    # 2d
    albedos, normals = estimateAlbedosNormals(B)
    albedoIm, normalIm = displayAlbedosNormals(albedos, normals, s)
    
    surface=estimateShape(normals, s)
    minv, maxv = np.min(surface), np.max(surface)
    surface = (surface - minv) / (maxv - minv)    
    # surface = (surface * 255.).astype('uint8')
    plotSurface(surface)


    # 2e
    normals = enforceIntegrability(normals, s)
    albedoIm, normalIm = displayAlbedosNormals(albedos, normals, s)

    surface=estimateShape(normals, s)
    minv, maxv = np.min(surface), np.max(surface)
    surface = (surface - minv) / (maxv - minv)    
    # surface = (surface * 255.).astype('uint8')
    plotSurface(surface)


    # 2f
    mu = 0
    nu = 0
    lam = 1
    G = np.asarray([[1, 0, 0], [0, 1, 0], [mu, nu, lam]])
    B = np.linalg.inv(G.T).dot(B)

    albedos, normals = estimateAlbedosNormals(B)
    normals = enforceIntegrability(normals, s)
    albedoIm, normalIm = displayAlbedosNormals(albedos, normals, s)

    surface=estimateShape(normals, s)
    minv, maxv = np.min(surface), np.max(surface)
    surface = (surface - minv) / (maxv - minv)
    plotSurface(surface)
