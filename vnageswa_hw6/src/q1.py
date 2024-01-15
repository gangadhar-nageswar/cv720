# ##################################################################### #
# 16720: Computer Vision Homework 6
# Carnegie Mellon University
# April 20, 2020
# ##################################################################### #

# Imports
from tkinter import Image
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import cv2
import os
from skimage.color import rgb2xyz
from utils import integrateFrankot

def renderNDotLSphere(center, rad, light, pxSize, res):

    """
    Question 1 (b)

    Render a sphere with a given center and radius. The camera is 
    orthographic and looks towards the sphere in the negative z
    direction. The camera's sensor axes are centerd on and aligned
    with the x- and y-axes.

    Parameters
    ----------
    center : numpy.ndarray
        The center of the hemispherical bowl in an array of size (3,)

    rad : float
        The radius of the bowl

    light : numpy.ndarray
        The direction of incoming light

    pxSize : float
        Pixel size

    res : numpy.ndarray
        The resolution of the camera frame

    Returns
    -------
    image : numpy.ndarray
        The rendered image of the hemispherical bowl
    """

    image = np.zeros(shape=res, dtype=np.uint8)
    img_center = res/2

    for xind in range(0, res[0]):
        x = (xind - img_center[0])*pxSize
        for yind in range(0, res[1]):
            y = (img_center[1] - yind)*pxSize
            rhs = rad**2 - (x-center[0])**2 - (y-center[1])**2
            if rhs >= 0:
                z = np.sqrt(rhs) + center[2]
                dir = np.array([x,y,z]) - center
                surf_rad = 1/np.pi * 255 * np.dot(dir, light)
                # surf_rad = 255*np.dot(dir, light)
                surf_rad = max(surf_rad, 0)

                image[xind][yind] = surf_rad


    return image


def loadData(path = "../data/"):

    """
    Question 1 (c)

    Load data from the path given. The images are stored as input_n.tif
    for n = {1...7}. The source lighting directions are stored in
    sources.mat.

    Paramters
    ---------
    path: str
        Path of the data directory

    Returns
    -------
    I : numpy.ndarray
        The 7 x P matrix of vectorized images

    L : numpy.ndarray
        The 3 x 7 matrix of lighting directions

    s: tuple
        Image shape

    """


    luminance_images = []
    for i in range(1,8):
        img = cv2.imread(path+"input_"+str(i)+".tif", cv2.IMREAD_UNCHANGED)
        xyz_img =rgb2xyz(img)

        luminance = xyz_img[:, :, 1]
        luminance_images.append(luminance.flatten())

        s = img.shape[:2]

    I = np.stack(luminance_images)

    L = np.load(os.path.join(path, 'sources.npy'))
    L = np.transpose(L)

    return I, L, s



def estimatePseudonormalsCalibrated(I, L):

    """
    Question 1 (e)

    In calibrated photometric stereo, estimate pseudonormals from the
    light direction and image matrices

    Parameters
    ----------
    I : numpy.ndarray
        The 7 x P array of vectorized images

    L : numpy.ndarray
        The 3 x 7 array of lighting directions

    Returns
    -------
    B : numpy.ndarray
        The 3 x P matrix of pesudonormals
    """

    B = np.linalg.lstsq(L.T, I, rcond=None)[0]

    return B


def estimateAlbedosNormals(B):

    '''
    Question 1 (e)

    From the estimated pseudonormals, estimate the albedos and normals

    Parameters
    ----------
    B : numpy.ndarray
        The 3 x P matrix of estimated pseudonormals

    Returns
    -------
    albedos : numpy.ndarray
        The vector of albedos

    normals : numpy.ndarray
        The 3 x P matrix of normals
    '''

    albedos = np.linalg.norm(B, axis=0)
    normals = B / albedos

    return albedos, normals


def displayAlbedosNormals(albedos, normals, s):

    """
    Question 1 (f)

    From the estimated pseudonormals, display the albedo and normal maps

    Please make sure to use the `gray` colormap for the albedo image
    and the `rainbow` colormap for the normals.

    Parameters
    ----------
    albedos : numpy.ndarray
        The vector of albedos

    normals : numpy.ndarray
        The 3 x P matrix of normals

    s : tuple
        Image shape

    Returns
    -------
    albedoIm : numpy.ndarray
        Albedo image of shape s

    normalIm : numpy.ndarray
        Normals reshaped as an s x 3 image

    """

    albedoIm = np.reshape(albedos, s)

    normalIm = np.dstack(np.split(normals, 3, axis=0))
    normalIm = normalIm.reshape(s[0], s[1], 3)

    plt.imshow(albedoIm, cmap='gray')
    plt.show()

    plt.imshow(normalIm, cmap='rainbow')
    plt.show()


    return albedoIm, normalIm 


def estimateShape(normals, s):

    """
    Question 1 (i)

    Integrate the estimated normals to get an estimate of the depth map
    of the surface.

    Parameters
    ----------
    normals : numpy.ndarray
        The 3 x P matrix of normals

    s : tuple
        Image shape

    Returns
    ----------
    surface: numpy.ndarray
        The image, of size s, of estimated depths at each point

    """

    df_dx = np.reshape(normals[0,:]/normals[2,:], s)
    df_dy = np.reshape(normals[1,:]/normals[2,:], s)

    surface = integrateFrankot(df_dx, df_dy)

    return surface



def plotSurface(surface):

    """
    Question 1 (i) 

    Plot the depth map as a surface

    Parameters
    ----------
    surface : numpy.ndarray
        The depth map to be plotted

    Returns
    -------
        None

    """


    x, y = np.arange(surface.shape[1]), np.arange(surface.shape[0])
    X, Y = np.meshgrid(x, y)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, surface, edgecolor='none', cmap=cm.coolwarm)
    ax.set_title('Face Surface')
    plt.show()


if __name__ == '__main__':

    # Put your main code here

    pxSize = 7e-04  # in micrometers
    res = np.array([2160, 3840])
    center = np.array([0,0,0])
    rad = 0.75

    light1 = np.array([1,1,1])/np.sqrt(3)
    light2 = np.array([1,-1,1])/np.sqrt(3)
    light3 = np.array([-1,-1,1])/np.sqrt(3)


    # rendered_image = renderNDotLSphere(center, rad, light1, pxSize, res)
    # plt.imshow(rendered_image, cmap="gray")
    # plt.show()

    # rendered_image = renderNDotLSphere(center, rad, light2, pxSize, res)
    # plt.imshow(rendered_image, cmap="gray")
    # plt.show()

    # rendered_image = renderNDotLSphere(center, rad, light3, pxSize, res)
    # plt.imshow(rendered_image, cmap="gray")
    # plt.show()


    I, L, s = loadData()
    # print(f"I: {I.shape} || L: {L.shape} || s: {s}")


    # question 1d
    u, sing, vh = np.linalg.svd(I,full_matrices=False)
    print(f"question 1c: singluar values = {sing}")

    B = estimatePseudonormalsCalibrated(I, L)
    # print(f"B : {B.shape}")
    
    albedos, normals = estimateAlbedosNormals(B)
    # print(f"albedos: {albedos.shape} || normals: {normals.shape}")

    albedoIm, normalIm = displayAlbedosNormals(albedos, normals, s)

    surface=estimateShape(normals, s)
    minv, maxv = np.min(surface), np.max(surface)
    surface = (surface - minv) / (maxv - minv)
    plotSurface(surface)





























'''

Previously Implemented Code:






loadData:
luminance_images = []

    # Loop through each image in the directory
    for filename in os.listdir(path):
        if filename.endswith(".tif"):
            img_path = os.path.join(path, filename)
            # img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            img = cv2.imread(img_path, -1)

            s = img.shape[:2]

            # Convert to XYZ color space
            xyz_img =rgb2xyz(img)

            # Extract luminance channel (Y channel of XYZ)
            luminance = xyz_img[:, :, 1]  # Assuming Y channel is at index 1
            luminance = luminance.reshape(luminance.shape[0]*luminance.shape[1],1)

            # Vectorize and store the luminance image
            # luminance_images.append(luminance.flatten())
            luminance_images.append(luminance)

    # I = np.stack(luminance_images)
    A = np.array(luminance_images)
    I = A.reshape(A.shape[0],A.shape[1])

    L = np.load(os.path.join(path, 'sources.npy'))
    L = np.transpose(L)

    return I, L, s




# get the image points array
x_positions = np.arange(0, res[0] * pxSize, pxSize)
y_positions = np.arange(0, res[1] * pxSize, pxSize)

X, Y = np.meshgrid(x_positions, y_positions)
pixel_positions = np.vstack([X.ravel(), Y.ravel()]).T

x2y2 = np.sum((pixel_positions - center[:2])**2, axis=1)
valid_points = x2y2[x2y2 <= rad**2]

x_points = pixel_positions[x2y2 <= rad**2][:,0]
y_points = pixel_positions[x2y2 <= rad**2][:,1]
z_points = np.sqrt(rad**2 - valid_points) + center[2]

sphere_points = np.vstack([x_points.ravel(), y_points.ravel(), z_points.ravel()]).T

n_dirs = sphere_points - center
n_dirs /= np.linalg.norm(n_dirs, axis=1)

surface_radiance = 1/np.pi * 255 * np.dot(n_dirs, light)

image = np.zeros(shape=res, dtype=np.uint8)





'''