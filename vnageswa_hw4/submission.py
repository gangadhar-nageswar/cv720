"""
Homework4.
Replace 'pass' by your implementation.
"""

# Insert your package here


'''
Q2.1: Eight Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix
'''
from cmath import sqrt
from json.encoder import INFINITY
from statistics import variance
from turtle import shape
from util import *
import numpy as np


def eightpoint(pts1, pts2, M):
    # Normalise the points
    cx, cy = 0, 0
    xmax, ymax = M, M

	#Similarity transform
    T = np.array([[1/xmax, 0, -cx/xmax], [0, 1/ymax, -cy/ymax], [0, 0, 1]])

    pts1_norm = (pts1 - [cx,cy])/M
    pts2_norm = (pts2 - [cx,cy])/M

    a1 = pts1_norm * np.expand_dims(pts2_norm[:,0], axis=1)
    a2 = np.expand_dims(pts2_norm[:,0], axis=1)
    a3 = pts1_norm * np.expand_dims(pts2_norm[:,1], axis=1)
    a4 = np.expand_dims(pts2_norm[:,1], axis=1)

    U_F8 = np.concatenate((a1, a2, a3, a4, pts1_norm, np.ones(shape=(pts1_norm.shape[0],1))), axis=1)
    _, _, V_U_F8 = np.linalg.svd(U_F8, full_matrices=True)
    F8_norm = V_U_F8[-1,:]
    F8_norm = F8_norm.reshape((3,3))

    # set the last eigenvalue of F8 to zero and recompute
    # W_F8, S_F8, V_F8 = np.linalg.svd(F8_norm, full_matrices=True)
    # S_F8[-1] = 0
    # F8_norm = (W_F8@np.diag(S_F8))@V_F8

    U, S, V = np.linalg.svd(F8_norm)
    S[-1] = 0
    # F8_norm = U.dot(np.diag(S).dot(V))
    F8_norm = U@(np.diag(S)@V)

    F8_norm = refineF(F8_norm, pts1_norm, pts2_norm)

    F8 = np.transpose(T) @ F8_norm @ T
    
    return F8


'''
Q3.1: Compute the essential matrix E.
    Input:  F, fundamental matrix
            K1, internal camera calibration matrix of camera 1
            K2, internal camera calibration matrix of camera 2
    Output: E, the essential matrix
'''
def essentialMatrix(F, K1, K2):
    E = np.transpose(K2) @ F @ K1
    return E


'''
Q3.2: Triangulate a set of 2D coordinates in the image to a set of 3D points.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx2 matrix with the 2D image coordinates per row
            C2, the 3x4 camera matrix
            pts2, the Nx2 matrix with the 2D image coordinates per row
    Output: P, the Nx3 matrix with the corresponding 3D points per row
            err, the reprojection error.
'''


def _get_reprojections(C, points):
    '''
    Input: inhomogenous points Nx3
    Return: inhomogenous camera points Nx2
    '''
    homogenous_3d_points = np.concatenate((points, np.ones(shape=(points.shape[0],1))), axis=1)

    reprojected_points = homogenous_3d_points@np.transpose(C)
    reprojected_points[:,0] = reprojected_points[:,0]/reprojected_points[:,2]
    reprojected_points[:,1] = reprojected_points[:,1]/reprojected_points[:,2]
    reprojected_points = np.delete(reprojected_points, -1, 1)

    return reprojected_points


def triangulate(C1, pts1, C2, pts2):
    N = pts1.shape[0]
    points_3d = np.zeros(shape=(N,3))

    reprojection_error = 0

    for i in range(N):
        # construct the A matrix
        a1r = np.expand_dims(pts1[i,1]*C1[2,:] - C1[1,:], axis=0)
        a2r = np.expand_dims(C1[0,:] - pts1[i,0]*C1[2,:], axis=0)
        a3r = np.expand_dims(pts2[i,1]*C2[2,:] - C2[1,:], axis=0)
        a4r = np.expand_dims(C2[0,:] - pts2[i,0]*C2[2,:], axis=0)

        A = np.concatenate((a1r, a2r, a3r, a4r), axis=0)

        # print(f"size of A: {A.shape}")

        _, _, V = np.linalg.svd(A, full_matrices=True)
        pointX = V[-1,:]
        # print(f"shape of pointX: {pointX.shape}")
        # print(f"pointX last ele: {[pointX[3]]}")
        points_3d[i,0], points_3d[i,1], points_3d[i,2] = pointX[0]/pointX[3], pointX[1]/pointX[3], pointX[2]/pointX[3]

    # print(f"points3d: {points_3d}")
    # Compute reprojection error
    reprojected_points1 = _get_reprojections(C1, points_3d)
    reprojected_points2 = _get_reprojections(C2, points_3d)

    reprojection_error1 = np.sum((np.sum((pts1 - reprojected_points1)**2, axis=1)), axis=0)
    reprojection_error2 = np.sum((np.sum((pts2 - reprojected_points2)**2, axis=1)), axis=0)

    reprojection_error = reprojection_error1 + reprojection_error2
    
    return points_3d, reprojection_error


'''
Q4.1: 3D visualization of the temple images.
    Input:  im1, the first image
            im2, the second image
            F, the fundamental matrix
            x1, x-coordinates of a pixel on im1
            y1, y-coordinates of a pixel on im1
    Output: x2, x-coordinates of the pixel on im2
            y2, y-coordinates of the pixel on im2

'''

def _createGaussianFilter(window_size, variance):
    center = (window_size - 1) / 2
    variance = 4.0

    gauss_x, gauss_y = np.meshgrid(np.arange(window_size), np.arange(window_size))

    distances_squared = (gauss_x - center)**2 + (gauss_y - center)**2
    gaussian_weights = np.exp(-distances_squared / (2 * variance**2))
    gaussian_weights /= np.max(gaussian_weights)
    gaussian_weights = np.dstack((gaussian_weights, gaussian_weights, gaussian_weights))

    return gaussian_weights


def epipolarCorrespondence(im1, im2, F, x1, y1):
    nx, ny = im2.shape[1], im2.shape[0]

    # obtain the line parameters
    a = np.transpose(F[0,:])@np.array([x1,y1,1])
    b = np.transpose(F[1,:])@np.array([x1,y1,1])
    c = np.transpose(F[2,:])@np.array([x1,y1,1])

    window_size = 19
    variance = 6
    lookup_size = int((window_size-1)/2)
    gaussian_weights = _createGaussianFilter(window_size, variance)
    # gaussian_weights = np.ones(shape=(window_size,window_size,3))


    # if the chosen point is near to edges; return the same point
    if x1-lookup_size<0 or x1+lookup_size>nx or y1-lookup_size<0 or y1+lookup_size>ny:
        return x1, y1

    # check for all points around the epipolar line
    min_cost = INFINITY
    best_match = None

    for yr in range(ny):
        xr = int(-(b*yr + c)/a)
        if xr < 0 or xr > nx or xr-lookup_size < 0 or xr+lookup_size+1> nx:
            continue
        if yr-lookup_size < 0 or yr+lookup_size+1 > ny:
            continue
        
        ''' THE CURRENT CODE ASSUMES NO PADDING ==> WILL BE IMPLEMENTED IN THE FUTURE'''
        wind1 = im1[y1-lookup_size:y1+lookup_size+1, x1-lookup_size:x1+lookup_size+1, :]
        wind2 = im2[yr-lookup_size:yr+lookup_size+1, xr-lookup_size:xr+lookup_size+1, :]

        # cost = np.sum(np.multiply(gaussian_weights_3, (wind1 - wind2)**2))
        cost = np.sum(np.multiply(gaussian_weights, abs(wind1 - wind2))) + 0.2*sqrt((x1-xr)**2 + (y1-yr)**2)

        # print(f"cost1: {np.sum(np.multiply(gaussian_weights, abs(wind1 - wind2)))} || cost2: {0.05*sqrt((x1-xr)**2 + (y1-yr)**2)}")
        # print(f"cost: {cost}")
        
        if cost < min_cost:
            best_match = [xr, yr]
            min_cost = cost


    return best_match[0], best_match[1]



'''
Q5.1: Extra Credit RANSAC method.
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scaler parameter
    Output: F, the fundamental matrix
            inliers, Nx1 bool vector set to true for inliers
'''

def _get_Distance_From_EpipolarLine(pt1, pt2, F):
    line_eq = F@np.array([[pt1[0]],[pt1[1]],[1]])
    pointLine_dist = abs(np.array([[pt2[1], pt2[2], 1]]) @ line_eq) / np.linalg.norm(line_eq)

    return pointLine_dist


def ransacF(pts1, pts2, M, nIters=1000, tol=0.42):

    locs1 = np.copy(pts1)
    locs2 = np.copy(pts2)

    max_iters = nIters
    inlier_tol = tol

    n = locs1.shape[0]
    print(f"n = {n}")
    print(f"shape of locs1: {pts1.shape} || locs2: {pts2.shape}")
    
    max_inliers = 0
    best_inliers = [False for _ in range(n)]
    F_best = None
    count_sum = 0

    for i in range(max_iters):
        print(f"\n\n =========== iter: {i} ========= \n\\n")
        inds = np.random.randint(0, high=n, size=8)
        # print(f"indices sampled: {inds}")
        feats1, feats2 = locs1[inds,:], locs2[inds,:]

        F_computed = eightpoint(feats1, feats2, M)

        # compute the epipolar lines matrix
        locs1_new = np.concatenate((locs1, np.ones(shape=(locs1.shape[0],1))), axis=1)
        locs2_new = np.concatenate((locs2, np.ones(shape=(locs2.shape[0],1))), axis=1)

        epipolar_lines = np.matmul(locs1_new, F_computed.transpose())
        
        dists_den = np.sqrt(epipolar_lines[:,0]**2 + epipolar_lines[:,1]**2)
        dists_num = np.sum(np.multiply(locs2_new, epipolar_lines), axis=1)

        dists = abs(dists_num)/dists_den

        # print(f"shape of dists: {dists.shape}")
        # print(f"max of dists: {np.max(dists)}")

        # compute the error
        inliers_bool = dists < inlier_tol
        inliers_inds = np.where(inliers_bool)[0]
        count_inls = len(inliers_inds)

        if count_inls > max_inliers:	
            max_inliers = count_inls
            best_inliers = inliers_bool
            F_best = F_computed
    
    best_inliers_inds = np.where(best_inliers)[0]
    locs1_best, locs2_best = locs1[best_inliers_inds,:], locs2[best_inliers_inds,:]
    F_best = eightpoint(locs1_best, locs2_best, M)


    return F_best, best_inliers



'''
Q5.2:Extra Credit  Rodrigues formula.
    Input:  r, a 3x1 vector
    Output: R, a rotation matrix
'''
def rodrigues(r):
    theta_r = np.linalg.norm(r)

    # obtain axis
    k = r/theta_r
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])

    R = np.eye(3) + np.sin(theta_r)*K + (1-np.cos(theta_r))*K@K

    return R

'''
Q5.2:Extra Credit  Inverse Rodrigues formula.
    Input:  R, a rotation matrix
    Output: r, a 3x1 vector
'''
def invRodrigues(R):
    theta_r = np.arccos((np.trace(R)-1)/2)
    omega_r = 1/(2*np.sin(theta_r)) * np.array([[R[2,1] - R[1,2]], [R[0,2] - R[2,0]], [R[1,0] - R[0,1]]])

    r = theta_r*omega_r

    return r

'''
Q5.3: Extra Credit Rodrigues residual.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2, the intrinsics of camera 2
            p2, the 2D coordinates of points in image 2
            x, the flattened concatenationg of P, r2, and t2.
    Output: residuals, 4N x 1 vector, the difference between original and estimated projections
'''
def rodriguesResidual(K1, M1, p1, K2, p2, x):
    # Replace pass by your implementation
    pass

'''
Q5.3 Extra Credit  Bundle adjustment.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2,  the intrinsics of camera 2
            M2_init, the initial extrinsics of camera 1
            p2, the 2D coordinates of points in image 2
            P_init, the initial 3D coordinates of points
    Output: M2, the optimized extrinsics of camera 1
            P2, the optimized 3D coordinates of points
'''
def bundleAdjustment(K1, M1, p1, K2, M2_init, p2, P_init):
    # Replace pass by your implementation

    pass
