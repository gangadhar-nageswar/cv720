'''
Q3.3:
    1. Load point correspondences
    2. Obtain the correct M2
    3. Save the correct M2, C2, and P to q3_3.npz
'''

import submission as sub
from helper import *
import numpy as np
import matplotlib.pyplot as plt



def _isInfront(M1, M2, points_3d):
    homogenous_3d_points = np.concatenate((points_3d, np.ones(shape=(points_3d.shape[0],1))), axis=1)
    points1 = homogenous_3d_points@np.transpose(M1)
    points2 = homogenous_3d_points@np.transpose(M2)

    num_points = points1.shape[0]

    if (sum(points1[:, 2] > 0) == num_points) and (sum(points2[:, 2] > 0) == num_points):
        return True
    else:
        return False


def getCorrentM2(pts1, pts2, K1, K2, M):
    F8 = sub.eightpoint(pts1, pts2, M)
    # F8, inls = sub.ransacF(pts1, pts2, M, nIters=1000, tol=0.42)

    E = sub.essentialMatrix(F8, K1, K2)
    M1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    M2s = camera2(E)

    correct_M2 = M2s[:,:,0]

    for m2_ind in range(4):
        C1 = K1@M1
        C2 = K2@M2s[:,:,m2_ind]

        points_3d, err = sub.triangulate(C1, pts1, C2, pts2)

        print(f"M2 ind: {m2_ind} || reprojection error: {err}")

        if _isInfront(M1, M2s[:,:,m2_ind], points_3d) == True:
            correct_M2 = M2s[:,:,m2_ind]
            correct_P = points_3d

            print(f"\n\nM2 ind: {m2_ind}")

    C2 = K2 @ correct_M2

    # np.savez('q3_3.npz', M2=correct_M2, C2=C2, P=correct_P)


    return correct_M2






if __name__ == "__main__":
    # data = np.load('../data/some_corresp_noisy.npz')
    data = np.load('../data/some_corresp.npz')

    N = data['pts1'].shape[0]
    pts1 = data['pts1']
    pts2 = data['pts2']

    K = np.load('../data/intrinsics.npz')
    K1, K2 = K['K1'], K['K2']

    M = 640

    M2 = getCorrentM2(pts1, pts2, K1, K2, M)

    print(f"correct M2: \n{M2}")
    print(f"correct C2: \n{K2@M2}")











    # F8, inls = sub.ransacF(pts1, pts2, M, nIters=1000, tol=0.42)
    # print(f"sum of inls: {sum(inls)}")
    # print(f"\n\nF8: {F8}\n\n")


    # # 3.1
    # K = np.load('../data/intrinsics.npz')
    # K1, K2 = K['K1'], K['K2']
    # E = sub.essentialMatrix(F8, K1, K2)

    # print(f"\nE : {E}")

    # M1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    # M2s = camera2(E)

    # # # camera2 possibility 1
    # # C1 = K1@M1
    # # C2 = K2@M2s[:,:,0]

    # # points_3d, err = sub.triangulate(C1, pts1, C2, pts2)

    # for m2_ind in range(4):
    #     C1 = K1@M1
    #     C2 = K2@M2s[:,:,m2_ind]

    #     points_3d, err = sub.triangulate(C1, pts1, C2, pts2)
    #     print(f"camera ind: {m2_ind} || isInfront: {isInfront(M1, M2s[:,:,m2_ind], points_3d)}")