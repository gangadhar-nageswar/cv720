    '''
Q4.2:
    1. Integrating everything together.
    2. Loads necessary files from ../data/ and visualizes 3D reconstruction using scatter
'''

import submission as sub
from helper import *
import numpy as np
import matplotlib.pyplot as plt

data = np.load('../data/some_corresp.npz')
im1 = plt.imread('../data/im1.png')
im2 = plt.imread('../data/im2.png')

print(f"shape of im1: {im1.shape}")
print(f"shape of im2: {im2.shape}")

N = data['pts1'].shape[0]
pts1 = data['pts1']
pts2 = data['pts2']

M = 640

# 2.1
F8 = sub.eightpoint(data['pts1'], data['pts2'], M)
print(f"\n\nF8: {F8}\n\n")


# 3.1
K = np.load('../data/intrinsics.npz')
K1, K2 = K['K1'], K['K2']
E = sub.essentialMatrix(F8, K1, K2)

print(f"\nE : {E}")

M1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
M2s = camera2(E)

M2 = M2s[:,:,0]

C1 = K1@M1
C2 = K2@M2

####### 3d visualize ######
data = np.load('../data/templeCoords.npz')
x1_data = data["x1"]
y1_data = data["y1"]
print(f"shape of x1_data: {x1_data.shape}")
N = x1_data.shape[0]

# find epipolar correspondences
x2_data = np.zeros(shape=(N,1))
y2_data = np.zeros(shape=(N,1))

for i in range(N):
    x2_data[i,0], y2_data[i,0] = sub.epipolarCorrespondence(im1, im2, F8, x1_data[i,0], y1_data[i,0])

pts1 = np.concatenate((x1_data, y1_data), axis=1)
pts2 = np.concatenate((x2_data, y2_data), axis=1)

points_3d, err = sub.triangulate(C1, pts1, C2, pts2)

print(f"reprojection error final: {err}")

# np.savez('q4_2.npz', F=F8, M1=M1, M2=M2, C1=C1, C2=C2)

# Create a figure and 3D subplot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(points_3d[:,0], points_3d[:,1], points_3d[:,2])
plt.show()