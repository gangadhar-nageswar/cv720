import numpy as np
import cv2
import skimage.io 
import skimage.color
from matchPics import matchPics
from helper import plotMatches
from opts import get_opts

#Import necessary functions
from planarH import computeH_ransac, compositeH


#Write script for Q2.2.4
opts = get_opts()


cv_cover = cv2.imread('../data/cv_cover.jpg')
cv_desk = cv2.imread('../data/cv_desk.png')
cv_hp = cv2.imread('../data/hp_cover.jpg')


matches, locs1, locs2 = matchPics(cv_cover, cv_desk, opts)
# print(f"size of matches: {matches.shape} | locs1: {locs1.shape} | locs2: {locs2.shape}")

# swap the positions of locs
locs1[:, [1, 0]] = locs1[:, [0, 1]]
locs2[:, [1, 0]] = locs2[:, [0, 1]]

x1 = np.ndarray(shape=(matches.shape[0],2))
x2 = np.ndarray(shape=(matches.shape[0],2))
arange_arr = np.expand_dims((np.arange(matches.shape[0])), axis=1)
new_matches = np.concatenate((arange_arr, arange_arr), axis=1)

for i in range(matches.shape[0]):
	ind1, ind2 = matches[i][0], matches[i][1]
	x1[i][0], x1[i][1] = locs1[ind1][0], locs1[ind1][1]
	x2[i][0], x2[i][1] = locs2[ind2][0], locs2[ind2][1]

# print(f"x1: {x1.shape} | x2: {x2.shape}")
H_computed, inliers = computeH_ransac(x2, x1, opts)
# H_computed = computeH(x1, x2)
H_opencv, mask = cv2.findHomography(x2, x1, 0)

destHeight, destWidth, _ = cv_desk.shape
srcHeight, srcWidth, _ = cv_cover.shape

#resize hp image to the size of cv_cover
cv_hp = cv2.resize(cv_hp, (srcWidth, srcHeight))

comp_img = compositeH(H_computed, cv_hp, cv_desk)
cv2.imshow('composite img',comp_img)
cv2.waitKey(0)