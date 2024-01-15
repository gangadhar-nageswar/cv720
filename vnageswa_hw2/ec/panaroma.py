from turtle import shape
import numpy as np
import cv2
#Import necessary functions

import skimage.io 
import skimage.color

import sys
sys.path.append("../python")
from matchPics import matchPics
from helper import plotMatches
from opts import get_opts

#Import necessary functions
from planarH import computeH_ransac, compositeH

#Write script for Q2.2.4
opts = get_opts()


pano_left = cv2.imread('../data/cat_left.JPG')
pano_right = cv2.imread('../data/cat_right.JPG')

pano_left = cv2.resize(pano_left, (1457, 1080))
pano_right = cv2.resize(pano_right, (1457, 1080))


dst_img = pano_left
src_img = pano_right

matches, locs1, locs2 = matchPics(dst_img, src_img, opts)

# swap the positions of locs
locs1[:, [1, 0]] = locs1[:, [0, 1]]
locs2[:, [1, 0]] = locs2[:, [0, 1]]

x1 = np.ndarray(shape=(matches.shape[0],2))
x2 = np.ndarray(shape=(matches.shape[0],2))

for i in range(matches.shape[0]):
	ind1, ind2 = matches[i][0], matches[i][1]
	x1[i][0], x1[i][1] = locs1[ind1][0], locs1[ind1][1]
	x2[i][0], x2[i][1] = locs2[ind2][0], locs2[ind2][1]

H_computed, inliers = computeH_ransac(x1, x2, opts)
H_opencv, mask = cv2.findHomography(x1, x2, 0)

destHeight, destWidth, _ = dst_img.shape
srcHeight, srcWidth, _ = src_img.shape

# compute the left edge points
top_right = np.matmul(H_computed, np.array([[srcWidth-1],[0],[1]]))
bottom_right = np.matmul(H_computed, np.array([[srcWidth-1],[srcHeight-1],[1]]))

top_right[0] /= top_right[2]
bottom_right[0] /= bottom_right[2]
print(f"Homographised x: bottom right={bottom_right[0]} || top right: {top_right[0]}")

finalWidth = int(min(bottom_right[0], top_right[0]))
finalHeight = destHeight
final_img = np.ndarray(shape=(finalHeight, finalWidth, 3))

# Create a new white canvas
ext_pano_left = np.zeros((finalHeight, finalWidth, 3), dtype=np.uint8)
ext_pano_left.fill(255)  # Fill with white color (255, 255, 255)
ext_pano_left[:, :srcWidth,:] = pano_left

# mask --> warp --> composite image
gray_template = cv2.cvtColor(pano_right, cv2.COLOR_BGR2GRAY)
mask = cv2.threshold(gray_template, 0, 255, cv2.THRESH_BINARY)[1]

#Warp mask by appropriate homography
mask = cv2.warpPerspective(mask,H_computed,(finalWidth, finalHeight))

#Warp template by appropriate homography
H_pano_right = cv2.warpPerspective(pano_right,H_computed,(finalWidth, finalHeight), borderValue=1)

#Use mask to combine the warped template and the image
mask_inv = cv2.bitwise_not(mask)

composite_img = cv2.bitwise_and(H_pano_right, H_pano_right, mask=mask)
composite_img += cv2.bitwise_and(ext_pano_left, ext_pano_left, mask=mask_inv)

cv2.imshow('final composite img',composite_img)
cv2.waitKey(0)