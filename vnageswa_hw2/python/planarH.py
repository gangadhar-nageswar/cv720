from cmath import sqrt
from distutils.log import error
from json.encoder import INFINITY
from turtle import shape
import numpy as np
import cv2
from matchPics import matchPics
from helper import plotMatches
from opts import get_opts
import scipy
from numpy.linalg import inv


def computeH(x1, x2):
	#Q2.2.1
	#Compute the homography between two sets of points

	# matrix A will be of 2N x 9 size
	temp = np.concatenate((x2,np.ones(shape=(x2.shape[0],1))), axis=1)

	a1 = np.concatenate((temp,np.zeros(shape=temp.shape)), axis=1)
	a1 = np.reshape(a1, (2*x2.shape[0], x2.shape[1]+1))
	
	a2 = np.concatenate((np.zeros(shape=temp.shape),temp), axis=1)
	a2 = np.reshape(a2, (2*x2.shape[0], x2.shape[1]+1))

	temp = np.concatenate((temp, temp), axis=1)
	temp = np.reshape(temp, (2*x2.shape[0], x2.shape[1]+1))

	x1_reshaped = np.reshape(x1, (2*x1.shape[0], 1))
	a3 = -x1_reshaped*temp

	A = np.concatenate((a1, a2, a3), axis=1)

	U, S, Vh = np.linalg.svd(A, full_matrices=True)
	H2to1 = Vh[-1,:]
	H2to1 = H2to1.reshape((3,3))

	return H2to1


def computeH_norm(x1, x2):
	#Q2.2.2
	# convert the Nx2 matrices to homogoneous coordinates system
	n1 = x1.shape[0]
	x1 = np.concatenate((x1, np.ones(shape=(n1,1))), axis=1)

	n2 = x2.shape[0]
	x2 = np.concatenate((x2, np.ones(shape=(n2,1))), axis=1)
	
	#Compute the centroid of the points
	cx1, cy1 = [x1[:,0].mean(), x1[:,1].mean()]
	cx2, cy2 = [x2[:,0].mean(), x2[:,1].mean()]

	#Normalize the points so that the largest distance from the origin is equal to sqrt(2)
	xmax1, ymax1 = [x1[:,0].max(), x1[:,1].max()]
	xmax2, ymax2 = [x2[:,0].max(), x2[:,1].max()]

	#Similarity transform 1
	T1 = np.array([[1/xmax1, 0, -cx1/xmax1], [0, 1/ymax1, -cy1/ymax1], [0, 0, 1]])
	x1_norm = np.matmul(T1, x1.transpose()).transpose()
	x1_norm = np.delete(x1_norm, -1, 1)

	#Similarity transform 2
	T2 = np.array([[1/xmax2, 0, -cx2/xmax2], [0, 1/ymax2, -cy2/ymax2], [0, 0, 1]])
	x2_norm = np.matmul(T2, x2.transpose()).transpose()
	x2_norm = np.delete(x2_norm, -1, 1)
	
	#Compute homography
	H_norm = computeH(x1_norm, x2_norm)

	#Denormalization
	H = np.matmul(inv(T1), H_norm)
	H2to1 = np.matmul(H, T2)
	
	return H2to1




def computeH_ransac(locs1, locs2, opts):
	#Q2.2.3
	#Compute the best fitting homography given a list of matching points
	max_iters = opts.max_iters  # the number of iterations to run RANSAC for
	inlier_tol = opts.inlier_tol # the tolerance value for considering a point to be an inlier

	n = locs1.shape[0]
	print(f"n = {n}")
	print(f"shape of locs1: {locs1.shape} || locs2: {locs2.shape}")
	
	max_inliers = 0
	best_inliers = [False for _ in range(n)]
	H_best = None

	for i in range(max_iters):
		inds = np.random.randint(0, high=n, size=4)
		# print(f"indices sampled: {inds}")
		feats1, feats2 = locs1[inds,:], locs2[inds,:]

		H_computed = computeH_norm(feats1, feats2)

		# compute the homographies of locs2
		locs2_new = np.concatenate((locs2, np.ones(shape=(locs2.shape[0],1))), axis=1)
		locs1_computed = np.matmul(locs2_new, H_computed.transpose())

		locs1_computed[:,0] = (locs1_computed[:,0]/locs1_computed[:,2])
		locs1_computed[:,1] = (locs1_computed[:,1]/locs1_computed[:,2])

		locs2_new = np.delete(locs2_new, -1, 1)
		locs1_computed = np.delete(locs1_computed, -1, 1)

		# compute the error
		e = np.sqrt(np.square(locs1_computed - locs1).sum(axis=1))
		inliers_bool = e < inlier_tol
		inliers_inds = np.where(inliers_bool)[0]
		count_inls = len(inliers_inds)

		if count_inls > max_inliers:	
			max_inliers = count_inls
			best_inliers = inliers_bool
			H_best = H_computed
	

	# best_inliers_inds = np.where(best_inliers)[0]
	# locs1_best, locs2_best = locs1[best_inliers_inds,:], locs2[best_inliers_inds,:]
	# bestH2to1 = computeH_norm(locs1_best, locs2_best)
	
	bestH2to1 = H_best
	inliers = best_inliers


	return bestH2to1, inliers



def compositeH(H2to1, template, img):
	
	#Create a composite image after warping the template image on top
	#of the image using the homography

	#Note that the homography we compute is from the image to the template;
	#x_template = H2to1*x_photo
	#For warping the template to the image, we need to invert it.
	# H2to1 = inv(H2to1)

	# !!!! - Important - !!!! #
	# inv of H2to1 is not taken since this is accounted 
	# by taking the homography according to the source and destination, i.e. x2 = src and x1 = dst

	
	imgHeight, imgWidth, _ = img.shape
	tempHeight, tempWidth, _ = template.shape

	#Create mask of same size as template
	#let the mask be consisting of ones to differentiate from the black regions in warped template
	# gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
	# mask = cv2.threshold(gray_template, 0, 255, cv2.THRESH_BINARY)[1]
	mask = np.zeros([tempHeight,tempWidth],dtype=np.uint8)
	mask.fill(255)

	#Warp mask by appropriate homography
	mask = cv2.warpPerspective(mask,H2to1,(imgWidth,imgHeight))
	
	#Warp template by appropriate homography
	template = cv2.warpPerspective(template,H2to1,(imgWidth,imgHeight))
	
	#Use mask to combine the warped template and the image
	mask_inv = cv2.bitwise_not(mask)

	composite_img = cv2.bitwise_and(template, template, mask=mask)
	composite_img += cv2.bitwise_and(img, img, mask=mask_inv)

	return composite_img


















# opts = get_opts()

# cv_cover = cv2.imread('../data/cv_cover.jpg')
# cv_desk = cv2.imread('../data/cv_desk.png')
# # cv_desk = scipy.ndimage.rotate(cv_cover, 180, reshape=False)
# # cv_desk = cv2.imread('../data/cv_cover.jpg')


# matches, locs1, locs2 = matchPics(cv_cover, cv_desk, opts)
# print(f"size of matches: {matches.shape} | locs1: {locs1.shape} | locs2: {locs2.shape}")
# # print(f"locs1: {locs1}\n")
# # print(f"locs2: {locs2}\n")
# # print(f"matches: {matches}\n")

# #display matched features
# plotMatches(cv_cover, cv_desk, matches, locs1, locs2)

# # swap the positions of locs
# locs1[:, [1, 0]] = locs1[:, [0, 1]]
# locs2[:, [1, 0]] = locs2[:, [0, 1]]

# x1 = np.ndarray(shape=(matches.shape[0],2))
# x2 = np.ndarray(shape=(matches.shape[0],2))
# arange_arr = np.expand_dims((np.arange(matches.shape[0])), axis=1)
# new_matches = np.concatenate((arange_arr, arange_arr), axis=1)

# for i in range(matches.shape[0]):
# 	ind1, ind2 = matches[i][0], matches[i][1]
# 	x1[i][0], x1[i][1] = locs1[ind1][0], locs1[ind1][1]
# 	x2[i][0], x2[i][1] = locs2[ind2][0], locs2[ind2][1]

# print(f"x1: {x1.shape} | x2: {x2.shape}")
# # H_computed, inliers = computeH_ransac(x2, x1, opts)
# H_computed = computeH(x1, x2)
# H_opencv, mask = cv2.findHomography(x1, x2, 0)
# print(H_computed)
# print(H_opencv)

# locs2_new = np.concatenate((x2, np.ones(shape=(x2.shape[0],1))), axis=1)
# locs1_computed = np.matmul(locs2_new, H_computed.transpose())

# locs1_computed[:,0] = (locs1_computed[:,0]/locs1_computed[:,2]).astype(int)
# locs1_computed[:,1] = (locs1_computed[:,1]/locs1_computed[:,2]).astype(int)

# locs2_new = np.delete(locs2_new, -1, 1)
# locs1_computed = np.delete(locs1_computed, -1, 1)

# locs1_computed[:,[0,1]] = locs1_computed[:, [1,0]]
# locs2_new[:, [0,1]] = locs2_new[:, [1,0]]

# plotMatches(cv_cover, cv_desk, new_matches, locs1_computed, locs2_new)

# maxHeight, maxWidth, C = cv_desk.shape
# warped_img = cv2.warpPerspective(cv_cover,H_computed,(maxWidth, maxHeight),flags=cv2.INTER_LINEAR)
# cv2.imshow('warped img',warped_img)
# cv2.waitKey(0)





'''


Past implmentations:

def computeH_norm(x1, x2):
	#Q2.2.2

	# #Compute the centroid of the points
	# mu_x_1, mu_y_1 = x1[:,0].mean(), x1[:,1].mean()
	# mu_x_2, mu_y_2 = x2[:,0].mean(), x2[:,1].mean()

	# #Shift the origin of the points to the centroid
	# x1_shifted = x1 - [mu_x_1, mu_y_1]
	# x2_shifted = x2 - [mu_x_2, mu_y_2]

	#Normalize the points so that the largest distance from the origin is equal to sqrt(2)
	xmax1, ymax1 = x1[:,0].max(), x1[:,1].max()
	xmax2, ymax2 = x2[:,0].max(), x2[:,1].max()
	lambda1 = sqrt(xmax1**2 + ymax1**2)
	lambda2 = sqrt(xmax2**2 + ymax2**2)

	#Similarity transform 1
	n1 = x1.shape[0]
	T1 = (np.identity(n1) - 1/n1*np.ones(shape=(n1,n1)))/lambda1
	x1_norm  = np.matmul(T1, x1)

	#Similarity transform 2
	n2 = x2.shape[0]
	T2 = (np.identity(n2) - 1/n2*np.ones(shape=(n2,n2)))/lambda2
	x2_norm  = np.matmul(T2, x2)

	#Compute homography
	H_norm = computeH(x1_norm, x2_norm)

	#Denormalization
	H = np.matmul(inv(T1), H_norm)
	H2to1 = np.matmul(H, T2)
	
	return H2to1

'''