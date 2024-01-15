import sys
sys.path.append("../python")
from plistlib import load
from turtle import end_fill, shape
import numpy as np
import cv2
import skimage.color
import skimage.feature
import scipy.io as sio

#Import necessary functions
from loadVid import loadVid
from matchPics import matchPics
from helper import plotMatches
from opts import get_opts
from planarH import computeH_ransac, compositeH
from helper import briefMatch
from helper import computeBrief
from helper import corner_detection

import time

import multiprocessing
n_cpu = multiprocessing.cpu_count()

#Write script for Q4.1x
opts = get_opts()

#Write script for Q3.1
src_path = "../data/ar_source.mov"
dst_path = "../data/book.mov"


# #Load the videos
src_video = loadVid(src_path)
dst_video = loadVid(dst_path)


# # save the video as numpy arrays for future purposes
# np.save("../data/src_video.npy", src_video)
# np.save("../data/dst_video.npy", dst_video)

# load the video numpy images
# start_time = time.time()
# src_video = np.load("../data/src_video.npy")
# dst_video = np.load("../data/dst_video.npy")
# end_time = time.time()

# print(f"time to video numpy arrays: {end_time - start_time}")

# print(f"source video shape: {src_video.shape}")
# print(f"dst video shape: {dst_video.shape}")

nf1, nf2 = src_video.shape[0], dst_video.shape[0]
N = min(nf1, nf2)

#Load images requried for computing homography
cv_cover = cv2.imread('../data/cv_cover.jpg')
ref_frame = cv_cover

#Crop and resize the source image to the cv_cover size
refHeight, refWidth, _ = cv_cover.shape
srcHeight, srcWidth = src_video.shape[1], src_video.shape[2]
dstHeight, dstWidth = dst_video.shape[1], dst_video.shape[2]

#Crop the middle section of the Video and remove the black portions in the video
src_video = src_video[:,44:-44,srcWidth//3:2*srcWidth//3,:]
resized_frames = []

for i in range(N):
    resized_frames.append(cv2.resize(src_video[i], (refWidth, refHeight)))
src_video = np.array(resized_frames)


N = min(nf1, nf2)

ratio = opts.ratio
sigma = opts.sigma

realTime_start = time.time()

for ind in range(N):
    # compute the homography between cv_cover the frame in dst video
    print(f"ind: {ind}")
    dst_frame = dst_video[ind]
    src_frame = src_video[ind]

    dst_gray = cv2.cvtColor(dst_frame, cv2.COLOR_BGR2GRAY)
    ref_gray = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create()

    kpts1, descs1 = orb.detectAndCompute(dst_gray,None)
    kpts2, descs2 = orb.detectAndCompute(ref_gray,None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    dmatches = bf.match(descs1, descs2)

    x1 = []
    x2 = []
    for m in dmatches:
        x1.append(kpts1[m.queryIdx].pt)
        x2.append(kpts2[m.trainIdx].pt)
    
    x1 = np.float32(x1)
    x2 = np.float32(x2)

    H_opencv, mask = cv2.findHomography(x2, x1, cv2.RANSAC, 2)
    comp_img = compositeH(H_opencv, src_frame, dst_frame)

    cv2.imshow('Composite Image', comp_img)
    key = cv2.waitKey(1)


realTime_end = time.time()
print(f"fps achieved: {N/(realTime_end-realTime_start)}")
























# for ind in range(N):
#     # compute the homography between cv_cover the frame in dst video
#     dst_frame = dst_video[ind]
#     src_frame = src_video[ind]

#     dst_gray = cv2.cvtColor(dst_frame, cv2.COLOR_BGR2GRAY)
#     src_gray = cv2.cvtColor(src_frame, cv2.COLOR_BGR2GRAY)

#     fast = cv2.FastFeatureDetector_create()
#     brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
    
#     key1 = fast.detect(dst_gray, None)
#     key2 = fast.detect(src_gray, None)

#     locs1, desc1 = brief.compute(dst_gray, key1)
#     locs2, desc2 = brief.compute(src_gray, key2)

#     matches = skimage.feature.match_descriptors(desc1,desc2,'hamming',cross_check=True,max_ratio=ratio)

#     print(locs1)
#     locs1[:, [1, 0]] = locs1[:, [0, 1]]
#     locs2[:, [1, 0]] = locs2[:, [0, 1]]
    
#     x1 = np.zeros(shape=(matches.shape[0],2))
#     x2 = np.zeros(shape=(matches.shape[0],2))

#     for i in range(matches.shape[0]):
#         ind1, ind2 = matches[i][0], matches[i][1]
#         x1[i][0], x1[i][1] = locs1[ind1][0], locs1[ind1][1]
#         x2[i][0], x2[i][1] = locs2[ind2][0], locs2[ind2][1]
    
#     H_opencv, mask =cv2.findHomography(x1, x2, cv2.RANSAC,2.0)
#     comp_img = compositeH(H_opencv, src_frame, dst_frame)


# realTime_end = time.time()
# print(f"fps achieved: {N/(realTime_end-realTime_start)}")