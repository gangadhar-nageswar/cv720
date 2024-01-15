from plistlib import load
from turtle import end_fill, shape
import numpy as np
import cv2

#Import necessary functions
from loadVid import loadVid
from matchPics import matchPics
from helper import plotMatches
from opts import get_opts
from planarH import computeH_ransac, compositeH

import time

import multiprocessing
n_cpu = multiprocessing.cpu_count()
print("------ n CPU: ", n_cpu)




# define a function which returns the homographised image
def apply_homography(frame_ind, dst_frame, ref_frame, src_frame, opts):

    print(f"Starting frame ind: {frame_ind}")
    matches, locs1, locs2 = matchPics(dst_frame, ref_frame, opts)
    locs1[:, [1, 0]] = locs1[:, [0, 1]]
    locs2[:, [1, 0]] = locs2[:, [0, 1]]

    print(f"num of locs1: {locs1.shape} || locs2: {locs2.shape} || matches: {matches.shape}")
    # plotMatches(dst_frame, ref_frame, matches, locs1, locs2)
    
    x1 = np.zeros(shape=(matches.shape[0],2))
    x2 = np.zeros(shape=(matches.shape[0],2))

    for i in range(matches.shape[0]):
        ind1, ind2 = matches[i][0], matches[i][1]
        x1[i][0], x1[i][1] = locs1[ind1][0], locs1[ind1][1]
        x2[i][0], x2[i][1] = locs2[ind2][0], locs2[ind2][1]
    
    H_computed, inliers = computeH_ransac(x1, x2, opts)

    # srcHeight, srcWidth, _ = ref_frame.shape
    # src_frame = cv2.resize(src_frame, (srcWidth, srcHeight))

    comp_img = compositeH(H_computed, src_frame, dst_frame)
    # np.save(f"../result_frames/comp_{frame_ind}.npy", comp_img)
    # cv2.imwrite(f"../result_frames/comp_{frame_ind}.png", comp_img)

    print(f"Finished frame ind: {frame_ind}")

    return comp_img



opts = get_opts()

#Write script for Q3.1
src_path = "../data/ar_source.mov"
dst_path = "../data/book.mov"

start_time = time.time()
#Load the videos
src_video = loadVid(src_path)
dst_video = loadVid(dst_path)
end_time = time.time()
print(f"time to load images: {end_time - start_time}")

# # save the video as numpy arrays for future purposes
# np.save("../data/src_video.npy", src_video)
# np.save("../data/dst_video.npy", dst_video)

# load the video numpy images
# start_time = time.time()
# src_video = np.load("../data/src_video.npy")
# dst_video = np.load("../data/dst_video.npy")
# end_time = time.time()
# print(f"time to video numpy arrays: {end_time - start_time}")

print(f"source video shape: {src_video.shape}")
print(f"dst video shape: {dst_video.shape}")

nf1, nf2 = src_video.shape[0], dst_video.shape[0]
N = min(nf1, nf2)

#Load images requried for computing homography
cv_cover = cv2.imread('../data/cv_cover.jpg')

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


# pool_args = []
# for i in range(N):
#     pool_args.append((i, dst_video[i], cv_cover, src_video[i], opts))

# print("starting processing data")

# pool = multiprocessing.Pool(n_cpu)
# features = pool.starmap(apply_homography, pool_args)

# compositeFrames = np.array(features)
# compositeVideo = compositeFrames.reshape((N,dstHeight,dstWidth,3))
# np.save("../resulting_frames.npy", compositeVideo)


for ind in range(10):
    # compute the homography between cv_cover the frame in dst video
    dst_frame = dst_video[ind]
    src_frame = src_video[ind]

    comp_frame = apply_homography(ind, dst_frame, cv_cover, src_frame, opts)