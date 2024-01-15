import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from SubtractDominantMotion import SubtractDominantMotion
from LucasKanadeAffine import LucasKanadeAffine

# write your script here, we recommend the above libraries for making your animation

parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e3, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-1, help='dp threshold of Lucas-Kanade for terminating optimization')
parser.add_argument('--tolerance', type=float, default=0.2, help='binary threshold of intensity difference when computing the mask')
args = parser.parse_args()
num_iters = int(args.num_iters)
threshold = args.threshold
tolerance = args.tolerance

seq = np.load('../data/aerialseq.npy')

num_frames = seq.shape[2]

for frame_id in range(1,num_frames):
    print(f"----- frame_id: {frame_id} -----")

    image1 = seq[:,:,frame_id-1]
    image2 = seq[:,:,frame_id]

    img_mask = SubtractDominantMotion(image1, image2, threshold, num_iters, tolerance)

    track_img = np.copy(image2)
    # track_img[img_mask == 255] = 255

    img_plot = plt.subplot()

    img_plot.imshow(track_img)
    # to convert the numpy array to cv compatible mat
    # track_img = np.ascontiguousarray(track_img)

    for i in range(img_mask.shape[1]):
        for j in range(img_mask.shape[0]):
            if img_mask[j][i] == 255:
                patch = patches.Circle((i,j), radius=1, facecolor='b')
                img_plot.add_patch(patch)
    
    plt.show()
    # plt.savefig(f"../aerial_images/frame{frame_id}.jpg")
    # plt.clf()