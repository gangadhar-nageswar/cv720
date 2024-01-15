import argparse
from pipes import Template
from tokenize import PlainToken
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from LucasKanade import LucasKanade

# write your script here, we recommend the above libraries for making your animation

parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e4, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-4, help='dp threshold of Lucas-Kanade for terminating optimization')
args = parser.parse_args()
num_iters = int(args.num_iters)
threshold = args.threshold

seq = np.load("../data/carseq.npy")
rect = [59, 116, 145, 151]

num_frames = seq.shape[2]
temp_img = seq[:,:,0]

ids = [1, 100, 200, 300, 400]

p0 = np.ones(2)
p = p0
print(f"initial p: {p} || {p.shape}")


fig, axes = plt.subplots(1, len(ids), figsize=(10, 2))

rects_hist = []

for frame_id in range(num_frames):
    print(f"----- frame_id: {frame_id} -----")
    img = seq[:,:,frame_id]
    p = LucasKanade(temp_img, img, rect, threshold, num_iters, p)
    
    temp_img = np.copy(img)
    rect_plot = [rect[0]+p[0], rect[1]+p[1], rect[2]+p[0], rect[3]+p[1]]
    rect = [rect[0]+p[0], rect[1]+p[1], rect[2]+p[0], rect[3]+p[1]]

    rects_hist.append(rect_plot)
    
    if frame_id in ids:

        ind = ids.index(frame_id)

        # rect_new = [rect[0]+p[0], rect[1]+p[1], rect[2]+p[0], rect[3]+p[1]]

        c1,r1 = rect_plot[0], rect_plot[1]
        c2,r2 = rect_plot[2], rect_plot[3]

        # Create a Rectangle object to represent the bounding box
        bounding_box = patches.Rectangle(xy=(c1,r1), width=c2-c1, height=r2-r1, edgecolor='red', facecolor="none")

        # Create a figure and add the image and bounding box
        axes[ind].imshow(img, cmap='gray')
        axes[ind].add_patch(bounding_box)
        axes[ind].set_title(f"Frame: {frame_id}")



rects_hist = np.array(rects_hist)
print(rects_hist)
np.save("../sub_files/carseqrects.npy", rects_hist)

plt.tight_layout()

# Show the plot
plt.show()
plt.clf()