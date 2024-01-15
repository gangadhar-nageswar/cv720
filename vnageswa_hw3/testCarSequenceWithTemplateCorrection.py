import argparse
from pipes import Template
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from LucasKanade import LucasKanade

# write your script here, we recommend the above libraries for making your animation

parser = argparse.ArgumentParser()
parser.add_argument('--num_iters', type=int, default=1e4, help='number of iterations of Lucas-Kanade')
parser.add_argument('--threshold', type=float, default=1e-2, help='dp threshold of Lucas-Kanade for terminating optimization')
parser.add_argument('--template_threshold', type=float, default=3, help='threshold for determining whether to update template')
args = parser.parse_args()
num_iters = int(args.num_iters)
threshold = args.threshold
template_threshold = args.template_threshold

seq = np.load("../data/carseq.npy")
rect = [59, 116, 145, 151]
orig_rect = rect.copy()

num_frames = seq.shape[2]

orig_temp_img = seq[:,:,0]
temp_img = np.copy(orig_temp_img)

ids = [1, 100, 200, 300, 400]

p0 = np.zeros(2)
p_prev = p0

rect_n = [rect[0]+p_prev[0], rect[1]+p_prev[1], rect[2]+p_prev[0], rect[3]+p_prev[1]]

fig, axes = plt.subplots(1, len(ids), figsize=(10, 2))

rects_hist = []

for frame_id in range(num_frames):
    print(f"\n----- frame_id: {frame_id} -----")
    
    img = seq[:,:,frame_id]

    pn = LucasKanade(temp_img, img, rect_n, threshold, num_iters, p_prev)
    pn_star = LucasKanade(orig_temp_img, img, orig_rect, threshold, num_iters, pn)

    p_prev = np.copy(pn_star)

    if LA.norm(pn_star-pn, 2) <= template_threshold:
        print("template updating...")
        temp_img = np.copy(img)
        rect_n = [rect_n[0]+pn_star[0], rect_n[1]+pn_star[1], rect_n[2]+pn_star[0], rect_n[3]+pn_star[1]]

    rect_plot = [orig_rect[0]+pn_star[0], orig_rect[1]+pn_star[1], orig_rect[2]+pn_star[0], orig_rect[3]+pn_star[1]]
    rects_hist.append(rect_plot)

    if frame_id in ids:
        ind = ids.index(frame_id)
        rect_new = [orig_rect[0]+pn_star[0], orig_rect[1]+pn_star[1], orig_rect[2]+pn_star[0], orig_rect[3]+pn_star[1]]

        c1,r1 = rect_new[0], rect_new[1]
        c2,r2 = rect_new[2], rect_new[3]

        # Create a Rectangle object to represent the bounding box
        bounding_box = patches.Rectangle(xy=(c1,r1), width=c2-c1, height=r2-r1, edgecolor='blue', facecolor="none")

        # Create a figure and add the image and bounding box
        axes[ind].imshow(img, cmap='gray')
        axes[ind].add_patch(bounding_box)
        axes[ind].set_title(f"Frame: {frame_id}")


rects_hist = np.array(rects_hist)
np.save("../sub_files/carseqrects-wcrt.npy", rects_hist)

plt.tight_layout()

# Show the plot
plt.show()
plt.clf()