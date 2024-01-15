import numpy as np
import cv2
from matchPics import matchPics
from opts import get_opts
import scipy
import matplotlib.pyplot as plt
from helper import plotMatches



opts = get_opts()
#Q2.1.6
#Read the image and convert to grayscale, if necessary
cv_cover = cv2.imread('../data/cv_cover.jpg')
# cv_cover = cv2.cvtColor(cv_cover, cv2.COLOR_BGR2GRAY)

hist = []

for i in range(37):
	#Rotate Image
	rt_cover = scipy.ndimage.rotate(cv_cover, 10*(i), reshape=False)

	#Compute features, descriptors and Match features
	matches, locs1, locs2 = matchPics(cv_cover, rt_cover, opts)

	# if i in [0, 1, 9, 18]:
	# 	plotMatches(cv_cover, rt_cover, matches, locs1, locs2)

	#Update histogram
	hist.append(matches.shape[0])


xpoints = 10*np.arange(37)

fig = plt.figure(figsize = (10, 5))

plt.bar(xpoints, hist, width = 5)
plt.xlabel("Angle Rotated (degrees)")
plt.ylabel("Number of Matches")
plt.title("Histogram for Matches")
# Show plot
plt.show()

fig.savefig('../rot_hist.png')


