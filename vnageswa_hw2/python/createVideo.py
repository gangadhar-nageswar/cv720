import cv2
import os

image_directory = '../result_frames'

# Output directory where you want to save the video
output_directory = '../'

# Output video file name
video_name = os.path.join(output_directory, 'output_video.avi')

# Frame rate (frames per second)
frame_rate = 30

# Get a list of all image files in the directory
images = [img for img in os.listdir(image_directory) if img.endswith(".png")]

# # Sort the images to ensure they are in the correct order
images.sort()

img_path0 = os.path.join(image_directory, images[0])
frame = cv2.imread(img_path0)
imgW, imgH = frame.shape[1], frame.shape[0]

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(video_name, fourcc, frame_rate, (imgW, imgH))  # Adjust resolution as needed

for i in range(510):
    print(i)
    img_path = os.path.join(image_directory, f'comp_{i}.png')
    frame = cv2.imread(img_path)
    out.write(frame)

# Release the VideoWriter
out.release()

# Optionally, display a message when the video creation is complete
print(f"Video '{video_name}' created successfully.")