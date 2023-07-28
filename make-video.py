import cv2
import numpy as np
import os
import time

# Path to the folder containing the images
image_folder = './images/'

# Get the list of image files in the folder
images = [img for img in os.listdir(image_folder) if img.endswith('.jpg')]

# Sort the images in ascending order based on their names
images.sort()

# Specify the video codec and frame rate
video_codec = cv2.VideoWriter_fourcc(*'mp4v')
fps = 10.0

# Set the output video file name
output_file = 'input.mp4'

# Get the first image to read its dimensions
first_image = cv2.imread(os.path.join(image_folder, images[0]))
height, width, _ = 1920, 1080, 3

# Create a VideoWriter object to write the video
video_writer = cv2.VideoWriter(output_file, video_codec, fps, (width, height))

# Loop through the images and add them to the video
for r in range(5):
    for image_name in images:
        for k in range(1):
            print(r, image_name, k)
            image_path = os.path.join(image_folder, image_name)
            image = cv2.imread(image_path)
            
            height,width = image.shape[0], image.shape[1]
            if (height>3000) :
                x = int(400 + 100*np.random.rand())
                y = int(200 + 100*np.random.rand())
                image = image[y:y+1920*2, x:x+1080*2]
                image = cv2.resize(image, (1080, 1920), interpolation = cv2.INTER_AREA)
            # cv2.imshow('hoho', image)
            # cv2.waitKey(0)
            # break
            video_writer.write(image)
        # break
    # break

# Release the VideoWriter and close the video file
video_writer.release()

print("Video created successfully.")
