import detect_from_images
import os
import time
import cv2
import numpy as np

from colour import read_image
# from colour_checker_detection import colour_checkers_coordinates_segmentation

dir_path = './images'
sstart_time = time.perf_counter()
# Loop through all files in the directory

for rrr in range(1):
    for filename in os.listdir(dir_path):
        if filename.endswith(".jpg") or filename.endswith(".png"): # Add or modify the file extensions you need
            print(os.path.join(dir_path, filename))
            img_path = os.path.join(dir_path, filename)

            # Read the image
            img = cv2.imread(img_path)

            # Construct the output file path
            base_filename, file_extension = os.path.splitext(filename)
            out_filename = base_filename + file_extension


            height,width = img.shape[0], img.shape[1]
            if (height>3000) :
                x = int(250 + 150*np.random.rand())
                y = int(150 + 100*np.random.rand())
                print(x,y,rrr)
                # img = img[y:y+1920*2, x:x+1080*2]
                # img = cv2.resize(img, (1080, 1920), interpolation = cv2.INTER_AREA)

            start_time = time.perf_counter()
            scale = int(max(img.shape[0], img.shape[1]) / 500)
            # img = detect_from_images.my_resize_image(img, 1./scale)
            # aaa = colour_checkers_coordinates_segmentation(img),print(aaa)
            # cv2.imshow("img", img), cv2.waitKey(0)
            # print(scale)
            
            # 1 means lying down
            # 2 means standing up
            # 3 means don't know
            print(scale)
            img = detect_from_images.process(img, 147, 72, 3, scale, 1)
            
            end_time = time.perf_counter()
            elapsed_time = end_time - start_time
            print(f'The function took {elapsed_time} seconds to complete.')

            # cv2.imshow('detection Image', img)
            # cv2.waitKey(0)
            cv2.imwrite(os.path.join(dir_path.replace("images", "gray_images"), filename.replace(".", "-"+str(rrr)+".")), img)
            # break
        else:
            continue
send_time = time.perf_counter()
selapsed_time = send_time - sstart_time
print(f'The function took {selapsed_time} seconds to complete.')
# Close all OpenCV windows
cv2.destroyAllWindows()