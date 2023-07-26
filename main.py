import detect_from_images
import os
import time
import cv2

dir_path = './images'

# Loop through all files in the directory
for filename in os.listdir(dir_path):
    if filename.endswith(".jpg") or filename.endswith(".png"): # Add or modify the file extensions you need
        print(os.path.join(dir_path, filename))
        img_path = os.path.join(dir_path, filename)

        # Read the image
        img = cv2.imread(img_path)    

        # Construct the output file path
        base_filename, file_extension = os.path.splitext(filename)
        out_filename = base_filename + file_extension

        start_time = time.perf_counter()
        new_img = detect_from_images.process(img, 147, 72, 3, 4)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f'The function took {elapsed_time} seconds to complete.')

        # cv2.imshow('detection Image', new_img)
        # cv2.waitKey(0)
        cv2.imwrite(os.path.join(dir_path.replace("images", "gray_images"), filename), new_img)
        # break
    else:
        continue

# Close all OpenCV windows
cv2.destroyAllWindows()