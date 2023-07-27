import detect_from_images
import os
import time
import cv2

dir_path = './images'
sstart_time = time.perf_counter()
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
        scale = int(img.shape[0] / 500)
        # print(scale)
        # img = detect_from_images.process(img, 147, 72, 3, scale)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f'The function took {elapsed_time} seconds to complete.')

        # cv2.imshow('detection Image', img)
        # cv2.waitKey(0)
        cv2.imwrite(os.path.join(dir_path.replace("images", "gray_images"), filename), img)
        # break
    else:
        continue
send_time = time.perf_counter()
selapsed_time = send_time - sstart_time
print(f'The function took {selapsed_time} seconds to complete.')
# Close all OpenCV windows
cv2.destroyAllWindows()