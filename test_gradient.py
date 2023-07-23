import cv2
import numpy as np
import timeit
import detect_from_images

# Load the image in grayscale
img = cv2.imread('./images/1054323-1154591_QC.jpg', cv2.IMREAD_GRAYSCALE)
img = detect_from_images.my_resize_image(img, 1./4)
cv2.imshow('detection Image', img)
cv2.waitKey(0)

# Using cv2.Sobel()
def sobel_gradient():
    gradient_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    cv2.imshow('detection Image', gradient_magnitude/255)
    cv2.waitKey(0)
    return gradient_magnitude

# Using custom gradient calculation with NumPy
def custom_gradient():
    gradient_x = np.diff(img.astype(np.float32), axis=1)
    gradient_y = np.diff(img.astype(np.float32), axis=0)
    gradient_x = np.pad(gradient_x, ((0, 0), (1, 0)), mode='constant', constant_values=0)
    gradient_y = np.pad(gradient_y, ((1, 0), (0, 0)), mode='constant', constant_values=0)
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    cv2.imshow('detection Image', gradient_magnitude/255)
    cv2.waitKey(0)
    return gradient_magnitude

# Measure execution time for each method
sobel_time = timeit.timeit(sobel_gradient, number=1)
custom_time = timeit.timeit(custom_gradient, number=1)

print("Sobel execution time:", sobel_time)
print("Custom gradient execution time:", custom_time)
