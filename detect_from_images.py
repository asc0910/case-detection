import cv2
import os
import numpy as np
import time

def my_resize_image(image, scale):
    width = int(image.shape[1] * scale)
    height = int(image.shape[0] * scale)
    return cv2.resize(image, (width, height), interpolation = cv2.INTER_AREA)

def get_color_squares(edges, scale) :
    kernel = np.ones((3,3),np.uint8)
    edges = cv2.dilate(edges, kernel, iterations = 1)

    xn, xm, yn, ym, delta = 103, 294, 429, 549, 28
    xn, xm, yn, ym, delta = xn*7//scale, xm*7//scale, yn*7//scale, ym*7//scale, delta*7//scale 
    area_s = 330 * 7 * 7 / scale / scale 

    crop_img = edges[yn-delta:ym+delta, xn-delta:xm+delta]
    # Find connected components (blobs) in the image
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cv2.bitwise_not(crop_img), connectivity=4)
    
    # Iterate over all blobs
    cenx, ceny, cens = np.zeros(num_labels, dtype=int), np.zeros(num_labels, dtype=int), 0
    for i in range(1, num_labels):  # Start from 1 to ignore the background
        # Get the area of the blob
        area = stats[i, cv2.CC_STAT_AREA]
        if area_s/10 <= area <= area_s*10:
            # print('--', area)
            cenx[cens] = area
            cens = cens+1
    area_s = np.median(cenx[0:cens])
    # print('area_s', area_s)
    cens = 0
    for i in range(1, num_labels):  # Start from 1 to ignore the background
        # Get the area of the blob
        area = stats[i, cv2.CC_STAT_AREA]
        if area_s/1.2 <= area <= area_s*1.2:# and xn-delta <= sLeft and sRight < xm+delta and yn-delta <= sTop and sBottom < ym+delta:
            # filtered[labels == i] = 255
            # print(area)
            cenx[cens], ceny[cens] = centroids[i]
            cenx[cens] = cenx[cens] + xn - delta
            ceny[cens] = ceny[cens] + yn - delta
            cens = cens+1
    best_answer = 1e8
    hx, hdx = 807, 247
    ix, idx = 807, 247
    ixs, idxs = 100, 25
    hx, hdx, ix, idx, ixs, idxs = hx//scale, hdx//scale, ix//scale, idx//scale, ixs//scale, idxs//scale
    for x in range(ix-ixs, ix+ixs):
        for dx in range(idx-idxs, idx+idxs):
            ans = 0
            for k in range(cens):
                cx = cenx[k]
                i = (cx - x + dx/2) // dx
                ans = ans + abs(cx - x-dx*i)
            if ans < best_answer:
                best_answer = ans
                hx = x
                hdx = dx
    best_answer = 1e8
    hy, hdy = 3061, 247
    iy, idy = 3061, 247
    iys, idys = 100, 25
    hy, hdy, iy, idy, iys, idys = hy//scale, hdy//scale, iy//scale, idy//scale, iys//scale, idys//scale
    for y in range(iy-iys, iy+iys):
        for dy in range(idy-idys, idy+idys):
            ans = 0
            for k in range(cens):
                cy = ceny[k]
                i = (cy - y + dy/2) // dy
                ans = ans + abs(cy - y - dy*i)
            if ans < best_answer:
                best_answer = ans
                hy = y
                hdy = dy
    return hx, hdx, hy, hdy

def get_case_bottom_line(hx, hy, hdx, hdy, scale) :
    HDY, UP = 33*7//scale, 23*7//scale
    by = hy - (UP * hdy) // HDY
    HDX, LEFT = 33*7//scale, 36*7//scale
    bx = hx - (LEFT * hdx) // HDX
    return bx, by

# Using cv2.Sobel()
def sobel_gradient(img):
    gradient_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    return gradient_magnitude

# Using custom gradient calculation with NumPy
def custom_gradient(img):
    gradient_x = np.diff(img.astype(np.float32), axis=1)
    gradient_y = np.diff(img.astype(np.float32), axis=0)
    gradient_x = np.pad(gradient_x, ((0, 0), (1, 0)), mode='constant', constant_values=0)
    gradient_y = np.pad(gradient_y, ((1, 0), (0, 0)), mode='constant', constant_values=0)
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    return gradient_magnitude

def get_gradient(img) :
    img_gray = np.max(img, axis=2)
    gradient = custom_gradient(img_gray)
    ret, thresh = cv2.threshold(gradient, 5, 255, cv2.THRESH_BINARY)
    return img_gray, thresh.astype(np.uint8)

def get_case_upper_line(gradient_img, bx, by, scale, hdx, hdxsize, case_height, case_width) :
    xn, xm, yn, ym = bx + 50*7//scale, 356*7//scale, 20*7//scale, by - 50*7//scale

    crop_img = gradient_img[yn:ym, xn:xm]

    # cv2.imshow('detection Image', crop_img)
    # cv2.waitKey(0)
    # Find connected components (blobs) in the image
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(crop_img, connectivity=8)

    # # Iterate over all blobs
    case_height = case_height*hdx/hdxsize*0.4
    case_width = case_width*hdx/hdxsize*0.4
    ux, uy = 0, ym-yn
    for i in range(1, num_labels):  # Start from 1 to ignore the background
        # Get the area of the blob
        area = stats[i, cv2.CC_STAT_AREA]
        sWidth = stats[i, cv2.CC_STAT_WIDTH]
        sHeight = stats[i, cv2.CC_STAT_HEIGHT]
        if (case_width < sWidth or case_height < sHeight) :
            sRight = stats[i, cv2.CC_STAT_LEFT] + sWidth
            if ux < sRight:
                ux = sRight
            sTop = stats[i, cv2.CC_STAT_TOP]
            if uy > sTop:
                uy = sTop  
    return ux+xn, uy+yn

    # # Create an empty image to draw the filtered blobs
    filtered = np.zeros_like(crop_img)
    
    # # Iterate over all blobs
    for i in range(1, num_labels):  # Start from 1 to ignore the background
        # Get the area of the blob
        sWidth = stats[i, cv2.CC_STAT_WIDTH]
        sHeight = stats[i, cv2.CC_STAT_HEIGHT]
        if (case_width < sWidth or case_height < sHeight) :
            filtered[labels == i] = min(255, sWidth)

    case_height = case_height*hdx/hdxsize*0.5
    case_width = case_width*hdx/hdxsize*0.5
    ux, uy = 0, ym-yn   
    
    row_sums = np.sum(filtered, axis=1)
    for y in range(ym-yn):
        if (row_sums[y] > case_width):
            uy = y
            break
    column_sums = np.sum(filtered, axis=0)
    for x in range(xm-xn):
        if (column_sums[x] > case_height):
            ux = x
    return ux+xn, uy+yn
    aaaaaaaaaaaaaaaaaaaaaaa
    filtered = np.zeros_like(crop_img)
    
    # # Iterate over all blobs
    case_height = case_height*hdx/hdxsize*0.4
    case_width = case_width*hdx/hdxsize*0.4
    ux, uy = 0, ym-yn
    for i in range(1, num_labels):  # Start from 1 to ignore the background
        # Get the area of the blob
        area = stats[i, cv2.CC_STAT_AREA]
        sWidth = stats[i, cv2.CC_STAT_WIDTH]
        sHeight = stats[i, cv2.CC_STAT_HEIGHT]
        if (case_width < sWidth or case_height < sHeight) :
            filtered[labels == i] = 255
            sRight = stats[i, cv2.CC_STAT_LEFT] + sWidth
            if ux < sRight:
                ux = sRight
            sTop = stats[i, cv2.CC_STAT_TOP]
            if uy > sTop:
                uy = sTop  
    # gradient_img = np.zeros_like(gradient_img)
    # gradient_img[yn:ym, xn:xm] = 255*np.ones_like(filtered)
    # cv2.imshow('filtered Image', filtered)
    # cv2.waitKey(0)
    return ux+xn, uy+yn

def normalize_image(image):
    # Convert image data to float32 (important for division)
    image = image.astype(np.float32)

    # Calculate the minimum and maximum pixel values in the image
    min_value = np.min(image)
    max_value = np.max(image)

    # Normalize the image to the range [0, 1]
    normalized_image = (image - min_value) / (max_value - min_value)

    return normalized_image*255

def z_score_normalize_image(image):
    # Convert image data to float32 (important for division)
    image = image.astype(np.float32)

    # Calculate the mean and standard deviation of the image
    mean_value = np.mean(image)
    std_value = np.std(image)

    # Z-score normalize the image
    normalized_image = (image - mean_value) / std_value

    return normalized_image*255

def finetune(bx, by, ux, uy, case_height, case_width):
    w,h = ux-bx, by-uy
    wh = w*h
    w = int(np.sqrt(wh*case_width/case_height))
    h = int(np.sqrt(wh*case_height/case_width))
    ux,uy = bx+w,by-h
    return bx,by,ux,uy

def process(origin_img, case_height, case_width, delta, scale):
    case_height, case_width = case_height+3, case_width+4
    img = my_resize_image(origin_img, 1./scale)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, gradient = get_gradient(img)
    # cv2.imshow('gradient Image', gradient), cv2.waitKey(0)
    # cv2.imwrite("./gray_images/gray_images_hoho.jpg", gradient)

    # Apply Canny Edge Detection
    edges = cv2.Canny(img_gray, 100, 200)
    # cv2.imshow('edges Image', edges), cv2.waitKey(0)

    # start_time = time.perf_counter()
    hx, hdx, hy, hdy = get_color_squares(edges, scale)
    print("get_color_square", hx, hdx, hy, hdy)
    # end_time = time.perf_counter()
    # elapsed_time = end_time - start_time
    # print(f'get color squares-The function took {elapsed_time} seconds to complete.')

    bx, by = get_case_bottom_line(hx, hy, hdx, hdy, scale) 

    ux, uy = get_case_upper_line(gradient, bx, by, scale, hdx, 15, case_height, case_width) #15mm=hdx


    bx, by, ux, uy = bx*scale, by*scale, ux*scale, uy*scale
    hx, hdx, hy, hdy = hx*scale, hdx*scale, hy*scale, hdy*scale
    # bx,by,ux,uy = finetune(bx, by, ux, uy, case_height, case_width)
    wc,wl = 8,2
    printscale = scale
    cv2.line(origin_img, (bx, by), (bx, uy), (0, 255, 0), wl)
    cv2.line(origin_img, (bx, by), (ux, by), (0, 255, 0), wl)
    cv2.line(origin_img, (ux, uy), (bx, uy), (0, 255, 0), wl)
    cv2.line(origin_img, (ux, uy), (ux, by), (0, 255, 0), wl)
    offset = delta * (ux-bx) // case_width
    cx, cy = (bx+ux)//2, (by+uy)//2
    cv2.circle(origin_img, (cx, cy), 3, (0, 0, 255), wc)
    cv2.circle(origin_img, (cx-offset, cy), 3, (0, 0, 255), wc)
    cv2.circle(origin_img, (cx+offset, cy), 3, (0, 0, 255), wc)
    cv2.circle(origin_img, (cx, cy-offset), 3, (0, 0, 255), wc)
    cv2.circle(origin_img, (cx, cy+offset), 3, (0, 0, 255), wc)
    for i in range(6):
        for j in range(4) :
            cv2.circle(origin_img, (hx+i*hdx, hy+j*hdy), 5, (0, 0, 255), 4)
    
    if (printscale > 1):
        origin_img = my_resize_image(origin_img, 1./printscale)
    return origin_img
