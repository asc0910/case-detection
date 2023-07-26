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

    width = int(edges.shape[1] * scale)
    height = int(edges.shape[0] * scale)

    xn, xm, yn, ym, delta = 103*7, 294*7, 429*7, 549*7, 28*7
    area_s = 330 * 7 * 7 / scale / scale 
    if (-100<(width*16-height*9)<100) :
        xn, xm, yn, ym, delta = 150, 1409, 1400, 1800, 28*7
        area_s = 150 * 7 * 7 / scale / scale 
    xn, xm, yn, ym, delta = xn//scale, xm//scale, yn//scale, ym//scale, delta//scale

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
    if (-100<(width*16-height*9)<100) :
        hx, hdx = 197, 120
        ix, idx = 197, 120
        ixs, idxs = 50, 12
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
    if (-100<(width*16-height*9)<100) :
        hy, hdy = 1450, 120
        iy, idy = 1450, 120
        iys, idys = 50, 12
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

    # direction = np.arctan2(gradient_y, gradient_x) * 128 / np.pi
    # direction[direction > 118] = 255
    # direction[direction < -118] = 255
    # direction[(direction>0) & (direction<10)] = 255
    # direction[(direction>-10) & (direction<0)] = 255
    # direction[(direction>54) & (direction<74)] = 255
    # direction[(direction>182) & (direction<202)] = 255
    # direction[direction < 255] = 0
    # # gradient_magnitude[direction != 255] = 0
    # # Convert direction to 8-bit image for visualization (optional)
    # direction_image = np.uint8(direction)

    return gradient_magnitude

def get_gradient(img) :
    img_gray = np.max(img, axis=2)
    gradient = custom_gradient(img_gray)
    ret, thresh = cv2.threshold(gradient, 5, 255, cv2.THRESH_BINARY)
    # cv2.imshow('direction_image bImage', thresh), cv2.waitKey(0)
    return thresh.astype(np.uint8)

def get_case_upper_line(gradient_img, bx, by, scale, hdx, hdxsize, case_height, case_width) :
    xn, xm, yn, ym = bx + 50*7//scale, 356*7//scale, 20*7//scale, by - 50*7//scale
    width = int(gradient_img.shape[1] * scale)
    height = int(gradient_img.shape[0] * scale)
    if (-100<(width*16-height*9)<100) :
        xn, xm, yn, ym = bx + 25*7//scale, 900//scale, 100//scale, by - 25*7//scale

    crop_img = gradient_img[yn:ym, xn:xm]

    # cv2.imshow('detection Image', crop_img)
    # cv2.waitKey(0)
    # Find connected components (blobs) in the image
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(crop_img, connectivity=8)

    # # Create an empty image to draw the filtered blobs
    filtered = np.zeros_like(crop_img)
    case_height = case_height*hdx/hdxsize
    case_width = case_width*hdx/hdxsize
    lambda1 = 0.05
    # # Iterate over all blobs
    for i in range(1, num_labels):  # Start from 1 to ignore the background
        # Get the area of the blob
        sWidth = stats[i, cv2.CC_STAT_WIDTH]
        sHeight = stats[i, cv2.CC_STAT_HEIGHT]
        if (case_width*lambda1 < sWidth or case_height*lambda1 < sHeight) :
            filtered[labels == i] = 1

    ux, uy = 0, ym-yn
    lambda2 = 0.3
    
    row_sums = np.sum(filtered, axis=1)
    tot_sum = 0
    for y in range(ym-yn):
        tot_sum = tot_sum + row_sums[y]
        if (tot_sum > case_width*lambda2):
            uy = y
            break
    column_sums = np.sum(filtered, axis=0)
    tot_sum = 0
    for x in range(xm-xn-1, -1, -1):
        tot_sum = tot_sum + column_sums[x]
        if (tot_sum > case_height*lambda2):
            ux = x
            break
    return ux+xn, uy+yn
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
    gradient = get_gradient(img)

    # cv2.imshow('gradient bImage', gradient), cv2.waitKey(0)
    # cv2.imwrite("./gray_images/gray_images_hoho.jpg", gradient)

    # Apply Canny Edge Detection
    edges = cv2.Canny(img_gray, 100, 200)
    # cv2.imshow('edges Image', edges), cv2.waitKey(0)

    # start_time = time.perf_counter()
    hx, hdx, hy, hdy = get_color_squares(edges, scale)
    # print("get_color_square", hx, hdx, hy, hdy)
    # end_time = time.perf_counter()
    # elapsed_time = end_time - start_time
    # print(f'get color squares-The function took {elapsed_time} seconds to complete.')

    bx, by = get_case_bottom_line(hx, hy, hdx, hdy, scale) 

    ux, uy = get_case_upper_line(gradient, bx, by, scale, hdx, 15, case_height, case_width) #15mm=hdx


    printscale = 1
    bx, by, ux, uy = bx*scale//printscale, by*scale//printscale, ux*scale//printscale, uy*scale//printscale
    hx, hdx, hy, hdy = hx*scale//printscale, hdx*scale//printscale, hy*scale//printscale, hdy*scale//printscale
    # bx,by,ux,uy = finetune(bx, by, ux, uy, case_height, case_width)
    if (printscale > 1):
        origin_img = my_resize_image(origin_img, 1./printscale)
    wc,wl,ra = 12,2,3
    wc = max(2, wc//printscale)
    wl = max(1, wl//printscale)
    ra = max(1, ra//printscale)
    # cv2.line(origin_img, (bx, by), (bx, uy), (0, 255, 0), wl)
    # cv2.line(origin_img, (bx, by), (ux, by), (0, 255, 0), wl)
    # cv2.line(origin_img, (ux, uy), (bx, uy), (0, 255, 0), wl)
    # cv2.line(origin_img, (ux, uy), (ux, by), (0, 255, 0), wl)
    offset = delta * (ux-bx) // case_width
    cx, cy = (bx+ux)//2, (by+uy)//2
    
    crop = origin_img[cy-wc-offset:cy+wc+offset+1, cx-wc-offset:cx+wc+offset+1, :]
    # cv2.imshow('direction_image bImage', crop), cv2.waitKey(0)
    
    crop = np.mean(np.mean(crop, axis=0), axis=0)
    # print(crop)
    best_answer = 0
    sum = [0, 0, 0]
    for a in range(2):
        for b in range(2):
            for c in range(2):
                if (a+b+c==0 or a+b+c==3):
                    continue
                ans = np.abs(crop[0]-255*a) + np.abs(crop[1]-255*b) + np.abs(crop[2]-255*c)
                if best_answer < ans:
                    best_answer = ans
                    sum = [255*a, 255*b, 255*c]
    # print('sum = ', sum)
    sum0, sum1, sum2 = int(sum[0]), int(sum[1]), int(sum[2])

    offx, offy = [-1, 0, 1, 0, 0], [0, 0, 0, -1, 1]
    for r in range(5):
        ox = cx + offx[r] * offset
        oy = cy + offy[r] * offset        
        cv2.circle(origin_img, (ox, oy), ra, (sum0, sum1, sum2), wc)
        # cv2.circle(origin_img, (ox, oy), 3, (255, 0, 0), wc)

    # for i in range(6):
    #     for j in range(4):
    #         cv2.circle(origin_img, (hx+i*hdx, hy+j*hdy), ra, (0, 0, 255), wc)
    
    return origin_img
