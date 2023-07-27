import cv2
import os
import numpy as np
import time

#define AST
    
#define AEN

def my_resize_image(image, scale):
    width = int(image.shape[1] * scale)
    height = int(image.shape[0] * scale)
    return cv2.resize(image, (width, height), interpolation = cv2.INTER_AREA)

def get_color_squares(edges, scale, bx1, by1) :
    kernel = np.ones((3,3),np.uint8)
    edges = cv2.dilate(edges, kernel, iterations = 1)
    width, height = edges.shape[1], edges.shape[0]
    crop = edges[by1:height, bx1:width]
    # cv2.imshow('crop Image', crop), cv2.waitKey(0)
    # Find connected components (blobs) in the image
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cv2.bitwise_not(crop), connectivity=4)
    
    # Iterate over all blobs
    cenx, ceny, cens = np.zeros(num_labels, dtype=int), np.zeros(num_labels, dtype=int), 0
    for i in range(1, num_labels):  # Start from 1 to ignore the background
        # Get the area of the blob
        area = stats[i, cv2.CC_STAT_AREA]
        sWidth = stats[i, cv2.CC_STAT_WIDTH]
        sHeight = stats[i, cv2.CC_STAT_HEIGHT]
        if sWidth<=1.2*sHeight and sHeight<=1.2*sWidth and sWidth*sHeight <= area*1.2 and 20 <= area:
            # print('--', area)
            cenx[cens] = area
            cens = cens+1
    area_s = np.median(cenx[0:cens])
    # print('area_s', area_s)
    cens = 0
    for i in range(1, num_labels):  # Start from 1 to ignore the background
        # Get the area of the blob
        area = stats[i, cv2.CC_STAT_AREA]
        sWidth = stats[i, cv2.CC_STAT_WIDTH]
        sHeight = stats[i, cv2.CC_STAT_HEIGHT]
        if sWidth<=1.2*sHeight and sHeight<=1.2*sWidth and sWidth*sHeight <= area*1.2 and 20 <= area and area_s/1.2 <= area <= area_s*1.2:
            cenx[cens], ceny[cens] = centroids[i]
            cens = cens+1
    best_answer = 1e8
    hx, hdx = 0, 0
    # print(cens)
    mnx,mxx = np.min(cenx[0:cens]), np.max(cenx[0:cens])
    for r in range(1,6):
        dx = int((mxx-mnx+r/2) // r)
        if (dx == 0):
            continue
        for j in range(6-r):
            x = mnx - j * dx
            if (x < 0):
                continue
            ans = 0
            for k in range(cens):
                cx = cenx[k]
                i = int((cx - x + dx/2) // dx)
                ans = ans + abs(cx - x-dx*i)
            if ans < best_answer:
                best_answer = ans
                hx = x
                hdx = dx
    ix, idx = max(6,hx), max(4,hdx)
    ixs, idxs = 5, 3
    for x in range(ix-ixs, ix+ixs):
        dx = idx-idxs
        while dx < idx + idxs:
            ans = 0
            for k in range(cens):
                cx = cenx[k]
                i = int((cx - x + dx/2) // dx)
                i = np.maximum(0,i)
                i = np.minimum(5, i)
                ans = ans + abs(cx - x-dx*i)
            if ans < best_answer:
                best_answer = ans
                hx = x
                hdx = dx
            dx += 0.5
    best_answer = 1e8
    hy, hdy = 0, 0
    mny,mxy = np.min(ceny[0:cens]), np.max(ceny[0:cens])
    for r in range(1,4):
        dy = int((mxy-mny+r/2) // r)
        if (dy == 0):
            continue
        for j in range(4-r):
            y = mny - j * dy
            if (y < 0):
                continue
            ans = 0
            for k in range(cens):
                cy = ceny[k]
                i = int((cy - y + dy/2) // dy)
                ans = ans + abs(cy - y - dy*i)
            if ans < best_answer:
                best_answer = ans
                hy = y
                hdy = dy
    iy, idy = max(6,hy), max(4,hdy)
    iys, idys = 5, 3
    for y in range(iy-iys, iy+iys):
        dy = idy-idys
        while dy < idy+idys:
            ans = 0
            for k in range(cens):
                cy = ceny[k]
                i = int((cy - y + dy/2) // dy)
                i = np.maximum(0,i)
                i = np.minimum(3, i)
                ans = ans + abs(cy - y - dy*i)
            if ans < best_answer:
                best_answer = ans
                hy = y
                hdy = dy
            dy += 0.5
    return hx+bx1, hdx, hy+by1, hdy

def get_case_bottom_line(hx, hy, hdx, hdy, scale) :
    bx = int(hx - hdx * 240/232)
    by = int(hy - hdy * 156/232)
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
    return thresh.astype(np.uint8)

def get_case_upper_line(gradient_img, bx, by, scale, hdx, hdxsize, case_height, case_width) :
    width = int(gradient_img.shape[1])
    height = int(gradient_img.shape[0])
    xn, xm, yn, ym = int(bx+width/6), int(width*3/4), int(height/15), int(by-height/15)
    

    crop_img = gradient_img[yn:ym, xn:xm]

    # cv2.imshow('detection Image', crop_img), cv2.waitKey(0)
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

def get_case_bottom_line1(img):
    width, height = img.shape[1], img.shape[0]
    crop = img[0:height//10, 0:width//2]
    # print(width, height)
    # cv2.imshow('crop Image', crop), cv2.waitKey(0)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(crop, connectivity=4)
    bx=0
    for i in range(1, num_labels):
        sHeight = stats[i, cv2.CC_STAT_HEIGHT]
        sLeft = stats[i, cv2.CC_STAT_LEFT]
        if height <= sHeight*11:
            if bx < sLeft:
                bx = sLeft
    crop = img[height//10:height, width*2//3:width*3//4]
    # cv2.imshow('crop Image', crop), cv2.waitKey(0)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(crop, connectivity=4)
    by=height
    for i in range(1, num_labels):
        sWidth = stats[i, cv2.CC_STAT_WIDTH]
        sTop = stats[i, cv2.CC_STAT_TOP]
        if width <= sWidth*13:
            if by > sTop:
                by = sTop
    return bx, by+height//10

def process(origin_img, case_height, case_width, delta, scale):
    case_height, case_width = case_height+3, case_width+4
    width, height = origin_img.shape[1], origin_img.shape[0]
    img = my_resize_image(origin_img, 1./scale)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # start_time = time.perf_counter()
    gradient = get_gradient(img)
    # end_time = time.perf_counter()
    # elapsed_time = end_time - start_time
    # print(f'get color squares-The function took {elapsed_time} seconds to complete.')
    # cv2.imshow('gradient bImage', gradient), cv2.waitKey(0)
    bx1, by1 = get_case_bottom_line1(gradient)

    # Apply Canny Edge Detection
    edges = cv2.Canny(img_gray, 100, 200)
    # cv2.imshow('edges Image', edges), cv2.waitKey(0)
    
    hx, hdx, hy, hdy = get_color_squares(edges, scale, bx1, by1)
    # cv2.line(img, (hx, hy), (hx, 0), (0, 255, 0), 1)
    # cv2.line(img, (hx, hy), (img.shape[1], hy), (0, 255, 0), 1)
    # for i in range(6):
    #     for j in range(4):
    #         cv2.circle(img, (hx+i*hdx, hy+j*hdy), 1, (0, 0, 255), 1)
            
    bx, by = get_case_bottom_line(hx, hy, hdx, hdy, scale) 

    ux, uy = get_case_upper_line(gradient, bx, by, scale, hdx, 15, case_height, case_width) #15mm=hdx


    printscale = 1
    bx, by, ux, uy = bx*scale//printscale, by*scale//printscale, ux*scale//printscale, uy*scale//printscale
    hx, hdx, hy, hdy = hx*scale//printscale, hdx*scale//printscale, hy*scale//printscale, hdy*scale//printscale
    # bx,by,ux,uy = finetune(bx, by, ux, uy, case_height, case_width)
    if (printscale > 1):
        origin_img = my_resize_image(origin_img, 1./printscale)
    print_height = int(origin_img.shape[0]/printscale)
    wc = max(2, print_height//336)
    wl = max(1, print_height//336//4)
    ra = max(1, print_height//336//3)
    # print(wc, wl, ra)
    
    cv2.line(origin_img, (bx, by), (bx, uy), (0, 255, 0), wl)
    cv2.line(origin_img, (bx, by), (ux, by), (0, 255, 0), wl)
    cv2.line(origin_img, (ux, uy), (bx, uy), (0, 255, 0), wl)
    cv2.line(origin_img, (ux, uy), (ux, by), (0, 255, 0), wl)
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

    for i in range(6):
        for j in range(4):
            cv2.circle(origin_img, (int(hx+i*hdx), int(hy+j*hdy)), ra, (0, 0, 255), wc)
    
    return origin_img
