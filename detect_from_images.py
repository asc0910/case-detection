import cv2
import os
import numpy as np
import time

def my_resize_image(image, scale):
    width = int(image.shape[1] * scale)
    height = int(image.shape[0] * scale)
    dim = (width, height)
    return cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

def get_color_squares(edges, scale) :
    kernel = np.ones((3,3),np.uint8)
    edges = cv2.dilate(edges, kernel, iterations = 1)

    xn, xm, yn, ym, delta = 103, 294, 429, 549, 28
    xn, xm, yn, ym, delta = xn*7//scale, xm*7//scale, yn*7//scale, ym*7//scale, delta*7//scale 
    area_s = 600 * 7 * 7 / scale / scale 

    crop_img = edges[yn-delta:ym+delta, xn-delta:xm+delta]
    # Find connected components (blobs) in the image
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cv2.bitwise_not(crop_img), connectivity=4)

    # # Create an empty image to draw the filtered blobs
    # filtered = np.zeros_like(edges)
    # print('num_labels = ', num_labels)
    
    # Iterate over all blobs
    cenx, ceny, cens = np.zeros(num_labels, dtype=int), np.zeros(num_labels, dtype=int), 0
    for i in range(1, num_labels):  # Start from 1 to ignore the background
        # Get the area of the blob
        area = stats[i, cv2.CC_STAT_AREA]
        # sLeft = stats[i, cv2.CC_STAT_LEFT]
        # sTop = stats[i, cv2.CC_STAT_TOP]
        # sRight = sLeft + stats[i, cv2.CC_STAT_WIDTH]
        # sBottom = sTop + stats[i, cv2.CC_STAT_HEIGHT]
        # If the area is between 100 and 200, draw the blob on the filtered image
        if area_s/1.5 <= area <= area_s*1.5:# and xn-delta <= sLeft and sRight < xm+delta and yn-delta <= sTop and sBottom < ym+delta:
            # filtered[labels == i] = 255
            cenx[cens], ceny[cens] = centroids[i]
            cenx[cens] = cenx[cens] + xn - delta
            ceny[cens] = ceny[cens] + yn - delta
            cens = cens+1
    # dx, dy = [-1, 1, 0, 0], [0, 0, -1, 1]
    # flag = np.zeros((4032//7, 3024//7), dtype=np.uint8) 
    # qx, qy = np.zeros(432*576, dtype = int), np.zeros(432*576, dtype = int)
    # cenx, ceny, cens = np.zeros(100, dtype=int), np.zeros(100, dtype=int), 0
    # tot = 0
    # for x in range(xn, xm) :
    #     for y in range(yn, ym) :
    #         if (flag[y][x] != 0 or edges[y][x] != 0) :
    #             continue
    #         st = 0
    #         en = 1
    #         flag[y][x] = 1
    #         qx[0] = x
    #         qy[0] = y
    #         sumx, sumy = 0, 0
    #         while (st < en) :
    #             xx = qx[st]
    #             yy = qy[st]
    #             sumx += xx
    #             sumy += yy
    #             st = st+1
    #             tot = tot+1
    #             for r in range(4) :
    #                 xxx = xx + dx[r]
    #                 yyy = yy + dy[r]
    #                 if (xxx < xn - delta or xxx > xm + delta or yyy < yn - delta or yyy > ym + delta or flag[yyy][xxx] != 0 or flag[yyy][xxx] != 0 or edges[yyy][xxx] != 0) :
    #                     continue
    #                 flag[yyy][xxx] = 1
    #                 qx[en] = xxx
    #                 qy[en] = yyy
    #                 en = en + 1
    #         # if (en > 1) :
    #         #     print("en = ", en)
    #         if (en > area_s * 1.5 or en < area_s / 1.5) :
    #             continue
    #         for r in range(en):
    #             flag[qy[r]][qx[r]] = 255
    #         cenx[cens] = sumx // en
    #         ceny[cens] = sumy // en
    #         cens = cens + 1
    # print('tot --- ', tot)
    best_answer = 1e8
    hx, hdx = 116, 35
    ix, idx = 116, 35
    hx, hdx, ix, idx = hx*7//scale, hdx*7//scale, ix*7//scale, idx*7//scale
    for x in range(ix-20, ix+20):
        for dx in range(idx-5, idx+6):
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
    hy, hdy = 438, 35
    iy, idy = 438, 35
    hy, hdy, iy, idy = hy*7//scale, hdy*7//scale, iy*7//scale, idy*7//scale
    for y in range(iy-20, iy+20):
        for dy in range(idy-5, idy+6):
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

    print('hx = ' + str(hx) + ' ,hy = ' + str(hy))
    
    # cv2.imshow('Edge Image', edges)
    # cv2.waitKey(0)
    flag = cv2.cvtColor(flag, cv2.COLOR_GRAY2BGR)
    for i in range(6):
        for j in range(4) :
            cv2.circle(flag, (hx+i*hdx, hy+j*hdy), 5, (0, 0, 255), 2)
    cv2.imshow('Edge Image', flag)
    cv2.waitKey(0)

def get_case_bottom_line(hx, hy, hdx, hdy, scale) :
    HDY, UP = 33*7//scale, 25*7//scale
    by = hy - (UP * hdy) // HDY
    HDX, LEFT = 33*7//scale, 34*7//scale
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
    return thresh.astype(np.uint8)

def get_case_upper_line(gradient_img, bx, by, scale) :
    xn, xm, yn, ym, area_s = bx + 50*7//scale, 356*7//scale, 20*7//scale, by - 50*7//scale, 500*7//scale*7//scale

    crop_img = gradient_img[yn:ym, xn:xm]

    # cv2.imshow('detection Image', crop_img)
    # cv2.waitKey(0)
    # Find connected components (blobs) in the image
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(crop_img, connectivity=8)

    # # Create an empty image to draw the filtered blobs
    filtered = np.zeros_like(crop_img)
    
    # # Iterate over all blobs
    ux, uy = 0, ym-yn
    for i in range(1, num_labels):  # Start from 1 to ignore the background
        # Get the area of the blob
        area = stats[i, cv2.CC_STAT_AREA]
        sTop = stats[i, cv2.CC_STAT_TOP]
        sRight = stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH]
        if area_s < area:
            filtered[labels == i] = 255
            if ux < sRight:
                ux = sRight
            if uy > sTop:
                uy = sTop
    # gradient_img = np.zeros_like(gradient_img)
    # gradient_img[yn:ym, xn:xm] = 255*np.ones_like(filtered)
    # cv2.imshow('filtered Image', filtered)
    # cv2.waitKey(0)
    return ux+xn, uy+yn

    flag = np.zeros((4032//7, 3024//7), dtype=np.uint8) 
    qx, qy = np.zeros(432*576, dtype = int), np.zeros(432*576, dtype = int)
    dx, dy = [-1, 0, 1, -1, 1, -1, 0, 1], [-1, -1, -1, 0, 0, 1, 1, 1]

    start_time = time.perf_counter()
    ux, uy = bx, by
    for x in range(xn, xm) :
        for y in range(yn, ym) :
            if (flag[y][x] != 0 or gradient_img[y][x] == 0) :
                continue
            # flag[y][x] = x*y%255
            # continue
            st = 0
            en = 1
            flag[y][x] = 1
            qx[0] = x
            qy[0] = y
            while (st < en) :
                xx = qx[st]
                yy = qy[st]
                st = st+1
                for r in range(8) :
                    xxx = xx + dx[r]
                    yyy = yy + dy[r]
                    if (xxx < xn or xxx > xm or yyy < yn or yyy > ym or flag[yyy][xxx] != 0 or gradient_img[yyy][xxx] == 0) :
                        continue
                    flag[yyy][xxx] = 1
                    qx[en] = xxx
                    qy[en] = yyy
                    en = en + 1
            # if (en > 1) :
            #     print("en = ", en)
            if (en < area_s) :
                continue
            for r in range(en):
                if (ux < qx[r]) :
                    ux = qx[r]
                if (uy > qy[r]) :
                    uy = qy[r]
    
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    print(f'6666The function took {elapsed_time} seconds to complete.')
    
    return ux, uy

def process(img, case_width, case_height, delta, scale):
    img = my_resize_image(img, 1./scale)
    # Convert the image to gray scale
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # blue_channel = img[:, :, 0]
    # grren_channel = img[:, :, 1]
    # red_channel = img[:, :, 2]

    # Apply Canny Edge Detection
    edges = cv2.Canny(img_gray, 100, 200)
    # cv2.imshow('edges Image', edges)
    # cv2.waitKey(0)

    # start_time = time.perf_counter()
    hx, hdx, hy, hdy = get_color_squares(edges, scale)
    # end_time = time.perf_counter()
    # elapsed_time = end_time - start_time
    # print(f'get color squares-The function took {elapsed_time} seconds to complete.')
    print(hx, hdx, hy, hdy)
    bx, by = get_case_bottom_line(hx, hy, hdx, hdy, scale)

    gradient = get_gradient(img)
    # gradient = cv2.Canny(img_gray, 10, 200)  

    ux, uy = get_case_upper_line(gradient, bx, by, scale)

    cv2.line(img, (bx, by), (bx, uy), (0, 255, 0), 3)
    cv2.line(img, (bx, by), (ux, by), (0, 255, 0), 3)
    cv2.line(img, (ux, uy), (bx, uy), (0, 255, 0), 3)
    cv2.line(img, (ux, uy), (ux, by), (0, 255, 0), 3)
    cv2.circle(img, ((bx+ux)//2, (by+uy)//2), 3, (0, 0, 255), 3)
    for i in range(6):
        for j in range(4) :
            cv2.circle(img, (hx+i*hdx, hy+j*hdy), 5, (0, 0, 255), 2)
    return img

    # visualize the binary image
    # thresh = my_resize_image(thresh, 0.25)

    # cv2.imshow('Binary image', thresh)
    cv2.waitKey(0)
    cv2.imwrite(os.path.join(out_path, 'image_thres1.jpg'), thresh)
    cv2.destroyAllWindows()

    return img_gray