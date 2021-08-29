import numpy as np
import cv2
import sklearn.cluster as sk

def create_value_edges(img, show=False):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hue = hsv_img.copy()
    # remove all saturation
    hue[:, :, 1] = 0

    edges = cv2.Canny(hue, 100, 150, apertureSize=3)
    if show:
        cv2.imshow('Current frame value', cv2.cvtColor(hue, cv2.COLOR_HSV2BGR))
        cv2.imshow('edges', edges)
    return edges


def bold_image(img, bold_color = [255, 255, 255], width = 1):
    '''Extends every pixel of bold_color to the surrounding width pixels.'''
    copy_img = img.copy()
    arr_img = np.array(img)
    # print(arr_img.shape)
    for i in range(width, arr_img.shape[0] - width):
        for j in range(width, arr_img.shape[1] - width):
            pixel = arr_img[i][j]
            # print("Pixel: {0}".format(pixel))
            if pixel[0] == bold_color[0] and pixel[1] == bold_color[1] and pixel[2] == bold_color[2]:
                for k in range(j - width, j + width):
                    for l in range (i - width, i + width):
                        copy_img[l][k] = bold_color

    return img

def shrinkto(img, max_dim):
    print("{}", img.shape)
    larger = 0
    if img.shape[0] > img.shape[1]:
        larger = img.shape[0]
    else:
        larger = img.shape[1]

    if larger > max_dim:
        scale_factor = max_dim / larger
        img = cv2.resize(img, None, img.size, scale_factor, scale_factor, cv2.INTER_LINEAR)
    return img


def posterize(img,  n_colors):
    h = img.shape[0]
    w = img.shape[1]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    cv2.imshow("LAB", img)
    img = img.reshape((h * w, 3))
    clt = sk.MiniBatchKMeans(n_clusters=n_colors)
    labels = clt.fit_predict(img)
    img = clt.cluster_centers_.astype("uint8")[labels]
    img = img.reshape((h, w, 3))
    # convert from L*a*b* to RGB
    img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
    cv2.imshow("LAB", img)
    return img


def points_from_rho_theta(rho, theta, debug=False, high_val=1000):
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    # Slope is (-b / a)
    # slope is (-np.tan(theta))
    # initial value is (cos(theta)rho)
    x1 = int(x0 + high_val * (-b))
    y1 = int(y0 + high_val * (a))
    x2 = int(x0 - high_val * (-b))
    y2 = int(y0 - high_val * (a))
    if debug:
        print(f'Initial value ({x0}, {y0}) generates ({x1}, {y1}) to ({x2},{y2})')
    return ((x1, y1), (x2, y2))
