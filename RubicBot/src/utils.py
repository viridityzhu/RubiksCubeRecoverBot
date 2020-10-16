from math import atan2, degrees

import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def visualize(image, mask, original_image=None, original_mask=None, text1='1', text2='2'):
    fontsize = 18

    if original_image is None and original_mask is None:
        f, ax = plt.subplots(1, 2, figsize=(16, 8))

        ax[0].imshow(image)
        ax[0].set_title(text2, fontsize=fontsize)

        ax[1].imshow(mask)
        ax[1].set_title(text1, fontsize=fontsize)
    else:
        f, ax = plt.subplots(2, 2, figsize=(8, 8))

        ax[0, 0].imshow(original_image)
        ax[0, 0].set_title('3', fontsize=fontsize)

        ax[1, 0].imshow(original_mask)
        ax[1, 0].set_title('4', fontsize=fontsize)

        ax[0, 1].imshow(image)
        ax[0, 1].set_title('1', fontsize=fontsize)

        ax[1, 1].imshow(mask)
        ax[1, 1].set_title('2', fontsize=fontsize)
    plt.show()


def visualize3(image1, image2, image3, text1='', text2='', text3='', is_show=True):
    fontsize = 12
    f, ax = plt.subplots(1, 3, figsize=(16, 8))

    ax[0].imshow(image1)
    ax[0].set_title(text1, fontsize=fontsize)

    ax[1].imshow(image2)
    ax[1].set_title(text2, fontsize=fontsize)

    ax[2].imshow(image3)
    ax[2].set_title(text3, fontsize=fontsize)

    if is_show:
        plt.show()


def make_gaussian(width, height, sigma=3, center=None):
    x = np.arange(0, width, 1, float)
    y = np.arange(0, height, 1, float)[:, np.newaxis]

    if center is None:
        x0 = width // 2
        y0 = height // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-4 * np.log(2) * ((x - x0)**2 + (y - y0)**2) / sigma**2)


def get_angle(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    dX = x2 - x1
    dY = y2 - y1
    rads = atan2(-dY, dX)

    return degrees(rads)


def is_epsilon_equal(x, y, epsilon=1):
    if abs(x - y) < epsilon:
        return True
    return False


def get_corners(binary_mask, alpha=0.03):
    cnts, _ = cv2.findContours(binary_mask,
                               cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)[-2:]

    if len(cnts) == 0:
        return None

    cnts = sorted(cnts, key=lambda x: cv2.contourArea(x))
    primary_cnt = cnts[0]

    epsilon = alpha * cv2.arcLength(primary_cnt, True)
    approx = cv2.approxPolyDP(primary_cnt, epsilon, True)

    if len(approx) != 6:
        return None

    hull = cv2.convexHull(approx)
    area1 = cv2.contourArea(approx)
    area2 = cv2.contourArea(hull)

    if not is_epsilon_equal(area1, area2, epsilon=1):
        return None

    return approx


def four_points_resize(image, points, size):
    src_pts = np.array(points, dtype=np.float32)
    dst_pts = np.array([[0, 0], [size, 0], [size, size],
                        [0, size]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warp = cv2.warpPerspective(image, M, (size, size))
    return warp


def get_center_point(binary_image):
    cnts, _ = cv2.findContours(binary_image.copy(),
                               cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)[-2:]

    if len(cnts) != 1:
        return None

    primary_cnt = cnts[0]

    x, y, w, h = cv2.boundingRect(primary_cnt)

    return x + w // 2, y + h // 2


def is_hexagon_valid(points, center_point, epsilon=12):
    order = [
        ((points[1], center_point), (points[2], points[3])),
        ((points[2], points[1]), (points[3], center_point)),
        ((points[1], center_point), (points[0], points[5])),
        ((points[1], points[0]), (center_point, points[5])),
        ((center_point, points[5]), (points[3], points[4])),
        ((center_point, points[3]), (points[5], points[4]))
    ]

    for item in order:
        angle1 = get_angle(item[0][0], item[0][1])
        angle2 = get_angle(item[1][0], item[1][1])
        if not is_epsilon_equal(angle1, angle2, epsilon=epsilon):
            return False

    return True
