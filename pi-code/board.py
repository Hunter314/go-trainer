import numpy as np
import cv2
import imutils
import cvHelperFunctions as hp

def board_image(image, corners=None, size=500):
    # get the corners of this board
    cv2.imshow("Before Board", image)
    if corners is not None:
        # do stuff
        source_triangle = np.array([corners[0], corners[3], corners[1]])
        destination_triangle = np.array([[0, 0], [size - 1, 0], [0, size - 1]])
    print(source_triangle)
    print(destination_triangle)
    warp_mat = cv2.getAffineTransform(source_triangle.astype(np.float32)
, destination_triangle.astype(np.float32))
    warp_dst = cv2.warpAffine(image, warp_mat, (size, size))

    # center = (warp_dst.shape[1]//2, warp_dst.shape[0]//2)
    cv2.imshow("Board", warp_dst)
    return image


def process(source_image, blur_amount=7):
    test_image = source_image.copy()
    test_image = cv2.medianBlur(test_image, blur_amount)

    cv2.imshow("Blurred Image", test_image)
    test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2HSV)

    test_image[:, :, 1] = 0
    test_image = cv2.cvtColor(test_image, cv2.COLOR_HSV2BGR)
    test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
    test_image = cv2.adaptiveThreshold(test_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 8)
    test_image = cv2.cvtColor(test_image, cv2.COLOR_GRAY2BGR)
    cv2.imshow("Adaptive Threshold Image", test_image)
    return test_image


def get_corners(source_image, rect_ratio_error=0.02, debug=False):
    contours_image = process(source_image, 13).copy()
    cv2.imshow("Contours Image", contours_image)
    # cv2.floodFill(contours_image, None, (0, 0), 0)
    hp.fill_all_edges_black(contours_image)
    cv2.imshow("After Floodfill", contours_image)

    contours = cv2.findContours(cv2.cvtColor(contours_image, cv2.COLOR_BGR2GRAY), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    rects = []

    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        M = cv2.moments(c)
        # look for largest square so save width and height
        max_width = 0
        max_height = 0

        if len(approx) == 4:
            # rectangle
            print("Found a rectangle")
            print(approx)
            width = np.linalg.norm(approx[0] - approx[1])
            height = np.linalg.norm(approx[0] - approx[3])
            # find the largest square
            # determine if its a square by comparing width height ratio
            print("Rectangle size is ({0}, {1})".format(width, height))
            if abs(width / height - 1) < rect_ratio_error:
                print("Made it here")
                if width > max_width:
                    print("Made it here")
                    max_width = width
                    corners = approx
                    print("Corners: {0}".format(corners))
        else:
            continue
        # compute the center of the contour


    print("Corners:{0}".format(corners))
    return corners



def board_state(image, corners=None):
    spots = np.zeros((19, 19))
    board_img = board_image(image, corners=None, size=500)
    tile_width = image.shape[0] / 19.0
    for row in spots:
        for col in row:
            print(col)


class Board:
    spots = np.zeros((19, 19))
    corners = {}



    def __init__(self, empty_board_path=None):
        if empty_board_path is None:
            # Try it without the empty board path
            self.spots[0, 0] = 1

    def __str__(self):
        return "Board: {0}".format(self.spots)

