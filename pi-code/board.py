import numpy as np
import cv2


def board_image(image, corners=None, size=500):
    # get the corners of this board
    cv2.imshow("Before Board", image)
    if corners is not None:
        # do stuff
        source_triangle = np.array([corners[0], corners[1], corners[2]])
        destination_triangle = np.array([[0, 0], [size - 1, 0], [0, size - 1]])
    print(source_triangle)
    print(destination_triangle)
    warp_mat = cv2.getAffineTransform(source_triangle.astype(np.float32)
, destination_triangle.astype(np.float32))
    warp_dst = cv2.warpAffine(image, warp_mat, (size, size))

    # center = (warp_dst.shape[1]//2, warp_dst.shape[0]//2)
    cv2.imshow("Board", warp_dst)
    return image


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

