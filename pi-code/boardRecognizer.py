import imutils
import numpy as np
import cv2
import cvHelperFunctions as hp
import board
from datetime import datetime


test_image_path = "test-images/IMG-0715.jpg"


source_image = cv2.imread(test_image_path, cv2.IMREAD_COLOR)
max_dim = 1000

source_image = hp.shrinkto(source_image, max_dim)


def contours_test(contours_image):


    contours = cv2.findContours(cv2.cvtColor(contours_image, cv2.COLOR_BGR2GRAY), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    for c in contours:
        # compute the center of the contour
        M = cv2.moments(c)
        if M["m00"] == 0:
            cX = 1000
            cY = 1000
        else:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        # draw the contour and center of the shape on the image
        cv2.drawContours(contours_image, [c], -1, (0, 255, 0), 2)
        cv2.circle(contours_image, (cX, cY), 7, (255, 255, 255), -1)
        cv2.putText(contours_image, "center", (cX - 20, cY - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        # show the image
        cv2.imshow("Image", contours_image)

# houghlines test


def houghlines_test():
    test_image = board.process(source_image)
    lines = cv2.HoughLines(hp.create_value_edges(test_image, show=True), 1, np.pi / 180, 120)
    if lines is not None:
        for arr in lines:
            # print("Array at 0:")
            # print(arr)
            rho = arr[0][0]
            theta = arr[0][1]
            start, end = hp.points_from_rho_theta(rho, theta)
            cv2.line(test_image, start, end, (0, 0, 255), 2)
    cv2.imshow("Hough Lines", test_image)


def board_test():
    my_board = board.Board(None)
    print(my_board)

    # board.board_image(source_image, [[89, 152], [94, 462], [403, 142]], max_dim)
    board.board_image(source_image, board.get_corners(source_image), max_dim)


def blob_detection_test():
    detector = cv2.SimpleBlobDetector()
    #keypoints = detector.detect()
    #im_with_keypoints = cv2.drawKeypoints(test_image, keypoints, np.array([]), (0, 0, 255),
    #                                      cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)



new_image = source_image.copy()
# houghlines_test()

contours_image = board.process(source_image, 9).copy()
cv2.imshow("Contours Image", contours_image)
cv2.floodFill(contours_image, None, (0, 0), 0)
contours_test(contours_image)
cv2.waitKey()
board_test()
#contours_test()
#houghlines_test()
board.get_corners(source_image)
cv2.waitKey(0)

