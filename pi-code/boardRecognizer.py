import numpy as np
import cv2
import cvHelperFunctions as hp
# import imutils as util
from datetime import datetime

test_image_path = "test-images/IMG-0713.jpg"


test_image = cv2.imread(test_image_path, cv2.IMREAD_COLOR)

max_dim = 500

test_image = hp.shrinkto(test_image, max_dim)
test_image = cv2.medianBlur(test_image, 5)

cv2.imshow("Blurred Image", test_image)
test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2HSV)

test_image[:, :, 1] = 0
test_image = cv2.cvtColor(test_image, cv2.COLOR_HSV2BGR)
test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
#test_image = cv2.adaptiveThreshold(test_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 2)
test_image = cv2.cvtColor(test_image, cv2.COLOR_GRAY2BGR)
ret, test_image = cv2.threshold(test_image, 127, 255, cv2.THRESH_BINARY)
cv2.imshow("Adaptive Threshold Image", test_image)
# cv2.imshow("Not Adaptive", comp_image)



# houghlines test

lines = cv2.HoughLines(hp.create_value_edges(test_image, show=True), 1, np.pi / 180, 150)
if lines is not None:
    for arr in lines:
        # print("Array at 0:")
        # print(arr)
        rho = arr[0][0]
        theta = arr[0][1]
        start, end = hp.points_from_rho_theta(rho, theta)
        cv2.line(test_image, start, end, (0, 0, 255), 2)


cv2.imshow("Test Image", test_image)
cv2.waitKey()
detector = cv2.SimpleBlobDetector()
keypoints = detector.detect()
im_with_keypoints = cv2.drawKeypoints(test_image, keypoints, np.array([]), (0, 0, 255),
                                      cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow("Keypoints", im_with_keypoints)
cv2.waitKey(0)