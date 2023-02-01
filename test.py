from __future__ import division

import cv2
import numpy as np
from matplotlib import pyplot as pl5

original = cv2.imread('original.png')
projected = cv2.imread('project102.jpg')
#image = cv2.resize(original, (600, 400))
projected = cv2.resize(projected, (0,0), fx=0.25, fy=0.25)
#projected = cv2.resize(projected, (600, 400))
image = original
oHeight, oWidth, oChannels = image.shape
pHeight, pWidth, pChannels = image.shape
#rows, cols, ch = image.shape
coord = []

w, h = 2, 4

redPoints = [[0 for j in range(w)] for i in range(h)]
bluePoints = np.array([
    [0, 0],
    [pWidth - 1, 0],
    [pWidth - 1, pHeight - 1],
    [0, pHeight - 1]], dtype="float32")

counter = 0
def mouseCallback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(coord) < 4:
        global counter

        coord.append((x, y))
        redPoints[counter][0] = x
        redPoints[counter][1] = y
        counter = counter + 1

        cv2.circle(projected, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("image", projected)
def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect

def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    '''
    widthA = np.sqrt(((br[0] - bl[0]) **2) + ((br[1] - bl[1]) **2))
    widthB = np.sqrt(((tr[0] - tl[0]) **2) + ((tr[1] - tl[1]) **2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) **2) + ((tr[1] - br[1]) **2))
    heightB = np.sqrt(((tl[0] - bl[0]) **2) + ((tl[1] - bl[1]) **2))
    maxHeight = max(int(heightA), int(heightB))

    print(maxWidth, maxHeight)

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    '''
    maxWidth = pWidth
    maxHeight = pHeight
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    dst = order_points(dst)

    # compute the perspective transform matrix and then apply it
    #M = cv2.getPerspectiveTransform(rect, dst)
    M = cv2.getPerspectiveTransform(dst, rect)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    #cv2.resize(warped, (1600, 900))
    cv2.imshow("warped", warped)
    #cv2.resize(warped, (1600, 900))

    # return the warped image
    return warped

cv2.imshow('image', projected)
cv2.setMouseCallback("image", mouseCallback)
cv2.waitKey(0)

sideA = abs(redPoints[0][1] - redPoints[3][1])
sideB = abs(redPoints[1][1] - redPoints[2][1])

# sideA is long else sideB is long; (longer / shorter) = 1,...
print "sides: ", sideA, sideB
print redPoints
rate = sideA / sideB if (sideA < sideB) else sideB / sideA
print rate, sideA / sideB
# top right corner
bluePoints[1][1] = pHeight - (pHeight * rate)
for index in range(len(bluePoints)):
    cv2.circle(image, (bluePoints[index][0], bluePoints[index][1]), 20, (0, 255, 0), -1)
cv2.imshow('image', image)
print bluePoints

warped = four_point_transform(image, bluePoints)
cv2.resize(warped, (1280, 800))
cv2.imwrite("output.jpg", warped)
cv2.waitKey(0)
cv2.destroyAllWindows()
