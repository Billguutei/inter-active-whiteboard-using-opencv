from __future__ import division
from transform import four_point_transform
import cv2
import numpy as np
from matplotlib import pyplot as pl5

coord = []
counter = 0

w, h = 2, 4
redPoints = [[0 for j in range(w)] for i in range(h)]
bluePoints = [[0 for j in range(w)] for i in range(h)]

def mouseCallback(event, x, y, flags, param):
    global counter

    if event == cv2.EVENT_LBUTTONDOWN and len(coord) < 4:
        coord.append((x, y))
        redPoints[counter][0] = x
        redPoints[counter][1] = y
        counter = counter + 1
        cv2.circle(projected, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("projected", projected)
    elif event == cv2.EVENT_LBUTTONDOWN and len(coord) < 8:
        if counter == 4:
            counter = 0
        coord.append((x, y))
        bluePoints[counter][0] = x
        bluePoints[counter][1] = y
        counter = counter + 1
        cv2.circle(projected, (x, y), 5, (255, 0, 0), -1)
        cv2.imshow("projected", projected)


original = cv2.imread('original.png')
projected = cv2.imread('project10.jpg')
#image = cv2.resize(original, (600, 400))
projected = cv2.resize(projected, (0,0), fx=0.25, fy=0.25)
#projected = cv2.resize(projected, (600, 400))
#image = original
oHeight, oWidth, oChannels = original.shape
pHeight, pWidth, pChannels = projected.shape
#rows, cols, ch = image.shape



dst = np.array([
    [0, 0],
    [oWidth - 1, 0],
    [oWidth - 1, oHeight - 1],
    [0, oHeight - 1]], dtype="float32")



cv2.imshow('projected', projected)
cv2.setMouseCallback("projected", mouseCallback)
cv2.waitKey(0)

bigA = abs(redPoints[0][1] - redPoints[3][1])
bigB = abs(redPoints[1][1] - redPoints[2][1])
lilA = abs(bluePoints[0][1] - bluePoints[3][1])
lilB = abs(bluePoints[1][1] - bluePoints[2][1])

rateA = lilA / bigA
rateB = lilB / bigB

dst[1][1] = oHeight - (oHeight * rateB)
dst[3][1] = oHeight * rateA
for index in range(len(dst)):
    cv2.circle(original, (int(dst[index][0]), int(dst[index][1])), 20, (0, 255, 0), -1)
cv2.imshow('original', original)
print bluePoints

warped = four_point_transform(original, dst)
#cv2.resize(warped, (1280, 800))
#cv2.resize(warped, (640, 400))
cv2.resize(warped, (640, 400))
cv2.imshow("warped", warped)

cv2.imwrite("output.jpg", warped)
cv2.waitKey(0)
cv2.destroyAllWindows()
