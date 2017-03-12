import cv2
import numpy as np
# import pyrebase

# config = {
#   "apiKey": "AIzaSyAj8ohVERrClVgWkQv2aR9_ZfrbwvgndAs",
#   "authDomain": "http://eyepatch-1b6f6.firebaseapp.com/",
#   "databaseURL": "https://eyepatch-1b6f6.firebaseio.com/",
#   "storageBucket": "http://eyepatch-1b6f6.appspot.com/",
#   "messagingSenderId": "820178481871"
# }

# storage = firebase.storage()
# storage.child("images/pic.jpg").download("dpic1.jpg")



image = cv2.imread('cataract.jpg')
cv2.imshow('original', image)
cv2.waitKey(0)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

blur = cv2.medianBlur(gray, 5)
cropSize = (100, 100)
circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1.5, 10)
if(circles != None):
    circles = np.uint16(np.around(circles))
    print(circles)
    for x,i in enumerate(circles[0,:2]):
        cv2.circle(image,(i[0], i[1]), i[2], (255, 0, 0), 2)
        cv2.circle(image, (i[0], i[1]), 2, (0, 255, 0), 5)
        if x == 0:
            cropCoords = (max(0, i[1]-cropSize[0]//2),min(image.shape[0], i[1]+cropSize[0]//2), max(0, i[0]-cropSize[1]//2),min(image.shape[1], i[0]+cropSize[1]//2)) 
            crop_cimg = image[cropCoords[0]:cropCoords[1], cropCoords[2]:cropCoords[3]]


            cv2.imshow('cropped', crop_cimg)
            cv2.waitKey(0)

    cv2.imshow('detected circles', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print('Pupil not found')

lower_brown = np.array([2, 100, 65])
upper_brown = np.array([12, 170, 100])

lower_grey = np.array([180, 70, 28])
upper_grey = np.array([199,140, 71])

lower_red = np.array([170, 70, 50])
upper_red = np.array([180, 255, 255])

x = [(lower_brown,upper_brown),(lower_grey,upper_grey),(lower_red,upper_red)]

hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
kernel = np.ones((5,5), np.uint8)

for i in x:
    mask = cv2.inRange(hsv_img, i[0], i[1])
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    res = cv2.bitwise_and(image, image, mask=mask)
    im2, contours, hierarchy = cv2.findContours(res[3].copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    print("Number of Contours found = " + str(len(contours)))
    cv2.drawContours(image, contours, -1, (0,255,0), 3)
    cv2.imshow('mask', mask)
    cv2.waitKey(0)
    cv2.imshow('Filtered Color Only', res)
    cv2.waitKey(0)
    cv2.imshow('Contours', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
