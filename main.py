import cv2
import numpy as np
import utils

cam = cv2.VideoCapture(0)
if cam == None:
    print('Error reading webcam')

while True:
    _, frame = cam.read()
    # to get hand histogram
    hist = utils.get_histogram(frame)
    cam.release()
    cv2.destroyAllWindows()
