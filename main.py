import cv2
import numpy as np
import imutils
import functions

cam = cv2.VideoCapture(0)

lower = np.array([0, 48, 80], dtype = "uint8")
upper = np.array([20, 255, 255], dtype = "uint8")

while(cam.isOpened()):
    _, frame = cam.read()
    frame = functions.draw_hand_rect(frame)
    hist = functions.set_hand_hist(frame)
    frame = functions.apply_hist_mask(frame, hist)
    cv2.imshow('frame', frame)













    # frame = imutils.resize(frame, width = 400)
    # converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # skinMask = cv2.inRange(converted, lower, upper)
	# # apply a series of erosions and dilations to the mask
    # # using an elliptical kernel
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    # skinMask = cv2.erode(skinMask, kernel, iterations = 2)
    # skinMask = cv2.dilate(skinMask, kernel, iterations = 2)
	# # blur the mask to help remove noise, then apply the
    # # mask to the frame
    # skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
    # skin = cv2.bitwise_and(frame, frame, mask = skinMask)
    # # show the skin in the image along with the mask
    # cv2.imshow("images", np.hstack([frame, skin]))
    # # if the 'q' key is pressed, stop the loop
    # if cv2.waitKey(1) & 0xFF == ord("q"):
    #     break
camera.release()
cv2.destroyAllWindows()
