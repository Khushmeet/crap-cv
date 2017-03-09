import cv2

def get_histogram(frame):
    # histogram rectangle
    frame = cv2.resize(frame, None, fx=0.75, fy=0.75)
    rows,cols,_ = frame.shape
    hand_row_nw = np.array([6*rows/20,6*rows/20,6*rows/20,
								10*rows/20,10*rows/20,10*rows/20,
								14*rows/20,14*rows/20,14*rows/20])

	hand_col_nw = np.array([9*cols/20,10*cols/20,11*cols/20,
							9*cols/20,10*cols/20,11*cols/20,
						    9*cols/20,10*cols/20,11*cols/20])
    hand_row_se = hand_row_nw + 10
    hand_col_se = hand_col_nw + 10

    size = hand_row_nw.size
    for i in range(size):
        cv2.rectangle(frame,(hand_col_nw[i],hand_row_nw[i]),(hand_col_se[i],hand_row_se[i]),(0,255,0),1)

    cv2.imshow('image', frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    roi = np.zeros([90,10,3], dtype=hsv.dtype)
    for i in xrange(size):
		roi[i*10:i*10+10,0:10] = hsv[hand_row_nw[i]:hand_row_nw[i]+10,
                                    hand_col_nw[i]:hand_col_nw[i]+10]
    hand_hist = cv2.calcHist([roi],[0, 1], None, [180, 256], [0, 180, 0, 256])																		
	cv2.normalize(hand_hist, hand_hist, 0, 255, cv2.NORM_MINMAX)
    return hand_hist
