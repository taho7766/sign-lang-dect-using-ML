import numpy as np
import cv2 as cv

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Error opening camera")
    exit()
while(cap.isOpened()):
    ret, frame = cap.read()
    
    if(ret == False):
        print("Capured frame is invalid")
        break
    
    cv.imshow('Testing', frame)
    if (cv.waitKey(1) == ord('q')):
        break

cap.release()
cv.destroyAllWindows