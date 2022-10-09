from turtle import color
from cv2 import threshold
import numpy as np
import cv2

vid = cv2.VideoCapture(0)
ret, prev = vid.read()

gray1 = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
mask = np.zeros_like(prev)
mask[..., 1] = 255

while True:   
    ret, frame = vid.read()

    cv2.imshow("input", frame)
    gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    next = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(gray1 ,gray, None,0.5, 3, 15, 3, 5, 1.2, 0)

    mag, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    mask[..., 0] = angle * 180/np.pi /2
    mask[..., 1] = 255
    mask[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    
    rgb = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
    
    cv2.imshow("dop", rgb)

    gray1 = gray
    print(mag)

    prev = frame
    ret, frame2 = vid.read()

    if (cv2.waitKey(1) & 0xFF == ord('Q')):
        break

vid.release()
cv2.destroyAllWindows()