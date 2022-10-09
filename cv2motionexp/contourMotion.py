from turtle import color
import numpy as np
import cv2

maxA = 900
vid = cv2.VideoCapture(0)

while True:
    ret, frame1 = vid.read()
    ret, frame2 = vid.read()

    diff = cv2.absdiff(frame1, frame2)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(diff_gray, (5, 5), 0)

    _,thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations = 3)

    conts, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for c in conts:
        (x, y, w ,h) = cv2.boundingRect(c)
        if cv2.contourArea(c) < maxA:
            continue
        cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame1, "Status: {}".format('Movement'), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    cv2.imshow("Video", frame1)
    frame1 = frame2
    ret, frame2 = vid.read()

    if (cv2.waitKey(1) & 0xFF == ord('Q')):
        break

vid.release()
cv2.destroyAllWindows()

"""
NOTES: 2/7/22
This is the most effective
Have CNN that identifies face in the frame then tracks its motion w/ field instead of all motion in the frame: 
https://www.youtube.com/watch?v=hfXMw2dQO4E
also have timer that sends shit over after 5 seconds
use mongoose w/ mongodb to send image w/ user password and auth key/username

"""