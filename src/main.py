import cv2
import math
import numpy as np
import os

VIDEO_NAME = 'test'

if __name__ == '__main__':
    # default camera
    webcam = cv2.VideoCapture(0)

    #  set width and height
    frame_width = int(webcam.get(3))
    frame_height = int(webcam.get(4)) 
    frame_size = (frame_width, frame_height)

    # init
    frame_count = 0

    while True:
        is_captured, frame = webcam.read()
        frame_count += 1
        if not is_captured:
            break

        cv2.imshow('Webcam', frame)


        key = cv2.waitKey(1)
        if key == 27: # esc
            break

    webcam.release()
    cv2.destroyAllWindows()