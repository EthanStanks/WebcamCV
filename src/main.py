import cv2

if __name__ == '__main__':
    # default camera
    webcam = cv2.VideoCapture(0)

    while True:
        is_captured, frame = webcam.read()
        if is_captured:
            cv2.imshow('Webcam', frame)

            key = cv2.waitKey(1)
            if key == 27: # esc
                break

    webcam.release()
    cv2.destroyAllWindows()