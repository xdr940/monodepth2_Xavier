import cv2
import matplotlib.pyplot as plt
cap = cv2.VideoCapture()
import time
cap.open(0)
while (1):
    before_op_time = time.time()

    ret, frame = cap.read()


    cv2.imshow('frae',frame)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break
    duration = time.time() - before_op_time

    #print("fps {%2f}".format(1/duration))
