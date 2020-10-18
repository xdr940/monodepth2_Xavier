import cv2
import matplotlib.pyplot as plt
from utils.logger import Writer
import threading
import time
cap = cv2.VideoCapture()
import time
cap.open("/dev/mycamera" )
writer = Writer()
duration=1



def cam():
    global duration
    while (1):
        before_op_time = time.time()

        ret, frame = cap.read()


        cv2.imshow('frae',frame)

        if cv2.waitKey(1) & 0xff == ord('q'):
            break
        duration = time.time() - before_op_time
    #print("fps {%2f}".format(1/duration))


def temout():
    global duration
    while True:
        writer.write("fps {:.2f}".format(1 / duration))
        time.sleep(1)
if __name__ == '__main__':
    t1 = threading.Thread(target=cam)
    t2 = threading.Thread(target=cam)
    t3 = threading.Thread(target=cam)
    t4 = threading.Thread(target=cam)
    t5 = threading.Thread(target=cam)

    t0 = threading.Thread(target=temout)
    t1.start()
    t2.start()
    t3.start()
    t4.start()
    t5.start()

    cam()