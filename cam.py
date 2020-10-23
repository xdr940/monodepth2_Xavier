import cv2
import matplotlib.pyplot as plt
from utils.logger import Writer
import threading
import time
cap = cv2.VideoCapture()
import time
cap.open("/dev/video0" )
writer = Writer()
duration=1



def cam():
    global duration
    while (1):
        try:
            t1 = time.time()

            ret, frame = cap.read()

            t2 = time.time()

            cv2.imshow('frame',frame)

            if cv2.waitKey(1) & 0xff == ord('q'):
               break
            duration = t2-t1
        except KeyboardInterrupt:
            return
        except:
            pass



def temout():
    global duration
    while True:
        writer.write("fps: {:.2f}".format( 1/duration))
        time.sleep(1)
if __name__ == '__main__':
    t1 = threading.Thread(target=cam)

    t0 = threading.Thread(target=temout)
    t1.start()
    t0.start()

    cam()