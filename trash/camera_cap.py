
from path import Path
import datetime
import cv2
import time
import os
import threading
from threading import Thread
class Camera():
    def __init__(self):
        self.cap = cv2.VideoCapture()  # 视频流
        self.dst_dir = Path("./camera_cap")
        self.dst_dir.mkdir_p()
        self.temp_length = 100#暂存图片数量
        self.cam_freq = 25
        self.resize = [640,480]
        self.CAM_NUM = 0
        flag = self.cap.open(self.CAM_NUM)
        if not flag:  # flag表示open()成不成功
            print("==> camera open error")



        pass
    def capture(self):



        while(True):
            before_op_time = time.time()

            flag, img = self.cap.read()
            #img_name = datetime.datetime.now().strftime("%m_%d-%H:%M:%S.%f.jpg")

            img_name = datetime.datetime.now().strftime("%f.jpg")
            cv2.imwrite(self.dst_dir/img_name, img)
            self.duration = time.time() - before_op_time


        pass
    def remove(self):
        files = self.dst_dir.files()
        files.sort()
        del_num = len(files)-self.temp_length
        if del_num<0:
            return
        else:
            for item in files[:del_num]:
                os.remove(item)

        time.sleep(0.1)

        #files.remove([])

    def run(self):
        cnt = 0


        print("==> camera temp length {}".format(self.temp_length))
        print("==> camera NUM {}".format(self.CAM_NUM))
        print("==> captrue save dir {}".format(self.dst_dir))
        t1 = threading.Thread(target=self.capture())
        t2 = threading.Thread(target=self.remove())

        t1.start()
        t2.start()








if __name__ == '__main__':
    pass
    cam = Camera()
    cam.run()
    #cam.capture()
    #cam.remove()