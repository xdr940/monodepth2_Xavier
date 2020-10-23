import sys
import cv2
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMainWindow, QPushButton, QApplication, QWidget, QLabel, QMessageBox


class Ui_MainWindow(QMainWindow):
    def __init__(self):
        super(Ui_MainWindow, self).__init__()
        self.initUi()

    def initUi(self):
        self.num=0
        self.timer_camera = QTimer()  # 定义定时器，用于控制显示视频的帧率
        self.timer_camera2 = QTimer()  # 定义定时器，用于控制显示视频的帧率
        self.cap = cv2.VideoCapture()  # 视频流
        self.CAM_NUM = "/dev/mycamera"  # 为0时表示视频流来自笔记本内置摄像头
        self.resize(705, 485)
        self.setWindowTitle("camera")
        self.centralwidget = QWidget()
        self.pushButton_open = QPushButton(self.centralwidget)
        self.pushButton_open.setText("打开视频")
        self.pushButton_open.setGeometry(2, 120, 55, 40)
        self.pushButton_close = QPushButton(self.centralwidget)
        self.pushButton_close.setText("退出")
        self.pushButton_close.setGeometry(2, 260, 55, 40)
        self.label_show_camera = QLabel(self.centralwidget)  # 定义显示视频的Label
        self.label_show_camera.setFixedSize(640, 480)
        self.label_show_camera.setGeometry(60, 0, 700, 480)
        self.setCentralWidget(self.centralwidget)
        self.pushButton_open.clicked.connect(self.open_camera)
        self.timer_camera.timeout.connect(self.show_camera)
        self.timer_camera2.timeout.connect(self.show_rate)
        self.pushButton_close.clicked.connect(self.close)

    def open_camera(self):
        if not self.timer_camera.isActive():  # 若定时器未启动
            # 参数是0，表示打开笔记本的内置摄像头，参数是视频文件路径则打开视频
            flag = self.cap.open(self.CAM_NUM)
            if not flag:  # flag表示open()成不成功
                msg = QMessageBox.warning(self, 'warning', "请检查相机于电脑是否连接正确", buttons=QMessageBox.Ok)
                print(msg)
            else:
                self.timer_camera.start(10)  # 定时器开始计时30ms，结果是每过30ms从摄像头中取一帧显示
                self.timer_camera2.start(1000)
                self.pushButton_open.setText('关闭相机')
        else:
            self.timer_camera.stop()  # 关闭定时器
            self.cap.release()  # 释放视频流
            self.label_show_camera.clear()  # 清空视频显示区域
            self.pushButton_open.setText('打开相机')

    def show_camera(self):
        flag, self.image = self.cap.read()  # 从视频流中读取
        show = cv2.resize(self.image, (640, 192))  # 把读到的帧的大小重新设置为 640x480
        show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)  # 视频色彩转换回RGB，这样才是现实的颜色
        cv2.imwrite('picture/test.jpg', show)
        show = cv2.resize(self.image, (640, 480))  # 把读到的帧的大小重新设置为 640x480
        show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)  # 视频色彩转换回RGB，这样才是现实的颜色
        show_image = QImage(show.data, show.shape[1], show.shape[0], QImage.Format_RGB888)  # 把读取到的视频数据变成QImage形式
        self.label_show_camera.setPixmap(QPixmap.fromImage(show_image))  # 往显示视频的Label里 显示QImage
        self.num = self.num+1

    def show_rate(self):
        print("视频帧率："+str(self.num))
        self.num = 0


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = Ui_MainWindow()
    win.show()
    sys.exit(app.exec_())
