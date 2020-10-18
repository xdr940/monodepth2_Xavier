from __future__ import absolute_import, division, print_function

import os
import sys
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import networks
from networks.layers import disp_to_depth

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QLabel


class Ui_MainWindow(QMainWindow):
    def __init__(self):
        super(Ui_MainWindow, self).__init__()
        self.initUi()

    def initUi(self):
        self.num=0
        self.timer_camera = QTimer()  # 定义定时器，用于控制显示视频的帧率
        self.timer_camera2 = QTimer()
        self.resize(640, 480)
        self.setWindowTitle("test_simple")
        self.centralwidget = QWidget()
        self.label_show_camera = QLabel(self.centralwidget)  # 定义显示视频的Label
        self.label_show_camera.setFixedSize(640, 480)
        self.label_show_camera.setGeometry(0, 0, 640, 480)
        self.setCentralWidget(self.centralwidget)
        self.timer_camera.timeout.connect(self.__show_camera__)
        self.timer_camera2.timeout.connect(self.__show_rate__)

        self.picture_path = "./picture/test.jpg"
        #self.encoder_path = "/home/wang/models/mono+stereo_640x192/encoder.pth"
        #self.depth_decoder_path = "/home/wang/models/mono+stereo_640x192/depth.pth"

        self.encoder_path = "/home/roit/models/monodepth2_official/mono_640x192/encoder.pth"
        self.depth_decoder_path = "/home/roit/models/monodepth2_official/mono_640x192/depth.pth"

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        print("-> device:",self.device)

        self.encoder = networks.ResnetEncoder(18, False)
        self.loaded_dict_enc = torch.load(self.encoder_path, map_location=self.device)

        self.feed_height = self.loaded_dict_enc['height']
        self.feed_width = self.loaded_dict_enc['width']
        self.filtered_dict_enc = {k: v for k, v in self.loaded_dict_enc.items() if k in self.encoder.state_dict()}


        self.encoder.load_state_dict(self.filtered_dict_enc)
        self.encoder.to(self.device)
        self.encoder.eval()


       #decoder
        self.depth_decoder = networks.DepthDecoder2([64, 64, 128, 256, 512])
        self.loaded_dict_dec = torch.load(self.depth_decoder_path, map_location=self.device)
        self.filtered_dict_dec = {k: v for k, v in self.loaded_dict_dec.items() if k in self.depth_decoder.state_dict()}

        self.depth_decoder.load_state_dict(self.filtered_dict_dec)
        self.depth_decoder.to(self.device)
        self.depth_decoder.eval()

        #
        self.paths = [self.picture_path]
        self.output_directory = os.path.dirname(self.picture_path)

        self.timer_camera.start(10 )
        self.timer_camera2.start(1000)



    def __show_camera__(self):
        self.__test_simple__()
        #self.image = cv2.imread('./picture/test_disp.jpeg')
        #show = cv2.resize(self.image, (640, 480))  # 把读到的帧的大小重新设置为 640x480
        #show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)  # 视频色彩转换回RGB，这样才是现实的颜色
        #show_image = QImage(show.data, show.shape[1], show.shape[0], QImage.Format_RGB888)  # 把读取到的视频数据变成QImage形式
        #self.label_show_camera.setPixmap(QPixmap.fromImage(show_image))  # 往显示视频的Label里 显示QImage
        self.num = self.num + 1

    def __show_rate__(self):
        print("视频帧率："+str(self.num))
        self.num = 0

    def __test_simple__(self):

        with torch.no_grad():
            for idx, image_path in enumerate(self.paths):

                # if image_path.endswith("_disp.jpg"):
                #     # don't try to predict disparity for a disparity image!
                #     continue

                # Load image and preprocess
                try:
                    input_image = pil.open(image_path).convert('RGB')
                    original_width, original_height = input_image.size
                    input_image = input_image.resize((self.feed_width, self.feed_height), pil.LANCZOS)
                    input_image = transforms.ToTensor()(input_image).unsqueeze(0)

                    # PREDICTION
                    input_image = input_image.to(self.device)
                    features = self.encoder(input_image)
                    disp = self.depth_decoder(features[0],features[1],features[2],features[3],features[4])

                    #disp = outputs[("disp", 0)]
                    disp_resized = torch.nn.functional.interpolate(
                        disp, (original_height, original_width), mode="bilinear", align_corners=False)

                    # Saving numpy file
                    output_name = os.path.splitext(os.path.basename(image_path))[0]
                    name_dest_npy = os.path.join(self.output_directory, "{}_disp.npy".format(output_name))
                    scaled_disp, _ = disp_to_depth(disp, 0.1, 100)
                    np.save(name_dest_npy, scaled_disp.cpu().numpy())

                    # Saving colormapped depth image
                    disp_resized_np = disp_resized.squeeze().cpu().numpy()
                    vmax = np.percentile(disp_resized_np, 95)
                    normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
                    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
                    colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
                    im = pil.fromarray(colormapped_im)

                    name_dest_im = os.path.join(self.output_directory, "{}_disp.jpeg".format(output_name))
                    im.save(name_dest_im)
                except :
                    print("File is not found.")



if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = Ui_MainWindow()
    win.show()
    sys.exit(app.exec_())