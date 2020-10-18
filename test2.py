from __future__ import absolute_import, division, print_function

import os
import sys
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
import torch
from torchvision import transforms

import networks
from networks.layers import disp_to_depth
import cv2

from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QLabel


class Ui_MainWindow(QMainWindow):
    def __init__(self):
        super(Ui_MainWindow, self).__init__()
        self.initUi()

    def initUi(self):
        self.num=0
        self.number = 0
        self.timer_camera = QTimer()  # 定义定时器，用于控制显示视频的帧率
        self.timer_camera2 = QTimer()
        self.resize(640, 480)
        self.setWindowTitle("test_simple")
        self.centralwidget = QWidget()
        self.label_show_camera = QLabel(self.centralwidget)  # 定义显示视频的Label
        self.label_show_camera.setFixedSize(640, 480)
        self.label_show_camera.setGeometry(0, 0, 640, 480)
        self.setCentralWidget(self.centralwidget)
        self.timer_camera.timeout.connect(self.show_camera)
        self.timer_camera2.timeout.connect(self.show_rate)

        self.picture_path = "picture/test.jpg"
        self.encoder_path = "models/mono+stereo_640x192/encoder.pth"
        self.depth_decoder_path = "models/mono+stereo_640x192/depth.pth"
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.encoder = networks.ResnetEncoder(18, False)
        self.loaded_dict_enc = torch.load(self.encoder_path, map_location=self.device)

        self.feed_height = self.loaded_dict_enc['height']
        self.feed_width = self.loaded_dict_enc['width']
        self.filtered_dict_enc = {k: v for k, v in self.loaded_dict_enc.items() if k in self.encoder.state_dict()}
        self.encoder.load_state_dict(self.filtered_dict_enc)
        self.encoder.to(self.device)
        self.encoder.eval()
        self.depth_decoder = networks.DepthDecoder(
            num_ch_enc=self.encoder.num_ch_enc, scales=range(4))

        self.loaded_dict = torch.load(self.depth_decoder_path, map_location=self.device)
        self.depth_decoder.load_state_dict(self.loaded_dict)

        self.depth_decoder.to(self.device)
        self.depth_decoder.eval()

        self.paths = [self.picture_path]
        self.output_directory = os.path.dirname(self.picture_path)

        self.timer_camera.start(1)
        self.timer_camera2.start(1000)


    def show_camera(self):
        self.number = self.number + 1
        picture_name = "picture/" + str(self.number) + ".jpg"
        self.picture_path = picture_name
        self.paths = [self.picture_path]
        self.output_directory = os.path.dirname(self.picture_path)
        print(picture_name)
        self.test_simple()
        self.image = cv2.imread("picture/" + str(self.number) + '_disp.jpeg')
        show = cv2.resize(self.image, (640, 480))  # 把读到的帧的大小重新设置为 640x480
        show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)  # 视频色彩转换回RGB，这样才是现实的颜色
        show_image = QImage(show.data, show.shape[1], show.shape[0], QImage.Format_RGB888)  # 把读取到的视频数据变成QImage形式
        self.label_show_camera.setPixmap(QPixmap.fromImage(show_image))  # 往显示视频的Label里 显示QImage
        self.num = self.num + 1

    def show_rate(self):
        print("视频帧率："+str(self.num))
        self.num = 0

    def test_simple(self):
        # if torch.cuda.is_available():
        #     device = torch.device("cuda")
        # else:
        #     device = torch.device("cpu")

        # picture_path = "picture/test.jpg"
        # encoder_path = "models/mono+stereo_640x192/encoder.pth"
        # depth_decoder_path = "models/mono+stereo_640x192/depth.pth"

        # encoder = networks.ResnetEncoder(18, False)
        # loaded_dict_enc = torch.load(self.encoder_path, map_location=self.device)

        # extract the height and width of image that this model was trained with
        # feed_height = self.loaded_dict_enc['height']
        # feed_width = self.loaded_dict_enc['width']
        # filtered_dict_enc = {k: v for k, v in self.loaded_dict_enc.items() if k in self.encoder.state_dict()}
        # self.encoder.load_state_dict(filtered_dict_enc)
        # self.encoder.to(self.device)
        # self.encoder.eval()

        # depth_decoder = networks.DepthDecoder(
        #     num_ch_enc=self.encoder.num_ch_enc, scales=range(4))

        # loaded_dict = torch.load(self.depth_decoder_path, map_location=self.device)
        # self.depth_decoder.load_state_dict(loaded_dict)

        # self.depth_decoder.to(self.device)
        # self.depth_decoder.eval()

        # FINDING INPUT IMAGES
        # if os.path.isfile(self.picture_path):
        #     # Only testing on a single image
        #     paths = [self.picture_path]
        #     output_directory = os.path.dirname(self.picture_path)
        # elif os.path.isdir(self.picture_path):
        #     # Searching folder for images
        #     paths = glob.glob(os.path.join(self.picture_path, '*.{}'.format("jpg")))
        #     output_directory = self.picture_path
        # else:
        #     raise Exception("Can not find args.image_path: {}".format(self.picture_path))

        # paths = [self.picture_path]
        # output_directory = os.path.dirname(self.picture_path)


        # PREDICTING ON EACH IMAGE IN TURN
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
                    outputs = self.depth_decoder(features)

                    disp = outputs[("disp", 0)]
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
                    self.number = self.number - 1
                    print("File is not found.")



if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = Ui_MainWindow()
    win.show()
    sys.exit(app.exec_())
