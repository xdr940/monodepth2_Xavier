from __future__ import absolute_import, division, print_function

from utils.logger import Writer
import torch
from torchvision import transforms
import threading
import networks
from networks.layers import disp_to_depth
from path import Path
import cv2
import time
import os
class Inference():
    def __init__(self):
        self.encoder_path = "/home/roit/models/monodepth2_official/mono_640x192/encoder.pth"
        self.depth_decoder_path = "/home/roit/models/monodepth2_official/mono_640x192/depth.pth"


        ##torch
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        print("==> device:", self.device)



        #encoder init
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

        ##
        self.feed_height = self.loaded_dict_enc['height']
        self.feed_width = self.loaded_dict_enc['width']
    ##
        self.cap = cv2.VideoCapture()
        self.cap.open(0)

        self.writer = Writer()
        self.duration=1
    def predict(self):
        with torch.no_grad():
            while(True):
                try:
                    before_op_time = time.time()

                    ret, frame = self.cap.read()
                    input_image = cv2.resize(frame, (self.feed_width, self.feed_height))

                    input_image = transforms.ToTensor()(input_image).unsqueeze(0)

                    # # PREDICTION
                    input_image = input_image.to(self.device)
                    features = self.encoder(input_image)
                    disp = self.depth_decoder(features[0], features[1], features[2], features[3], features[4])

                    # disp = outputs[("disp", 0)]
                    disp_resized = torch.nn.functional.interpolate(
                        disp, (480, 640), mode="bilinear", align_corners=False).cpu().numpy()[0,0]

                    cv2.imshow('depth',disp_resized)
                    if cv2.waitKey(12) & 0xff == ord('q'):
                        break
                except KeyboardInterrupt:
                    return
                self.duration = time.time() - before_op_time
    def run(self):
        t1 = threading.Thread(target=self.predict)
        t2 = threading.Thread(target=self.get_fps)
        t1.start()
        t2.start()
    def get_fps(self):
        while True:
            self.writer.write("fps {:.2f}".format(1 /self.duration))
            time.sleep(1.1)



if __name__ == "__main__":
    inf = Inference()
    inf.run()
