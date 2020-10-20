#from __future__ import absolute_import, division, print_function

from utils.logger import Writer
import torch
from torchvision import transforms
import threading
import networks
from networks.layers import disp_to_depth
import cv2
import time
import os
class Inference():
    def __init__(self,dev='cuda',runing_on='pc'):
        if runing_on=='pc':
            self.encoder_path = "/home/roit/models/monodepth2_official/mono_640x192/encoder.pth"
            self.depth_decoder_path = "/home/roit/models/monodepth2_official/mono_640x192/depth.pth"
        elif runing_on =='Xavier':
            self.encoder_path = "/home/wang/970evop1/models/mono_640x192/encoder.pth"
            self.depth_decoder_path = "/home/wang/970evop1/models/mono_640x192/depth.pth"

        print('==> runing on ',runing_on)
        self.device = torch.device(dev)
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
        if runing_on=='pc':
            self.cap.open("/dev/video0")
        elif runing_on=='Xavier':
            self.cap.open("/dev/mycamera")


        self.writer = Writer()
        self.duration={'cap':1,'transform':2,'prediction':3,'to_cpu':1,'show':4,'final':5}


    def predict(self):
        with torch.no_grad():
            while(True):
                try:


                    #capture

                    t1 = time.time()
                    ret, frame = self.cap.read()
                    t2 = time.time()
                    self.duration['cap'] = t2-t1

                    #transform

                    input_image = cv2.resize(frame, (self.feed_width, self.feed_height))
                    input_image = transforms.ToTensor()(input_image).unsqueeze(0)
                    input_image = input_image.to(self.device)
                    t3 = time.time()
                    self.duration['transform']= t3-t2

                    # # PREDICTION
                    torch.cuda.synchronize(self.device)

                    features = self.encoder(input_image)
                    disp = self.depth_decoder(features[0], features[1], features[2], features[3], features[4])



                    #show
                    disp = torch.nn.functional.interpolate(
                        disp, (480, 640), mode="bilinear", align_corners=False)
                    torch.cuda.synchronize(self.device)
                    t4 = time.time()

                    self.duration['prediction'] = t4 - t3
                    disp = disp[0, 0].to('cpu')

                    t5 =  time.time()

                    self.duration['to_cpu'] = t5-t4

                    disp= disp.detach().numpy()
                    cv2.imshow('depth',disp)
                    t6 = time.time()
                    self.duration['show'] = t6-t5
                    self.duration['final']=t6-t1

                    if cv2.waitKey(1) & 0xff == ord('q'):
                        break
                except KeyboardInterrupt:

                    return
                except:
                    pass
    def run(self):
        t1 = threading.Thread(target=self.predict)
        #t3 = threading.Thread(target=self.predict)

        t2 = threading.Thread(target=self.get_fps)
        t1.start()
        t2.start()
        #t3.start()
    def get_fps(self):
        while True:
            cnt=5
            for k,v in self.duration.items():

                self.writer.write("{}: {:.2f}".format(k, 1 /v),location=(0,cnt))
                cnt+=1
            cnt=5




            time.sleep(2.1)



if __name__ == "__main__":
    inf = Inference('cuda',runing_on='Xavier')
    inf.run()
  #  inf2 =Inference('cpu')
  #  inf2.run()

