#from __future__ import absolute_import, division, print_function

from utils.logger import Writer
from torchvision import transforms
import threading
import networks

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
cudnn.benchmark = True
import cv2
import time
class Inference():
    def __init__(self,dev='cuda',running_on='pc',arch='monodepth2'):
        print('==> runing on ', running_on)
        self.device = torch.device(dev)
        print("==> device:", self.device)

        print('==> running on ',running_on)

        print('==> arch',arch)
        self.arch=arch
        #models path
        if arch=='monodepth2':
            self.monodepth2_init(running_on=running_on)
        elif arch =='fastdepth':
            if running_on == 'pc':
                self.model_path = "/home/roit/models/fast-depth/mobilenet-nnconv5.pth.tar"
            elif running_on=='Xavier':
                self.model_path = "/home/wang/models/fast-depth/mobilenet-nnconv5.pth.tar"


        ##init

        if arch=='monodepth2':
            self.monodepth2_init(running_on=running_on)
        elif arch=='fastdepth':
            self.fastdepth_init(running_on=running_on)



        ##camera
        try:
            self.cap = cv2.VideoCapture()
            if running_on=='pc':
                self.cap.open("/dev/video0")
            elif running_on=='Xavier':
                self.cap.open("/dev/mycamera")
        except:
            print("==> camera open failed")
            return


        self.writer = Writer()
        self.duration={'cap':1,'transform':2,'encoder':3,'decoder':4,'final':5}
        _,self.frame = self.cap.read()
    def capture(self):
        while True:
            t1 = time.time()
            _, self.frame = self.cap.read()
            t2 = time.time()
            self.duration['cap'] = t2 - t1
            #time.sleep(0.01)
    def predict(self):
        with torch.no_grad():
            while(True):
                try:


                    #capture
                    t1 = time.time()
                    _, self.frame = self.cap.read()
                    t2 = time.time()
                    self.duration['cap'] = t2 - t1
                    cv2.imshow('frame',self.frame)


                    #transform
                    t2 = time.time()
                    input_image = cv2.resize(self.frame, (self.feed_width, self.feed_height))
                    input_image = transforms.ToTensor()(input_image).unsqueeze(0)
                    input_image = input_image.to(self.device)
                    t3 = time.time()
                    self.duration['transform']= t3-t2

                    # # PREDICTION
                    torch.cuda.synchronize(self.device)

                    features = self.encoder(input_image)
                    t33 = time.time()

                    disp = self.depth_decoder(features[0], features[1], features[2], features[3], features[4])

                    disp = torch.nn.functional.interpolate(
                        disp, (480, 640), mode="bilinear", align_corners=False)[0, 0].to('cpu').detach().numpy()

                    torch.cuda.synchronize(self.device)
                    t4 = time.time()

                    self.duration['encoder'] = t33 - t3
                    self.duration['decoder'] = t4 - t33
                    cv2.imshow('depth',disp)

                    self.duration['final']=t4-t1

                    if cv2.waitKey(1) & 0xff == ord('q'):
                        break
                except KeyboardInterrupt:

                    return
                #except:
                #    pass
    def run(self):
        #t0 = threading.Thread(target=self.capture)

        t1 = threading.Thread(target=self.predict)

        t2 = threading.Thread(target=self.get_fps)
        t1.start()
        t2.start()
        #t0.start()
        #t3.start()
    def get_fps(self,line=7):
        while True:
            cnt=line
            for k,v in self.duration.items():

                self.writer.write("{}: {:.2f}ms fps={:.2f} ".format(k, 1000*v, 1/v),location=(0,cnt))
                cnt+=1

            time.sleep(2.1)

    def monodepth2_init(self,running_on):

        if running_on == 'pc':
            self.encoder_path = "/home/roit/bluep2/models/monodepth2/MC/ours_06020659/models/weights_19/encoder.pth"
            self.depth_decoder_path = "/home/roit/bluep2/models/monodepth2/MC/ours_06020659/models/weights_19/depth.pth"
        elif running_on == 'Xavier':
            self.encoder_path = "/home/wang/970evop1/models/mono_640x192/encoder.pth"
            self.depth_decoder_path = "/home/wang/970evop1/models/mono_640x192/depth.pth"


        # encoder init
        self.encoder = networks.ResnetEncoder(18, False)
        self.loaded_dict_enc = torch.load(self.encoder_path, map_location=self.device)

        self.feed_height = self.loaded_dict_enc['height']
        self.feed_width = self.loaded_dict_enc['width']
        self.filtered_dict_enc = {k: v for k, v in self.loaded_dict_enc.items() if k in self.encoder.state_dict()}
        self.encoder.load_state_dict(self.filtered_dict_enc)
        self.encoder.to(self.device)
        self.encoder.eval()

        # decoder

        self.depth_decoder = networks.DepthDecoder2([64, 64, 128, 256, 512])
        self.loaded_dict_dec = torch.load(self.depth_decoder_path, map_location=self.device)
        self.filtered_dict_dec = {k: v for k, v in self.loaded_dict_dec.items() if k in self.depth_decoder.state_dict()}

        self.depth_decoder.load_state_dict(self.filtered_dict_dec)
        self.depth_decoder.to(self.device)
        self.depth_decoder.eval()

        ## inputs size
        self.feed_height = self.loaded_dict_enc['height']
        self.feed_width = self.loaded_dict_enc['width']
    def monodepth2(self,input_image):
        features = self.encoder(input_image)
        disp = self.depth_decoder(features[0], features[1], features[2], features[3], features[4])
        disp = torch.nn.functional.interpolate(
            disp, (480, 640), mode="bilinear", align_corners=False)[0, 0].to('cpu').detach().numpy()
        return disp

    def fastdepth_init(self,running_on):
        if running_on=='pc':
            model = torch.load('/home/roit/models/fast-depth/mobilenet-nnconv5.pth.tar')
        elif running_on=='Xavier':
            model = torch.load('/home/wang/models/fast-depth/mobilenet-nnconv5.pth.tar')
        else:
            print('==>error')
        self.model = model['model']
        self.feed_height = 224
        self.feed_width = 384
        pass

    def fastdepth(self,input_image):
        disp = self.model(input_image)
        disp = torch.nn.functional.interpolate(
            disp, (480, 640), mode="bilinear", align_corners=False)[0, 0].to('cpu').detach().numpy()
        return  1/disp


if __name__ == "__main__":
    inf = Inference('cuda',running_on='pc',arch='monodepth2')
    inf.run()
  #  inf2 =Inference('cpu')
  #  inf2.run()

