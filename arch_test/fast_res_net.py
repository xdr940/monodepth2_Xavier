
import torch
import networks

from torchstat import stat
import time
from .test_net import TestNet

class FastResNet(TestNet):

    def __init__(self, *args, **kwargs):
        super(FastResNet, self).__init__(*args, **kwargs)


        self.encoder = networks.ResNet(layers=50,pretrained=False).to(self.device)
        self.decoder = networks.choose_decoder(decoder=self.name).to(self.device)
    # def infer(self,input):
    #     pass
        self.encoder.eval()
        self.decoder.eval()
        print('==> model name:{}\nfeed_height:{}\nfeed_width:{}\n'.format(self.name,self.feed_height,self.feed_width))
    def infer(self,input):
        torch.cuda.synchronize(self.device)

        t1 = time.time()
        features = self.encoder(input)
        #disp = self.depth_decoder(features)
        torch.cuda.synchronize(self.device)

        t2 = time.time()

        disp = self.decoder(features)
        torch.cuda.synchronize(self.device)

        t3 = time.time()
        self.duration_en = t2-t1
        self.duration_de = t3-t2
        self.duration = t3-t1
        return disp
    def torch_stat(self):
        stat(self.encoder.to('cpu'), input_size=(3, self.feed_width, self.feed_height))

