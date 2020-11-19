
import torch
import networks

from torchstat import stat
import time
from .test_net import TestNet

class Monodepth2(TestNet):

    def __init__(self, *args, **kwargs):
        super(Monodepth2, self).__init__(*args, **kwargs)
        self.load_dict = False
        self.onnx_outpart = ['encoder','decoder']

        if self.running_on == 'pc':
            #mono_1024x320
            #mono_640x192
            self.encoder_path = "/home/roit/models/monodepth2_official/mono_1024x320/encoder.pth"
            self.depth_decoder_path = "/home/roit/models/monodepth2_official/mono_1024x320/depth.pth"
        elif self.running_on == 'Xavier':
            self.encoder_path = "/home/wang/970evop1/models/mono_640x192/encoder.pth"
            self.depth_decoder_path = "/home/wang/970evop1/models/mono_640x192/depth.pth"


        # encoder init
        self.encoder = networks.ResnetEncoder(18, False)

        if self.load_dict:
            self.loaded_dict_enc = torch.load(self.encoder_path, map_location=self.device)
            self.filtered_dict_enc = {k: v for k, v in self.loaded_dict_enc.items() if k in self.encoder.state_dict()}
            self.encoder.load_state_dict(self.filtered_dict_enc)
        self.encoder.to(self.device)
        self.encoder.eval()

        # decoder
        if self.name=='arch2':
            self.decoder = networks.DepthDecoder2([64, 64, 128, 256, 512])
        elif self.name=='arch1':
            self.decoder = networks.DepthDecoder([64, 64, 128, 256, 512])
        elif self.name=='arch3':
            self.decoder = networks.DepthDecoder3([64, 64, 128, 256, 512])



        if self.load_dict:
            self.loaded_dict_dec = torch.load(self.depth_decoder_path, map_location=self.device)
            self.filtered_dict_dec = {k: v for k, v in self.loaded_dict_dec.items() if k in self.decoder.state_dict()}
            self.decoder.load_state_dict(self.filtered_dict_dec)
        self.decoder.to(self.device)
        self.decoder.eval()

        ## inputs size
        # self.feed_height = self.loaded_dict_enc['height']
        # self.feed_width = self.loaded_dict_enc['width']


        print('==> model name:{}\nfeed_height:{}\nfeed_width:{}\n'.format(self.name,self.feed_height,self.feed_width))

        #self.network = self.depth_decoder(self.encoder)
    def infer(self,input):
        torch.cuda.synchronize(self.device)

        t1 = time.time()
        features = self.encoder(input)
        #disp = self.depth_decoder(features)
        torch.cuda.synchronize(self.device)

        t2 = time.time()

        if self.name=='arch2':
            disp = self.decoder(features[0], features[1], features[2], features[3], features[4])
        elif self.name=='arch1':
            disp = self.decoder(features)
        elif self.name=='arch3':
            disp = self.decoder(features)


        torch.cuda.synchronize(self.device)

        t3 = time.time()
        self.duration_en = t2-t1
        self.duration_de = t3-t2
        self.duration = t3-t1
        return disp
    def torch_stat(self):
        stat(self.encoder.to('cpu'), input_size=(3, self.feed_width, self.feed_height))
        #stat(self.decoder, input_size=(3, self.feed_width, self.feed_height))



    def onnx_out(self):

        example_inputs = torch.rand(1, 3, self.feed_width,self.feed_height).cuda()
        if 'encoder' in self.onnx_outpart:
            encoder_out = torch.onnx.export(model=self.encoder,
                                          args=example_inputs,
                                          f=self.onnx_dir/"monodepth2_encoder18.onnx",
                                            output_names=['f0','f1','f2','f3','f4'],
                                          verbose=True,
                                          export_params=True  # 带参数输出
                                          )
        if 'decoder' in self.onnx_outpart:

            features = self.encoder(example_inputs)
            #arch1 -->list
            #arch2 --> 5 val
            #arch3 same with last one



            if self.name=='arch2':
                args = (features[0],features[1],features[2],features[3],features[4])
                print('arch2')
            elif self.name =='arch1':
                args = features# list get failed
            else:# arch3
                args = tuple(features)
                print('arch3')


            decoder_out = torch.onnx.export(model=self.decoder,
                                            args=args,
                                            input_names=['f0','f1','f2','f3','f4'],
                                            output_names=['disp'],
                                            f = self.onnx_dir/"monodepth2_decoder.onnx",
                                            verbose=True,
                                            export_params=True  # 带参数输出
                                          )

