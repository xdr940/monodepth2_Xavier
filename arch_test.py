
from utils.logger import Writer
import threading
import torch
import networks
from torchstat import stat
from path import Path
import time
lock = threading.Lock()



class TestNet():
    def __init__(self,running_on,dev,name,height,width):
        super(TestNet, self).__init__()
        self.running_on = running_on
        self.device = torch.device(dev)
        self.feed_width = width
        self.feed_height = height


        self.duration = 1
        self.writer = Writer()
        self.duration = 1
        self.out_width = 640
        self.out_height = 480

        self.network=None
        self.name=name
        # self.network = networks.ResNet(layers=18, decoder='nnconv5', output_size=[self.feed_width, self.feed_height]).to(
        #     self.device)

        self.duration_en = 1
        self.duration_de = 1
        self.duration = 1
    def infer(self,input):

        torch.cuda.synchronize(self.device)

        t1 = time.time()
        features = self.encoder(input)
        # disp = self.depth_decoder(features)
        torch.cuda.synchronize(self.device)

        t2 = time.time()

        disp = self.decoder(features)
        torch.cuda.synchronize(self.device)

        t3 = time.time()
        self.duration_en = t2 - t1
        self.duration_de = t3 - t2
        self.duration = t3 - t1
        return disp

    def Test(self):
        th2 = threading.Thread(target=self.fps)
        th2.start()

        example_inputs = torch.rand(1, 3, self.feed_height, self.feed_width).to(self.device)
        while True:
            try:


                ##infer
                disp = self.infer(example_inputs)


            except KeyboardInterrupt:
                return

    def fps(self):
        while True:
            time.sleep(1.1)
            self.writer.write("encoder: {:.2f} ms\ndecoder: {:.2f}ms\ntotal: {:.2f}ms\n".format(self.duration_en*1000,self.duration_de*1000,self.duration*1000), location=(0, 5))

    def onnx_out(self):
        example_inputs = torch.rand(8, 3, 224, 224)  # bs is not matter for latency

        output = torch.onnx.export(model=self.network,
                                   args=example_inputs,
                                   f=onnx_dir / self.name+".onnx",
                                   # output_names=['f0','f1','f2','f3','f4'],
                                   verbose=True,
                                   export_params=True  # 带参数输出
                                   )

    def torch_stat(self):
        stat(self.encoder.to('cpu'), input_size=(3, self.feed_width, self.feed_height))



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

class Monodepth2(TestNet):

    def __init__(self, *args, **kwargs):
        super(Monodepth2, self).__init__(*args, **kwargs)
        self.load_dict = False


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
        else:
            self.decoder = networks.DepthDecoder([64, 64, 128, 256, 512])

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
        else:
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

        pass


    def onnx_out(self):
        example_inputs = torch.rand(8, 3, 640, 192).cuda()

        encoder_out = torch.onnx.export(model=self.encoder,
                                      args=example_inputs,
                                      f=onnx_dir/"monodepth2_encoder18.onnx",
                                        output_names=['f0','f1','f2','f3','f4'],
                                      verbose=True,
                                      export_params=True  # 带参数输出
                                      )

        features = self.encoder(example_inputs)

        depth_decoder_out = self.depth_decoder(features[0],features[1],features[2],features[3],features[4])

        decoder_out = torch.onnx.export(model=self.depth_decoder,
                                      args=(features[0],features[1],features[2],features[3],features[4]),
                                        #args=features,

                                        input_names=['f0','f1','f2','f3','f4'],
                                        output_names=['o0'],
                                      f = onnx_dir/"monodepth2_decoder.onnx",
                                      verbose=True,
                                      export_params=True  # 带参数输出
                                      )


class FastMobileNet(TestNet):
    def __init__(self, *args, **kwargs):
        super(FastMobileNet, self).__init__(*args, **kwargs)

        self.encoder = networks.MobileNet().to(self.device)
        self.decoder = networks.choose_decoder(decoder=self.name).to(self.device)
        self.encoder.eval()
        self.decoder.eval()
    def infer(self,input):

        torch.cuda.synchronize(self.device)

        t1 = time.time()
        features = self.encoder(input)
        # disp = self.depth_decoder(features)
        torch.cuda.synchronize(self.device)

        t2 = time.time()

        disp = self.decoder(features)
        torch.cuda.synchronize(self.device)

        t3 = time.time()
        self.duration_en = t2 - t1
        self.duration_de = t3 - t2
        self.duration = t3 - t1
        return disp
    def onnx_out(self):
        example_inputs = torch.rand(8, 3, 224, 224)

        encoder_out = torch.onnx.export(model=self.network,
                                        args=example_inputs,
                                        f=onnx_dir / "MobileNet-nnconv5dw.onnx",
                                        # output_names=['f0','f1','f2','f3','f4'],
                                        verbose=True,
                                        export_params=True  # 带参数输出
                                        )



if  __name__ == '__main__':
    onnx_dir = Path("onnxs")
    onnx_dir.mkdir_p()
    #network = FastResNet(running_on='pc', dev='cuda',name='nnconv5dw',height=224,width=224)

    #network = FastMobileNet(running_on='pc', dev='cuda',name='nnconv5dw',height=224,width=224)
    network = Monodepth2(running_on='pc',dev='cuda',name='arch1',height=224,width=384)
    network.Test()

    #network.torch_stat()
    #network = Monodepth2()
    #network.onnx_out()
    #network.torch_stat()
    print('ok')

    pass


