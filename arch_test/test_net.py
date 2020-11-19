

from utils.logger import Writer
import threading
import torch
from torchstat import stat
import time
from path import Path
class TestNet():
    def __init__(self,running_on,dev,name,height,width):
        super(TestNet, self).__init__()
        self.running_on = running_on
        self.device = torch.device(dev)
        self.feed_width = width
        self.feed_height = height

        self.onnx_dir = Path("./onnxs")
        self.onnx_dir.mkdir_p()

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
        example_inputs = torch.rand(1, 3, 224, 224)  # bs is not matter for latency

        output = torch.onnx.export(model=self.network,
                                   args=example_inputs,
                                   f=self.onnx_dir / self.name+".onnx",
                                   # output_names=['f0','f1','f2','f3','f4'],
                                   verbose=True,
                                   export_params=True  # 带参数输出
                                   )

    def torch_stat(self):
        stat(self.encoder.to('cpu'), input_size=(3, self.feed_width, self.feed_height))

