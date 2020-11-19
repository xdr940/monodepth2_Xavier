import torch
import networks

from torchstat import stat
import time
from .test_net import TestNet


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
                                        f=self.onnx_dir / "MobileNet-nnconv5dw.onnx",
                                        # output_names=['f0','f1','f2','f3','f4'],
                                        verbose=True,
                                        export_params=True  # 带参数输出
                                        )
