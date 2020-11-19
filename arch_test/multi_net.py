
import torch
import networks

from torchstat import stat
import time
from .test_net import TestNet

class MultiNet(TestNet):
    def __init__(self, *args, **kwargs):
        super(MultiNet, self).__init__(*args, **kwargs)

        self.encoder = networks.ResnetEncoder(num_layers=18,pretrained=False)
        self.encoder.to(self.device)
        self.encoder.eval()

        self.decoder = networks.DepthDecoder3( [64, 64, 128, 256, 512])
        self.decoder.to(self.device)
        self.decoder.eval()
        self.onnx_outpart = [
                       # 'encoder',
                        'decoder'
        ]


    def onnx_out(self):
        example_inputs = torch.rand(8, 3, self.feed_width,self.feed_height).cuda()

        features = self.encoder(example_inputs)
        #features = tuple(features)
        out = self.decoder(*features)
        features_name = ["f{}".format(i) for i in range(len(features))],
        if 'encoder' in self.onnx_outpart:
            encoder_out = torch.onnx.export(model=self.encoder,
                                            args=example_inputs,
                                            f=self.onnx_dir /( self.name+".onnx"),
                                            output_names=features_name,
                                            verbose=True,
                                            export_params=True  # 带参数输出
                                            )

        if 'decoder' in self.onnx_outpart:
            decoder_out = torch.onnx.export(model=self.decoder,
                                            args=tuple(features),
                                            f=self.onnx_dir / "Mencoder.onnx",
                                            input_names=features_name,
                                            #output_names=['f0', 'f1', 'f2', 'f3', 'f4'],
                                            verbose=True,
                                            export_params=True  # 带参数输出
                                            )
    def torch_stat(self):
        stat(self.encoder.to('cpu'), input_size=(3, self.feed_width, self.feed_height))





if __name__ == '__main__':
    pass


