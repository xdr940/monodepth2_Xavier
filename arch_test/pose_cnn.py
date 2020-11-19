
import torch
import networks

from torchstat import stat
import time
from .test_net import TestNet

class PoseCNN(TestNet):
    def __init__(self, *args, **kwargs):
        super(PoseCNN, self).__init__(*args, **kwargs)
        if self.running_on == 'pc':
            self.encoder_path = "/home/roit/models/SCBian_official/cs+k_pose.tar"
        elif self.running_on == 'Xavier':
            pass
            #self.encoder_path = "/home/wang/970evop1/models/mono_640x192/pose_encoder.pth"

        self.encoder = networks.pose_cnn.PoseCNN(num_input_frames=2)

        self.encoder.to(self.device)

        self.encoder.eval()

    def onnx_out(self):
        example_inputs = torch.rand(8, 6, self.feed_width,self.feed_height).cuda()

        encoder_out = torch.onnx.export(model=self.encoder,
                                        args=example_inputs,
                                        f=self.onnx_dir /( self.name+".onnx"),
                                        # output_names=['f0', 'f1', 'f2', 'f3', 'f4'],
                                        verbose=True,
                                        export_params=True  # 带参数输出
                                        )

    def torch_stat(self):
        stat(self.encoder.to('cpu'), input_size=(6, self.feed_width, self.feed_height))
