
import torch
import networks

from torchstat import stat
import time
from .test_net import TestNet

class MonoPose(TestNet):
    def __init__(self, *args, **kwargs):
        super(MonoPose, self).__init__(*args, **kwargs)
        if self.running_on == 'pc':
            self.encoder_path = "/home/roit/models/monodepth2_official/mono_640x192/pose_encoder.pth"
            self.pose_decoder_path = "/home/roit/models/monodepth2_official/mono_640x192/pose.pth"
        elif self.running_on == 'Xavier':
            self.encoder_path = "/home/wang/970evop1/models/mono_640x192/pose_encoder.pth"
            self.pose_decoder_path = "/home/wang/970evop1/models/mono_640x192/pose.pth"

        self.encoder = networks.resnet_encoder.ResnetEncoder(num_input_images=2,num_layers=18,pretrained=False)
        self.encoder.to(self.device)
        self.encoder.eval()


        self.decoder = networks.pose_decoder.PoseDecoder(num_ch_enc=[64, 64, 128, 256, 512],num_frames_to_predict_for=2,num_input_features=1)
        self.decoder.to(self.device)
        self.decoder.eval()


        self.onnx_outpart = [
            # 'encoder',
            'decoder'
        ]


    def onnx_out(self):
        example_inputs = torch.rand(8, 6, 640, 192).cuda()

        features = self.encoder(example_inputs)
        # features = tuple(features)
        out = self.decoder(features)

        encoder_out = torch.onnx.export(model=self.encoder,
                                        args=example_inputs,
                                        f=self.onnx_dir / "monodepth2_poseencoder18.onnx",
                                        #output_names=['f0', 'f1', 'f2', 'f3', 'f4'],
                                        verbose=True,
                                        export_params=True  # 带参数输出
                                        )

    def torch_stat(self):
        stat(self.encoder.to('cpu'), input_size=(6, self.feed_width, self.feed_height))
