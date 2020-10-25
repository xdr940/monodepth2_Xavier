import argparse
class OPTIONS:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Monodepthv2 training options")
        self.parser.add_argument("--running_on",default='pc',choices=['pc','Xavier'])
        self.parser.add_argument("--arch", default='monodepth2')
        self.parser.add_argument("--device", default='cuda')

        self.parser.add_argument("--encoder_path",
                                 default="/home/roit/models/monodepth2/mc_06020659/encoder.pth"#pc mc
                                 #default = "/home/roit/models/monodepth2_official/mono_640x192/encoder.pth"# pc kitti
                                 )
        self.parser.add_argument("--depth_decoder_path",
                                 default="/home/roit/models/monodepth2/mc_06020659/depth.pth"# pc mc
                                 #default = "/home/roit/models/monodepth2_official/mono_640x192/depth.pth"  # pc kitti

                                 #default = "/home/roit/models/monodepth2/mc_06020659/depth.pth"
                                 #default = "/home/wang/970evop1/models/mono_640x192/encoder.pth"#Xavier mono

        )
        self.parser.add_argument("--camera_name",default="/dev/video0")


        self.parser.add_argument("--feed_width",default=384)
        self.parser.add_argument("--feed_height",default=224)

        self.parser.add_argument("--show_frame",default=True) #show input frames
        self.parser.add_argument("--capture_width",default=640)
        self.parser.add_argument("--capture_height",default=480)





    def args(self):
        self.options = self.parser.parse_args()
        return self.options


