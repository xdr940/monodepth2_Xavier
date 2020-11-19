
import threading

from path import Path
lock = threading.Lock()

from arch_test import *



if  __name__ == '__main__':
    #onnx_dir = Path("onnxs")
    #onnx_dir.mkdir_p()

    # network = FastResNet(running_on='pc', dev='cuda',name='nnconv5dw',height=224,width=224)
    # network.onnx_out()
    #network = FastMobileNet(running_on='pc', dev='cuda',name='nnconv5dw',height=224,width=224)


    network = Monodepth2(running_on='pc',dev='cuda',name='arch3',height=224,width=384)
    network.onnx_out()


    # network = MonoPose(running_on='pc',dev='cuda',name='pose',height=224,width=384)
    # network.onnx_out()

    #
    # network = PoseCNN(running_on='pc',dev='cuda',name='posecnn',height=224,width=384)
    # network.torch_stat()
    # network.onnx_out()

    # network = MultiNet(running_on='pc',dev='cuda',height=224,width=384,name='mdecoder')
    # network.onnx_out()
    # network.torch_stat()


    pass


