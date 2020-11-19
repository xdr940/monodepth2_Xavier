import onnx
import numpy as np
import tvm
from tvm import te
import matplotlib.pyplot as plt
import cv2
import tvm.relay as relay
from tvm.contrib.download import download_testdata
import torch
# now you have super_resolution.onnx on disk


class TVMTest():
    def __init__(self):

        onnx_model_encoder = onnx.load('./onnxs/monodepth2_encoder18.onnx')
        onnx_model_decoder = onnx.load('./onnxs/monodepth2_decoder.onnx')

        self.target = "llvm"

        input_name = "input.1"





        shape_dict = {input_name: (1,3,384,224)}

        shape_dict2 = {"f4": (1, 512, 12, 7),
                       "f3": (1, 256, 24, 14),
                       "f2": (1, 128, 48, 28),
                       "f1": (1, 64, 96, 56),
                       "f0": (1, 64, 192, 112)
                       }
        self.encoder, self.en_params = relay.frontend.from_onnx(onnx_model_encoder, shape_dict)
        self.decoder, self.de_params = relay.frontend.from_onnx(onnx_model_decoder, shape_dict2)



    def save(self):
        graph, lib, params = relay.build_module.build(self.encoder, self.target, params=self.en_params)

        libpath = "gemfield.so"
        lib.export_library(libpath)

        graph_json_path = "gemfield.json"
        with open(graph_json_path, 'w') as fo:
            fo.write(graph)

        param_path = "gemfield.params"
        with open(param_path, 'wb') as fo:
            fo.write(relay.save_param_dict(params))


    def __call__(self,):
        img = cv2.imread('./img.jpg')
        img = cv2.resize(img, (224, 384))
        img = img.transpose([2, 0, 1])
        x = np.expand_dims(img, axis=0)
        dtype = "float32"
        input = tvm.nd.array(x.astype(dtype))

        f4 = np.random.random([1,512,12,7])
        f4 = tvm.nd.array(f4.astype(dtype))

        f3 = np.random.random([1,256,24,14])
        f3 = tvm.nd.array(f3.astype(dtype))

        f2 = np.random.random([1,128,48,28])
        f2 = tvm.nd.array(f2.astype(dtype))

        f1 = np.random.random([1,64,96,56])
        f1 = tvm.nd.array(f1.astype(dtype))

        f0 = np.random.random([1,64,192,112])
        f0 = tvm.nd.array(f0.astype(dtype))


        features = tuple([f0,f1,f2,f3,f4])


        with tvm.transform.PassContext(opt_level=1):
            intrp_en = relay.build_module.create_executor("graph", self.encoder, tvm.cpu(0), self.target)
            infer_en = intrp_en.evaluate()
            features = infer_en(input, **self.en_params)
            features= tuple(features)



            #
            intrp_de = relay.build_module.create_executor("debug", self.decoder, tvm.cpu(0), self.target)
            infer_de = intrp_de.evaluate()
            disp = infer_de(features)
            print('ok')




            #depth = infer_de(features, **self.de_params)








if __name__ == '__main__':
    func = TVMTest()
    func()
