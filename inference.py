#from __future__ import absolute_import, division, print_function
from opts import OPTIONS

from utils.logger import Writer
from torchvision import transforms
import threading
import networks
import numpy as np
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
cudnn.benchmark = True

from path import Path
import cv2
import time

from utils.formater import pose6dof2kitti,np2line,kitti2pose6dof
from utils.inverse_warp import pose_vec2mat

class Inference():
    def __init__(self,args):
        self.running_on = args.running_on
        self.arch = args.arch
        self.colormap=cv2.COLORMAP_BONE

        #print('==> runing on ', self.running_on)
        print('==> runing on ', 'Xavier')
        self.device = torch.device(args.device)
        print("==> device:", self.device)
        print('==> arch',self.arch)
        #models path

        if self.arch=='monodepth2':
            self.monodepth2_init(args)
        elif self.arch =='fastdepth':
            if self.running_on == 'pc':
                self.model_path = "/home/roit/models/fast-depth/mobilenet-nnconv5.pth.tar"
            elif self.running_on=='Xavier':
                self.model_path = "/home/wang/models/fast-depth/mobilenet-nnconv5.pth.tar"


        ##init

        if self.arch=='monodepth2':
            self.monodepth2_init(args)
        elif self.arch=='fastdepth':
            pass
            #self.fastdepth_init(running_on=self.running_on)

        #poses
        if args.pose_arch!=None:
            self.monopose_init(args)
            self.fin = "./poses_6dof.txt"
            self.duration = {'cap': 1, 'pose': 2, 'encoder': 3, 'decoder': 4, 'final': 5}
        else:
            self.duration = {'cap': 1, 'transform': 2, 'encoder': 3, 'decoder': 4, 'final': 5}

        self.capture_width = args.capture_width
        self.capture_height = args.capture_height

        self.writer = Writer()




        ##camera
        try:
            self.cap = cv2.VideoCapture()
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.capture_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.capture_height)
            self.cap.open(args.camera_name)
            print('==> cap size{}x{}, camera name{}'.format(self.capture_width,self.capture_height,args.camera_name))
        except:
            print("==> camera open failed")
            return

        _, self.frame = self.cap.read()

        self.frames = torch.ones([1, 6, self.feed_height, self.feed_width]).to(self.device)

    def prediction(self):
        cnt=0
        poses_6dof=[]
        MAX_LENTH=200
        global_pose = np.eye(4)
        with torch.no_grad():
            while(True):
                try:


                    #capture
                    t1 = time.time()
                    _, self.frame = self.cap.read()
                    t2 = time.time()

                    self.duration['cap'] = t2 - t1

                    cv2.imshow('frame',self.frame)


                    #transform
                    if args.cap_width!=args.feed_width or args.cap_height!=args.cap_width:
                        input_image = cv2.resize(self.frame, (self.feed_width, self.feed_height))
                    else:
                        input_image = self.frame
                    input_image = transforms.ToTensor()(input_image).unsqueeze(0)
                    input_image = input_image.to(self.device)
                    # predict pose

                    if args.pose_arch != None and cnt % 5 == 0:
                        t2 = time.time()

                        self.frames = torch.cat([self.frames[:,3:,:,:],input_image],dim=1)
                        # pose_6dof = self.posenet(self.frames)
                        #
                        # pose_mat = pose_vec2mat(pose_6dof).squeeze(0).cpu().numpy()  # 1,6-->3x4
                        #
                        # pose_mat = np.vstack([pose_mat, np.array([0, 0, 0, 1])])  # 4X4
                        # global_pose = pose_mat @ global_pose  # 这地方类似与disp， 为了限制在0~1， 训练的时候也是用的T-1， 得到的是invpose
                        self.monopose_pred(self.frames)
                        #poses_write(self.fin,pose_6dof)
                        # poses_6dof.append(poses_6dof)
                        # if len(poses_6dof)>MAX_LENTH:
                        #     poses_6dof.pop(0)


                        t3 = time.time()

                        self.duration['pose']= t3-t2
                    else:
                        t3 = time.time()
                        self.duration['transform']= t3-t2

                    # # PREDICTION
                    torch.cuda.synchronize(self.device)

                    features = self.encoder(input_image)
                    t33 = time.time()

                    disp = self.depth_decoder(features[0], features[1], features[2], features[3], features[4])

                    disp = torch.nn.functional.interpolate(
                        disp, (480, 640), mode="bilinear", align_corners=False)[0, 0].to('cpu').detach().numpy()

                    torch.cuda.synchronize(self.device)
                    t4 = time.time()

                    self.duration['encoder'] = t33 - t3
                    self.duration['decoder'] = t4 - t33

                    disp=(disp*255).astype(np.uint8)
                    disp = cv2.applyColorMap(disp,colormap=self.colormap)
                    cv2.imshow('depth',disp)


                    self.duration['final']=t4-t1



                    cnt+=1
                    if cv2.waitKey(1) & 0xff == ord('q'):
                        break
                except KeyboardInterrupt:
                    self.fpose.close()
                    self.file_pip.remove()

                    return

    def run(self):
        #t0 = threading.Thread(target=self.capture)

        t1 = threading.Thread(target=self.prediction)

        t2 = threading.Thread(target=self.get_fps)

        #t3 = threading.Thread(target=draw,args=[self.fin])
        t1.start()
        t2.start()
        #t0.start()
        #t3.start()
    def get_fps(self,line=7):
        while True:
            cnt=line
            for k,v in self.duration.items():

                self.writer.write("{}: {:.2f}ms fps={:.2f} ".format(k, 1000*v, 1/v),location=(0,cnt))
                cnt+=1

            time.sleep(2.1)

    def monodepth2_init(self,args):
        self.encoder_path = args.encoder_path
        self.depth_decoder_path = args.depth_decoder_path


        # encoder init
        self.encoder = networks.ResnetEncoder(18, False)
        self.loaded_dict_enc = torch.load(self.encoder_path, map_location=self.device)

        self.feed_height = self.loaded_dict_enc['height']
        self.feed_width = self.loaded_dict_enc['width']
        filtered_dict_enc = {k: v for k, v in self.loaded_dict_enc.items() if k in self.encoder.state_dict()}
        self.encoder.load_state_dict(filtered_dict_enc)
        self.encoder.to(self.device)
        self.encoder.eval()

        # decoder

        self.depth_decoder = networks.DepthDecoder2([64, 64, 128, 256, 512])
        self.loaded_dict_dec = torch.load(self.depth_decoder_path, map_location=self.device)
        filtered_dict_dec = {k: v for k, v in self.loaded_dict_dec.items() if k in self.depth_decoder.state_dict()}

        self.depth_decoder.load_state_dict(filtered_dict_dec)
        self.depth_decoder.to(self.device)
        self.depth_decoder.eval()

        ## inputs size

        if args.feed_height and args.feed_width:
            self.feed_height =  args.feed_height
            self.feed_width = args.feed_width
        else:
            self.feed_height = self.loaded_dict_enc['height']
            self.feed_width = self.loaded_dict_enc['width']
    def monodepth2(self,input_image):
        features = self.encoder(input_image)
        disp = self.depth_decoder(features[0], features[1], features[2], features[3], features[4])
        disp = torch.nn.functional.interpolate(
            disp, (self.capture_height, self.capture_width), mode="bilinear", align_corners=False)[0, 0].to('cpu').detach().numpy()
        return disp


    def monopose_init(self,args):

        if args.pose_arch=='posecnn':
            self.posenet = networks.posenet.PoseNet()
            pose_dict = torch.load(args.posecnn_path,map_location=self.device)['state_dict']
            self.posenet.load_state_dict(pose_dict)
            self.posenet.eval()
            self.posenet.to(self.device)
        elif args.pose_arch=='EnDePose':
            self.pose_decoder = networks.pose_decoder.PoseDecoder(num_ch_enc=3,num_input_features=1,num_frames_to_predict_for=2)
            self.pose_decoder_path = args.pose_decoder_path
            pose_dict = torch.load(args.pose_decoder_path)
            self.pose_decoder.load_state_dict(pose_dict).to(self.device)
            self.pose_decoder.eval()
        elif args.pose_arch==None:
            pass


        self.file_pip = Path("/home/roit/Desktop/fpose.txt")
        self.fpose = open(self.file_pip,'a')
        self.global_pose= np.eye(4)
        self.pose_cnt=0


    def monopose_pred(self,input_frames):

        pose_6dof = self.posenet(input_frames)

        pose_mat = pose_vec2mat(pose_6dof).squeeze(0).cpu().numpy()  # 1,6-->3x4

        pose_mat = np.vstack([pose_mat, np.array([0, 0, 0, 1])])  # 4X4
        self.global_pose = pose_mat @ self.global_pose  # 这地方类似与disp， 为了限制在0~1， 训练的时候也是用的T-1， 得到的是invpose
        self.pose_cnt+=1

        if self.pose_cnt%5==0:
            global_pose = self.global_pose[:3,:].reshape([1,12])
            pose6dof = kitti2pose6dof(global_pose)#1x6
            pose6dof = np.concatenate([np.ones([1,1])*self.pose_cnt,pose6dof],axis=1)


            pose6dof = np2line(pose6dof)
            self.fpose = open('/home/roit/Desktop/fpose.txt', 'a')
            self.fpose.write(pose6dof)
            self.fpose.close()

if __name__ == "__main__":
    args = OPTIONS().args()
    inf = Inference(args)
    inf.run()


