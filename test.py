import os
import cv2
import argparse
import platform
import yolo.config as cfg
from yolo.yolo_net import YOLO
from detector import Detector
import warnings

def main():
    parser = argparse.ArgumentParser(description=' Test the detector ')
    parser.add_argument('--weights', default="save.ckpt-15000", metavar='', type=str, help= ' weights file' )
    parser.add_argument('--weight_dir', default='weights', type=str,metavar='', help= 'Folder contaning the weight file')
    parser.add_argument('--data_dir', default="data", type=str,metavar='', help = 'Folder contaning the weight folder')
    parser.add_argument('--gpu', default='', action='store_const', const='0', help = 'Use gpu for inference')  
    parser.add_argument('--camera', action='store_const', const=True, help = 'Test the detector on the live feed of the camera')
    parser.add_argument('--img_path',metavar='', help = 'Test the detector on a static image, give the path')
    parser.add_argument('--video_path',metavar='', help = 'Test the detector on a video, give the path')
    parser.add_argument('--test_img', action='store_const', const=True, help = 'Test the detector on sample images')
    parser.add_argument('--test_video', action='store_const', const=True, help = 'Test the detector on a sample video')
        
    args = parser.parse_args()
   
    warnings.filterwarnings("ignore")
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if(args.camera or args.test_img or args.test_video or args.img or args.video):
        yolo = YOLO(False)
        weight_file = os.path.join(args.data_dir, args.weight_dir, args.weights)
        detector = Detector(yolo, weight_file)
    
        if args.camera:
            # detect from camera
            if (platform.release() == '4.4.15-tegra'): #only way to make it work on jetson
                cap = cv2.VideoCapture("nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720, \
                    format=(string)I420, framerate=(fraction)12/1 ! \
        			nvvidconv flip-method=6 ! video/x-raw, format=(string)I420 ! \
        			videoconvert ! video/x-raw, format=(string)BGR ! \
        			appsink")
            else:
                cap = cv2.videoCapture(0)
            detector(cap)
            
        if args.test_img:
            for img_name in os.listdir(cfg.TEST_DIR):
                detector(cfg.TEST_DIR + '/'+ img_name)
        
        if args.test_video:
            print('Test video ')
            print('Does video file exist : ', os.path.isfile(cfg.TEST_VIDEO_FILE))
    #        import skvideo.io
    #        cap = skvideo.io.vreader(cfg.TEST_VIDEO_FILE)
    #        cap = skvideo.io.VideoCapture(cfg.TEST_VIDEO_FILE)
            cap = cv2.VideoCapture(cfg.TEST_VIDEO_FILE) # doesnt work, dont know why
            print("opened", cap.isOpened())
            if cap.isOpened():
                detector(cap)
                
        if args.img_path:
            if os.path.isfile(args.img):
                detector(args.img)
            else:
                print('Image file, ', args.img,' does not exist !')
        
        if args.video_path:
            if os.path.isfile(args.video):
    
                detector(cv2.VideoCapture(args.video))
            else:
                print('Video file, ', args.video,' does not exist !')
    else:
        print('Nothing to do')
        print('use -h or --help for more information')
        
if __name__ == '__main__':
    main()
