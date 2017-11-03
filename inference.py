import os
import cv2
import argparse
import platform
import network.config as cfg
from network.net import YOLO
from network.detector import Detector
import warnings

def main():    
    parser = argparse.ArgumentParser(description=' Test the detector ')
    parser.add_argument('--weights', default=cfg.WEIGHTS_FILE, metavar='', type=str, help= ' weights file' )
    parser.add_argument('--gpu', default='', action='store_const', const='0', help = 'Use gpu for inference')  
    parser.add_argument('--camera', action='store_const', const=True, help = 'Test the detector on the live feed of the camera')
    parser.add_argument('--img_path',metavar='', help = 'Test the detector on a static image, give the path')
    parser.add_argument('--video_path',metavar='', help = 'Test the detector on a video, give the path')
    parser.add_argument('--test_img', action='store_const', const=True, help = 'Test the detector on sample images')
    parser.add_argument('--test_video', action='store_const', const=True, help = 'Test the detector on a sample video')
    parser.add_argument('--save', action='store_const', const=True, help = 'Save the results')     
    args = parser.parse_args()
       
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if(args.camera or args.test_img or args.test_video or args.img_path or args.video_path):
        yolo = YOLO(False)
        detector = Detector(yolo, args.weights, args.save)
    
        if args.camera:
            # detect from camera
            if platform.release() == '4.4.15-tegra': #only way to make it work on jetson
                cap = cv2.VideoCapture("nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720, \
                    format=(string)I420, framerate=(fraction)12/1 ! \
        			nvvidconv flip-method=6 ! video/x-raw, format=(string)I420 ! \
        			videoconvert ! video/x-raw, format=(string)BGR ! \
        			appsink")
            else:
                cap = cv2.videoCapture(0)
            detector(cap)
            
        if args.test_img:
            print('Test images')
            for img_name in os.listdir(cfg.TEST_IMG_DIR):
                detector(os.path.join(cfg.TEST_IMG_DIR,img_name))
        
        if args.test_video:
            print('Test video')
            for video_name in os.listdir(cfg.TEST_VIDEO_DIR):
                cap = cv2.VideoCapture(os.path.join(cfg.TEST_VIDEO_DIR,video_name))
                if cap.isOpened():
                    detector(cap)
                else:
                    print('Impossible to video :', video_name)
                
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
