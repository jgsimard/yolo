import os
import argparse
import network.config as cfg
from network.net import YOLO
from network.detector import Detector

def main():    
    parser = argparse.ArgumentParser(description=' Test the detector ')
    parser.add_argument('--weights', default=cfg.WEIGHTS_FILE, metavar='', type=str, help= ' weights file' )
    parser.add_argument('--gpu', default='', action='store_const', const='0', help = 'Use gpu for inference')  
    parser.add_argument('--camera', action='store_const', const=True, help = 'Test the detector on the live feed of the camera')
    parser.add_argument('--file_path',metavar='', help = 'Test the detector on a media file, give the path')
    parser.add_argument('--test_img', action='store_const', const=True, help = 'Test the detector on sample images')
    parser.add_argument('--test_video', action='store_const', const=True, help = 'Test the detector on a sample video')
    parser.add_argument('--save', action='store_const', const=True, help = 'Save the results')   
    parser.add_argument('--alive', action='store_const', const=True, help = 'Wait for files to analyse') 
    args = parser.parse_args()
       
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    if(args.camera or args.test_img or args.test_video or args.file_path or args.alive):
        detector = Detector(YOLO(False), args.weights, args.save)
    
        if args.camera:
            detector(cfg.CAMERA)
            
        if args.test_img:
            detector(cfg.TEST_IMG)
                   
        if args.test_video:
            detector(cfg.TEST_VIDEO)
                
        if args.file_path:
            detector(args.file_path)
            
        if args.alive:
            while True:
               detector(input( "Give me a file to analyse : "))
               
    else:
        print('Nothing to do \nuse -h or --help for more information')
        
if __name__ == '__main__':
    main()
