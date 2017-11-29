import os
from argparse import ArgumentParser
import network.config as cfg
from network.net import YOLO
from network.detector import Detector

def main():    
    
    parser = ArgumentParser(description=' Test the detector ')
    parser.add_argument('-weights',    metavar='', default=cfg.WEIGHTS_FILE,           help = 'Weights file' )
    parser.add_argument('-gpu',        action='store_const', const='0', default='',    help = 'Use gpu for inference')  
    parser.add_argument('-file_path',  metavar= None,                                  help = 'Test the detector on a media file, give the path')
    parser.add_argument('-camera',     action='store_const', const=cfg.CMD_CAMERA,     help = 'Test the detector on the live feed of the camera')
    parser.add_argument('-test_img',   action='store_const', const=cfg.CMD_TEST_IMG,   help = 'Test the detector on sample images')
    parser.add_argument('-test_video', action='store_const', const=cfg.CMD_TEST_VIDEO, help = 'Test the detector on a sample video')
    parser.add_argument('-save',       action='store_const', const=True,               help = 'Save the results')   
    parser.add_argument('-alive',      action='store_const', const=cfg.CMD_ALIVE,      help = 'Wait for files to analyse') 
    args = parser.parse_args()
    
    cmd = next((i for i in [args.camera,  args.test_img, args.test_video, args.file_path, args.alive]  if i is not None), None)
    
    if cmd :
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu if args.gpu is not None else cfg.GPU
        detector = Detector(YOLO(False), args.weights, args.save)
        detector(cmd)              
    else:
        print('Nothing to do \nUse -h or --help for more information')
        
if __name__ == '__main__':
    main()