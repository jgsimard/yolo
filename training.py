import os
import argparse
import network.config as cfg
from network.net import YOLO
from utils.pascal_voc import pascal_voc
from trainer import Trainer

def main():
    #command line interface
    parser = argparse.ArgumentParser()
    parser.add_argument('-threshold', default = cfg.THRESHOLD, type=float)
    parser.add_argument('-iou_threshold', default= cfg.IOU_THRESHOLD, type=float)
    parser.add_argument('-gpu', default='0', type=str)
    args = parser.parse_args()

    if args.gpu is not None:
        cfg.GPU = args.gpu

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.GPU
      
    yolo = YOLO()
    pascal = pascal_voc('train')

    trainer = Trainer(yolo, pascal)

    print('Start training ...')
    trainer.train()
    print('Done training.')

if __name__ == '__main__':
    main()
