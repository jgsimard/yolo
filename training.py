import os
import argparse
import network.config as cfg
from network.net import YOLO
from utils.pascal_voc import pascal_voc
from trainer import Trainer


def update_config_paths(data_dir, weights_file):
    cfg.DATA_PATH = data_dir
    cfg.PASCAL_PATH = os.path.join(data_dir, 'pascal_voc')
    cfg.CACHE_PATH = os.path.join(cfg.PASCAL_PATH, 'cache')
    cfg.OUTPUT_DIR = os.path.join(cfg.PASCAL_PATH, 'output')
    cfg.WEIGHTS_DIR = os.path.join(cfg.PASCAL_PATH, 'weights')
    cfg.WEIGHTS_FILE = os.path.join(cfg.WEIGHTS_DIR, weights_file)

def main():
    #command line interface
    parser = argparse.ArgumentParser()
    parser.add_argument('-weights', default="YOLO_small.ckpt", type=str)
    parser.add_argument('-data_dir', default="data", type=str)
    parser.add_argument('-threshold', default=0.2, type=float)
    parser.add_argument('-iou_threshold', default=0.5, type=float)
    parser.add_argument('-gpu', default='0', type=str)
    args = parser.parse_args()

    if args.gpu is not None:
        cfg.GPU = args.gpu

    if args.data_dir != cfg.DATA_PATH:
        update_config_paths(args.data_dir, args.weights)

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.GPU
      
    yolo = YOLO()
    pascal = pascal_voc('train')

    trainer = Trainer(yolo, pascal)

    print('Start training ...')
    trainer.train()
    print('Done training.')

if __name__ == '__main__':
    main()
