import os

''' Paths '''
NETWORK_PATH = 'network'
DATA_PATH = os.path.join(NETWORK_PATH, 'data')
PASCAL_PATH = os.path.join(DATA_PATH, 'pascal_voc')
CACHE_PATH = os.path.join(PASCAL_PATH, 'cache')
OUTPUT_DIR = os.path.join(PASCAL_PATH, 'output')
WEIGHTS_DIR = os.path.join(DATA_PATH, 'weights')
WEIGHTS_FILE = os.path.join(WEIGHTS_DIR, 'weights_JG')

TEST_DIR = 'test'
TEST_VIDEO_DIR = os.path.join(TEST_DIR, 'video')
TEST_IMG_DIR = os.path.join(TEST_DIR, 'img')

CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
           'bus', 'car', 'cat', 'chair', 'cow',
           'diningtable', 'dog', 'horse', 'motorbike', 'person',
           'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

# random set of color
COLORS = [(200, 220, 120),
          (250, 33, 244),
          (156, 250, 58),
          (202, 168, 95),
          (12, 240, 35),
          (66, 245, 233),
          (53, 25, 231),
          (231, 32, 195),
          (254, 42, 52),
          (170, 83, 239),
          (127, 148, 250),
          (134, 164, 166),
          (126, 125, 1),
          (65, 228, 212),
          (24, 219, 199),
          (253, 204, 234),
          (144, 156, 215),
          (125, 103, 182),
          (134, 89, 17),
          (41, 223, 235)]
FLIPPED = True

''' Model '''
IMAGE_SIZE = 448
CELL_SIZE = 7
BOXES_PER_CELL = 2
ALPHA = 0.1
DISP_CONSOLE = False
OBJECT_SCALE = 1.0
NO_OBJECT_SCALE = 1.0
CLASS_SCALE = 2.0
COORD_SCALE = 5.0

''' Training '''
GPU = '0'
LEARNING_RATE = 0.0001
DECAY_STEPS = 10000
DECAY_RATE = 0.1
STAIRCASE = True
BATCH_SIZE = 45
MAX_ITER = 10000
SUMMARY_ITER = 10
SAVE_ITER = 1000

''' Test '''
THRESHOLD = 0.1
IOU_THRESHOLD = 0.5
