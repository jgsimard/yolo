import tensorflow as tf
import numpy as np
import cv2
import network.config as cfg
from utils.timer import step_time
from network.net import YoloNetwork
import os
import platform
import mimetypes


class Detector(object):
    def __init__(self, save_file):
        self.yolo_model = YoloModel(save_file)
        self.classes = cfg.CLASSES
        self.colors = cfg.COLORS
        self.save = True if save_file is not None else False

    def draw_prediction(self, img, result):
        for i in range(len(result)):
            x, y = int(result[i][1]), int(result[i][2])
            w, h = int(result[i][3] / 2), int(result[i][4] / 2)

            # bouding box
            cv2.rectangle(img, (x - w, y - h), (x + w, y + h), self.colors[result[i][0]], 2)

            # text background box
            cv2.rectangle(img, (x - w, y - h - 20), (x + w, y - h), self.colors[result[i][0]], -1)

            # class name
            cv2.putText(img,
                        self.classes[result[i][0]] + ' : %.2f' % result[i][5],
                        (x - w + 5, y - h - 7),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                        cv2.LINE_AA)

    def image(self, img, img_name, wait=0):
        result = self.yolo_model.inference(img)
        self.draw_prediction(img, result)
        cv2.imshow(img_name, img)
        cv2.waitKey(wait)
        cv2.destroyAllWindows()

        if self.save:
            if not os.path.exists('results'):
                os.makedirs('results')
            cv2.imwrite(os.path.join('results', os.path.basename(img_name)), img)

    def image_file(self, input_name):
        image = cv2.imread(input_name)
        self.image(image, input_name, 0)

    def video_file(self, input_name, wait):
        cap = cv2.VideoCapture(input_name)
        if cap.isOpened():
            while True:
                ret, frame = cap.read()
                if ret:
                    self.image(frame, 'Image of video' + input_name, wait)
                else:
                    break
        else:
            print('Impossible to open video :', input_name)

    def file(self, input_name):
        t = mimetypes.guess_type(input_name)
        if t[0] is not None:
            file_type = t[0].split('/')[0]
            if file_type == 'image':
                self.image_file(input_name)
            elif file_type == 'video':
                self.video_file(input_name)
        else:
            print('Impossible to open file :', input_name)

    def test_images(self):
        for img_name in os.listdir(cfg.TEST_IMG_DIR):
            self.image_file(os.path.join(cfg.TEST_IMG_DIR, img_name))

    def test_videos(self):
        for video_name in os.listdir(cfg.TEST_VIDEO_DIR):
            self.video_file(os.path.join(cfg.TEST_VIDEO_DIR, video_name), 10)

    def alive(self):
        while True:
            self.file(input("Give me a file to analyse : "))

    def camera(self):
        if platform.release() == '4.4.15-tegra':  # only way to make it work on jetson
            self.video_file("nvcamerasrc \
                            ! video/x-raw(memory:NVMM), \
                            width=(int)1280, \
                            height=(int)720, \
                            format=(string)I420,\
                            framerate=(fraction)12/1 ! \
                            nvvidconv flip-method=6 ! \
                            video/x-raw, format=(string)I420 ! \
                            videoconvert ! \
                            video/x-raw, \
                            format=(string)BGR ! \
                            appsink", 10)
        else:
            self.video_file(0, 10)


class YoloModel(object):

    def __init__(self, save_file):
        self.net = YoloNetwork()
        self.weights_file = cfg.WEIGHTS_FILE

        self.save = save_file
        self.num_class = len(cfg.CLASSES)
        self.image_size = cfg.IMAGE_SIZE
        self.cell_size = cfg.CELL_SIZE
        self.boxes_per_cell = cfg.BOXES_PER_CELL
        self.threshold = cfg.THRESHOLD
        self.iou_threshold = cfg.IOU_THRESHOLD
        self.boundary1 = self.cell_size ** 2 * self.num_class
        self.boundary2 = self.cell_size ** 2 * self.boxes_per_cell + self.boundary1

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        print('Restoring weights from: ' + self.weights_file)
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, self.weights_file)

    @step_time('Detection')
    def inference(self, inputs):
        img_h, img_w, _ = inputs.shape
        inputs = cv2.resize(inputs, (self.image_size, self.image_size))
        inputs = cv2.cvtColor(inputs, cv2.COLOR_BGR2RGB).astype(np.float32)  # because opencv is weird
        inputs = (inputs / 255.0) * 2.0 - 1.0  # normalization
        inputs = np.reshape(inputs, (1, self.image_size, self.image_size, 3))  # 4D tensor (batch_size = 1)

        net_output = self.sess.run(self.net.logits, feed_dict={self.net.images: inputs})

        results = []
        for i in range(net_output.shape[0]):
            results.append(self.interpret(net_output[i]))

        result = results[0]

        for i in range(len(result)):
            result[i][1] *= img_w / self.image_size
            result[i][2] *= img_h / self.image_size
            result[i][3] *= img_w / self.image_size
            result[i][4] *= img_h / self.image_size

        return result

    def iou(self, box1, box2):
        ud = min(box1[0] + box1[2] / 2, box2[0] + box2[2] / 2) - max(box1[0] - box1[2] / 2, box2[0] - box2[2] / 2)
        lr = min(box1[1] + box1[3] / 2, box2[1] + box2[3] / 2) - max(box1[1] - box1[3] / 2, box2[1] - box2[3] / 2)

        intersection = 0 if (ud < 0 or lr < 0) else ud * lr

        return intersection / (box1[2] * box1[3] + box2[2] * box2[3] - intersection)

    def interpret(self, output):

        probs = np.zeros((self.cell_size, self.cell_size, self.boxes_per_cell, self.num_class))

        # P(class|object is present)
        class_probs = np.reshape(output[0:self.boundary1], (self.cell_size, self.cell_size, self.num_class))

        # confidence score of detected boxes = P(object is present)
        scales = np.reshape(output[self.boundary1:self.boundary2],
                            (self.cell_size, self.cell_size, self.boxes_per_cell))
        # (x,y,sqrt(w),sqrt(h)) of the detected boxes
        boxes = np.reshape(output[self.boundary2:], (self.cell_size, self.cell_size, self.boxes_per_cell, 4))
        offset = np.transpose(np.reshape(np.array([np.arange(self.cell_size)] * self.cell_size * self.boxes_per_cell),
                                         [self.boxes_per_cell, self.cell_size, self.cell_size]), (1, 2, 0))

        boxes[:, :, :, 0] += offset
        boxes[:, :, :, 1] += np.transpose(offset, (1, 0, 2))
        boxes[:, :, :, :2] /= self.cell_size
        boxes[:, :, :, 2:] = np.square(boxes[:, :, :, 2:])

        boxes *= self.image_size

        # compute likelihood of each class as each cell
        for i in range(self.boxes_per_cell):
            for j in range(self.num_class):
                # p(class) = P(class|object is present)P(object is present)
                probs[:, :, i, j] = np.multiply(class_probs[:, :, j], scales[:, :, i])

        filter_mat_probs = np.array(probs >= self.threshold, dtype='bool')
        filter_mat_boxes = np.nonzero(filter_mat_probs)  # indices

        # box that heve at least one likely class
        boxes_filtered = boxes[filter_mat_boxes[0], filter_mat_boxes[1], filter_mat_boxes[2]]

        # probs above the threshold
        probs_filtered = probs[filter_mat_probs]

        # find the most likely class for each cell
        classes_num_filtered = np.argmax(filter_mat_probs, axis=3)[filter_mat_boxes[0], \
                                                                   filter_mat_boxes[1], \
                                                                   filter_mat_boxes[2]]
        # sort the  from most likely to less likely
        argsort = np.array(np.argsort(probs_filtered))[::-1]
        boxes_filtered = boxes_filtered[argsort]
        probs_filtered = probs_filtered[argsort]
        classes_num_filtered = classes_num_filtered[argsort]

        # filter overlaping boxes
        for i in range(len(boxes_filtered)):
            if probs_filtered[i] == 0:
                continue
            for j in range(i + 1, len(boxes_filtered)):
                # if two boxes are over each other, the least likely of the two is destroyed
                if self.iou(boxes_filtered[i], boxes_filtered[j]) > self.iou_threshold:
                    probs_filtered[j] = 0.0

        filter_iou = np.array(probs_filtered > 0.0, dtype='bool')
        boxes_filtered = boxes_filtered[filter_iou]
        probs_filtered = probs_filtered[filter_iou]
        classes_num_filtered = classes_num_filtered[filter_iou]

        result = []
        for i in range(len(boxes_filtered)):
            result.append([classes_num_filtered[i],
                           boxes_filtered[i][0], boxes_filtered[i][1],
                           boxes_filtered[i][2], boxes_filtered[i][3],
                           probs_filtered[i]])

        return result