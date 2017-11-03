import numpy as np
import tensorflow as tf
import network.config as cfg

slim = tf.contrib.slim

'''
TODO

Build a network that can retrain only the final n layers
'''

class YOLO(object):
    '''
    Class that contains the network and its corresponding loss function
    
    The network parameters are contained in the config.py file    
    '''

    def __init__(self, is_training=True):
        self.classes        = cfg.CLASSES
        self.num_class      = len(self.classes)
        self.image_size     = cfg.IMAGE_SIZE
        self.cell_size      = cfg.CELL_SIZE
        self.boxes_per_cell = cfg.BOXES_PER_CELL
        
        self.output_size      = (self.cell_size ** 2) * (self.num_class + self.boxes_per_cell * 5)
        self.scale            = 1.0 * self.image_size / self.cell_size
        self.boundary_classes = (self.cell_size ** 2) * self.num_class #taille des prediction sur les classes
        self.boundary_scale   = self.boundary_classes + self.cell_size ** 2 * self.boxes_per_cell 

        self.object_scale    = cfg.OBJECT_SCALE
        self.no_object_scale = cfg.NO_OBJECT_SCALE
        self.class_scale     = cfg.CLASS_SCALE
        self.coord_scale     = cfg.COORD_SCALE

        self.learning_rate = cfg.LEARNING_RATE
        self.batch_size    = cfg.BATCH_SIZE
        self.alpha         = cfg.ALPHA

        self.offset = np.transpose(
                        np.reshape(
                            np.array([np.arange(self.cell_size)] * self.cell_size * self.boxes_per_cell),
                            (self.boxes_per_cell, self.cell_size, self.cell_size)),
                      (1, 2, 0))
        
        #input of the network
        self.images = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, 3], name='images')
        #output of the network
        self.logits = self.build_network(self.images, num_outputs=self.output_size, alpha=self.alpha, is_training=is_training)
        
        #build the part of the graph necessary for training only if necessary
        if is_training:
            self.labels = tf.placeholder(tf.float32, [None, self.cell_size, self.cell_size, 5 + self.num_class])
            self.loss_layer(self.logits, self.labels)
            self.total_loss = tf.losses.get_total_loss()
            tf.summary.scalar('total_loss', self.total_loss)

    def build_network(self, images, num_outputs, alpha, keep_prob = 0.5, is_training = True, scope = 'yolo'):
        with tf.variable_scope(scope):
            with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                activation_fn=leaky_relu(alpha),
                                weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                                weights_regularizer=slim.l2_regularizer(0.0005)):
                
                #Stage 1
                net = tf.pad(images, np.array([[0, 0], [3, 3], [3, 3], [0, 0]]), name='pad_1')
                net = slim.conv2d(net, 64, 7, 2, padding='VALID', scope='conv_2')
                net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_3')
                
                #Stage 2
                net = slim.conv2d(net, 192, 3, scope='conv_4')
                net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_5')
                
                #Stage 3
                net = slim.conv2d(net, 128, 1, scope='conv_6')
                net = slim.conv2d(net, 256, 3, scope='conv_7')
                net = slim.conv2d(net, 256, 1, scope='conv_8')
                net = slim.conv2d(net, 512, 3, scope='conv_9')
                net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_10')
                
                #Stage 4
                for i in range(4):
                    net = slim.conv2d(net, 256, 1, scope='conv_' + str(11+i*2))
                    net = slim.conv2d(net, 512, 3, scope='conv_' + str(12+i*2))
                net = slim.conv2d(net, 512, 1, scope='conv_19')
                net = slim.conv2d(net, 1024, 3, scope='conv_20')
                net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_21')
                
                #Stage 5
                for i in range(2):
                    net = slim.conv2d(net, 512, 1, scope='conv_' + str(22 + i * 2))
                    net = slim.conv2d(net, 1024, 3, scope='conv_' + str(23 + i * 2))
                    
                net = slim.conv2d(net, 1024, 3, scope='conv_26')
                net = tf.pad(net, np.array([[0, 0], [1, 1], [1, 1], [0, 0]]), name='pad_27')
                net = slim.conv2d(net, 1024, 3, 2, padding='VALID', scope='conv_28')
                
                #Stage 6
                net = slim.conv2d(net, 1024, 3, scope='conv_29')
                net = slim.conv2d(net, 1024, 3, scope='conv_30')
                
                #Stage 7
                net = tf.transpose(net, [0, 3, 1, 2], name='trans_31')
                net = slim.flatten(net, scope='flat_32')
                
                #Stage  not in original structure
                net = slim.fully_connected(net, 512, scope='fc_33')
                net = slim.fully_connected(net, 4096, scope='fc_34')
                net = slim.dropout(net, keep_prob = keep_prob, is_training = is_training, scope='dropout_35')
                
                #Stage 8
                net = slim.fully_connected(net, num_outputs, activation_fn=None, scope='fc_36')
        return net

    def iou(self, boxes1, boxes2, scope='iou'):
        """
        calculate ious (intersertion over union) of 2 set of boxes
        Inputs:
          boxes1: 5-D tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4] : 4 ===> (x_center, y_center, w, h)
          boxes2: 5-D tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4] : 4 ===> (x_center, y_center, w, h)
        Output:
          iou: 4-D tensor [BATCH_SIZE,CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
        """
        with tf.variable_scope(scope):
            #5-D : [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4] ===> (left, up, right, down)
            def extreme_pts(boxes):
                return tf.stack([boxes[:, :, :, :, 0] - boxes[:, :, :, :, 2] / 2.0,  #left
                                 boxes[:, :, :, :, 1] - boxes[:, :, :, :, 3] / 2.0,  #up, axes are the inverse
                                 boxes[:, :, :, :, 0] + boxes[:, :, :, :, 2] / 2.0,  #right
                                 boxes[:, :, :, :, 1] + boxes[:, :, :, :, 3] / 2.0], #down
                                axis = 4)
                 
            boxes1 = extreme_pts(boxes1)
            boxes2 = extreme_pts(boxes2)

            # calculate the left up point & right down point
            lu = tf.maximum(boxes1[:, :, :, :, :2], boxes2[:, :, :, :, :2]) # left and up
            rd = tf.minimum(boxes1[:, :, :, :, 2:], boxes2[:, :, :, :, 2:]) # right and down

            # intersection , create 4D tensor
            intersection = tf.maximum(0.0, rd - lu)
            inter_square = intersection[:, :, :, :, 0] * intersection[:, :, :, :, 1]

            # calculate the boxs1 square and boxs2 square, creates 4D tensor
            square1 = (boxes1[:, :, :, :, 2] - boxes1[:, :, :, :, 0]) * (boxes1[:, :, :, :, 3] - boxes1[:, :, :, :, 1])
            square2 = (boxes2[:, :, :, :, 2] - boxes2[:, :, :, :, 0]) * (boxes2[:, :, :, :, 3] - boxes2[:, :, :, :, 1])

            union_square = tf.maximum(square1 + square2 - inter_square, 1e-10)

        return tf.clip_by_value(inter_square / union_square, 0.0, 1.0)

    def loss_layer(self, predicts, labels, scope='loss_layer'):
        '''
        Define the loss function of the network
        Input:
            predicts : tensor of prediction     : [?, outputsize]
            labels   : tensor of correct labels : [?, outputsize]
        '''
        with tf.variable_scope(scope):
            predict_classes = tf.reshape(predicts[:, :self.boundary_classes], [self.batch_size, self.cell_size, self.cell_size, self.num_class])
            predict_scales = tf.reshape(predicts[:, self.boundary_classes:self.boundary_scale], [self.batch_size, self.cell_size, self.cell_size, self.boxes_per_cell])
            predict_boxes = tf.reshape(predicts[:, self.boundary_scale:], [self.batch_size, self.cell_size, self.cell_size, self.boxes_per_cell, 4])

            response = tf.reshape(labels[:, :, :, 0], [self.batch_size, self.cell_size, self.cell_size, 1])
            boxes = tf.reshape(labels[:, :, :, 1:5], [self.batch_size, self.cell_size, self.cell_size, 1, 4])
            boxes = tf.tile(boxes, [1, 1, 1, self.boxes_per_cell, 1]) / self.image_size
            classes = labels[:, :, :, 5:]

            offset = tf.constant(self.offset, dtype=tf.float32)
            offset = tf.reshape(offset, [1, self.cell_size, self.cell_size, self.boxes_per_cell])
            offset = tf.tile(offset, [self.batch_size, 1, 1, 1])
            
            predict_boxes_tran = tf.stack([(predict_boxes[:, :, :, :, 0] + offset) / self.cell_size, # 
                                           (predict_boxes[:, :, :, :, 1] + tf.transpose(offset, (0, 2, 1, 3))) / self.cell_size, #because the y axis goes down in a pictue
                                           tf.square(predict_boxes[:, :, :, :, 2]), # box height and wigth variation are less important then absolute position
                                           tf.square(predict_boxes[:, :, :, :, 3])],
                                            axis = 4)

            iou_predict_truth = self.iou(predict_boxes_tran, boxes)

            # calculate I tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
            '''
            The tensor I determine which predicator is responsible for the preciton by picking the highest IOU
            between the prediction and the ground truth. The tensor is also <<active>> only if there is an object 
            in the cell. This enable the use of conditional probability            
            '''
            object_mask = tf.reduce_max(iou_predict_truth, 3, keep_dims=True)
            object_mask = tf.cast((iou_predict_truth >= object_mask), tf.float32) * response

            # calculate no_I tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
            no_object_mask = tf.ones_like(object_mask, dtype=tf.float32) - object_mask

            boxes_tran = tf.stack([boxes[:, :, :, :, 0] * self.cell_size - offset,
                                   boxes[:, :, :, :, 1] * self.cell_size - tf.transpose(offset, (0, 2, 1, 3)),
                                   tf.sqrt(boxes[:, :, :, :, 2]),
                                   tf.sqrt(boxes[:, :, :, :, 3])],
                                    axis = 4)
            
            #reduce mean -> take the average over the batch
            # class_loss 
            class_delta = response * (predict_classes - classes)
            class_loss = self.class_scale * tf.reduce_mean(tf.reduce_sum(tf.square(class_delta), axis=[1, 2, 3]), name='class_loss')

            # object_loss (error of predicting if an object is in a cell given that there is one)
            object_delta = object_mask * (predict_scales - iou_predict_truth)
            object_loss = self.object_scale * tf.reduce_mean(tf.reduce_sum(tf.square(object_delta), axis=[1, 2, 3]), name='object_loss')

            # no_object_loss (error of predicting if an object is in a cell given that the cell is empty, this exost to not averpower the cell containg objects)
            no_object_delta = no_object_mask * predict_scales
            no_object_loss = self.no_object_scale * tf.reduce_mean(tf.reduce_sum(tf.square(no_object_delta), axis=[1, 2, 3]), name='noobject_loss')

            # coord_loss
            coord_mask = tf.expand_dims(object_mask, 4)
            boxes_delta = coord_mask * (predict_boxes - boxes_tran)
            coord_loss = self.coord_scale * tf.reduce_mean(tf.reduce_sum(tf.square(boxes_delta), axis=[1, 2, 3, 4]), name='coord_loss')

            tf.losses.add_loss(class_loss)
            tf.losses.add_loss(object_loss)
            tf.losses.add_loss(no_object_loss)
            tf.losses.add_loss(coord_loss)

            tf.summary.scalar('class_loss', class_loss)
            tf.summary.scalar('object_loss', object_loss)
            tf.summary.scalar('noobject_loss', no_object_loss)
            tf.summary.scalar('coord_loss', coord_loss)

#            tf.summary.histogram('boxes_delta_x', boxes_delta[:, :, :, :, 0])
#            tf.summary.histogram('boxes_delta_y', boxes_delta[:, :, :, :, 1])
#            tf.summary.histogram('boxes_delta_w', boxes_delta[:, :, :, :, 2])
#            tf.summary.histogram('boxes_delta_h', boxes_delta[:, :, :, :, 3])
            tf.summary.histogram('iou', iou_predict_truth)


def leaky_relu(alpha):
    '''
    Defines the avtivations function of the leaky-relu
    Input:
        alpha : the slope of the function when the input is negative
    Ouput:
        op : operation of leaky-relu
    '''
    def op(inputs):
        return tf.maximum(alpha * inputs, inputs, name='leaky_relu')
    return op
