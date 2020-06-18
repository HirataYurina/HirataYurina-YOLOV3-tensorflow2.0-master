# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:yolo3.py
# software: PyCharm

import numpy as np
import tensorflow as tf


# ---------------------------------------------------#
#   yolo_head
#   将预测结果进行解码
#   param:
#       features:网络预测结果
#       anchors:anchors
#       num_classes:number of classes
#       input_shape:输入图像的形状(n, h, w, c)
# ---------------------------------------------------#
def yolo_decode(features, anchors, num_classes, input_shape, cal_loss=False):
    # (w,h)
    grid_shape = tf.shape(features)[1:3][::-1]
    num_anchors = len(anchors)
    # (1, 1, 1, 3, 2)
    anchors = tf.cast(tf.reshape(tf.constant(anchors), (1, 1, 1, num_anchors, 2)), features.dtype)

    features = tf.reshape(features, (-1, grid_shape[1], grid_shape[0], num_anchors, num_classes + 5))
    # (n, h, w, 3, 2)
    box_xy = features[..., :2]
    box_wh = features[..., 2:4]
    grid_x = tf.reshape(tf.range(grid_shape[0]), (1, -1, 1, 1))
    # (1, h, w, 1, 2)
    grid_x = tf.tile(grid_x, (grid_shape[1], 1, 1, 1))
    grid_y = tf.reshape(tf.range(grid_shape[1]), (-1, 1, 1, 1))
    grid_y = tf.tile(grid_y, (1, grid_shape[0], 1, 1))

    grid_xy = tf.concat([grid_x, grid_y], axis=-1)
    grid_xy = tf.cast(grid_xy, features.dtype)

    box_xy = 1 / (1 + tf.exp(-box_xy)) + grid_xy
    box_wh = anchors * tf.exp(box_wh)

    # 归一化
    box_xy = box_xy / tf.cast(grid_shape, features.dtype)
    box_wh = box_wh / tf.cast(input_shape[::-1], features.dtype)

    confidence = 1 / (1 + tf.exp(-features[..., 4]))
    class_prob = 1 / (1 + tf.exp(-features[..., 5:]))

    if cal_loss:
        return grid_xy, features, box_xy, box_wh

    return box_xy, box_wh, confidence, class_prob


# ---------------------------------------------------#
#   在fed数据时，对图片进行了letter_box处理
#   对box进行调整，使其符合真实图片的样子
# ---------------------------------------------------#
def yolo_correct_box(box_xy, box_wh, input_shape, img_shape):
    # 先将归一化后的预测值进行还原
    box_yx = box_xy[::-1] * input_shape
    box_hw = box_wh[::-1] * input_shape

    scale = np.min(input_shape / img_shape)
    new_shape = np.round(img_shape * scale)
    offset = (input_shape - new_shape) / 2
    box_yx = (box_yx - offset) / new_shape * img_shape
    box_hw = box_hw / new_shape * img_shape
    # 左上角
    x1y1 = (box_yx - box_hw / 2)[::-1]
    # 右下角
    x2y2 = (box_yx + box_hw / 2)[::-1]

    return x1y1, x2y2


if __name__ == '__main__':
    random1 = np.random.rand(2, 2)
    random2 = np.random.rand(13, 13, 1, 2)
    print((random1 + random2).shape)
