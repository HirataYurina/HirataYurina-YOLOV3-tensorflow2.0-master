# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:test_train.py
# software: PyCharm

from net.dark53 import yolo3_body
from net.yolo3 import yolo_decode, yolo_correct_box
from net.yolo3_loss import yolo3_loss
from utils.util import *
import cv2
import numpy as np
import tensorflow.keras as keras
import tensorflow as tf

# 测试get_random_date
input_shape = (416, 416)

with open('./2020_train.txt', encoding='utf-8') as f:
    annotations = f.readlines()
    annotations = [anno.strip() for anno in annotations]

annotation_line = annotations[0]

img_data, boxes_data = get_random_data(annotation_line, input_shape)

for rec in boxes_data[0:2]:
    x1 = int(rec[0])
    y1 = int(rec[1])
    x2 = int(rec[2])
    y2 = int(rec[3])
    cv2.rectangle(img_data, (x1, y1), (x2, y2), (0, 255, 0))

# cv2.imshow('random', img_data)
# cv2.waitKey(0)
# print(boxes_data[0:3])

# 测试true_boxes_preprocess

boxes_data = np.expand_dims(boxes_data, axis=0)
print(boxes_data.shape)

with open('../yolo_data/yolo_anchors.txt') as f:
    line = f.readline().split(',')
    anchors = [float(x) for x in line]

anchors = np.reshape(anchors, (-1, 2))
num_classes = 20

y_true = true_boxes_preprocess(boxes_data, input_shape, anchors, 20)

print(y_true[0].shape)

inputs = keras.Input(shape=(416, 416, 3))

yolo3_model = yolo3_body(inputs, 3, 20)
# yolo3_model.summary()

# 开始预测：
img_data = img_data[np.newaxis, :]
img_data = tf.convert_to_tensor(img_data, dtype=tf.float32)
feature1, feature2, feature3 = yolo3_model(img_data)
print(feature1.shape)
print(feature2.shape)
print(feature3.shape)
print(feature1.dtype)

# 测试loss函数
args = [feature1, feature2, feature3, *y_true]
losses = yolo3_loss(args, anchors, num_classes, ignore_threshold=0.5, coord_scale=5, noobj_scale=1)
print(losses)
