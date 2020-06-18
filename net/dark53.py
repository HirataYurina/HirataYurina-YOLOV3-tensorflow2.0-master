# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:2020/4/12 0012 15:37
# filename:dark53.py
# software: PyCharm

import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import tensorflow as tf


# -------------------- #
# 单次卷积 无激活函数层和BN层
# -------------------- #
def darknet_conv2d(inputs, filters, kernel_size, strides, use_bias=True):
    padding = 'valid' if strides == 2 else 'same'

    y = layers.Conv2D(filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias,
                      kernel_regularizer=keras.regularizers.l2(5e-4))(inputs)

    return y


# -------------------- #
# 卷积 BN+leaky
# -------------------- #
def darknet_con2d_bn_leaky(inputs, filters, kernel_size, strides):
    y = darknet_conv2d(inputs, filters, kernel_size, strides, use_bias=False)
    y = layers.BatchNormalization()(y)
    y = layers.LeakyReLU(alpha=0.1)(y)

    return y


# -------------------- #
# 残差块
# -------------------- #
def darknet_res_block(inputs, filters, num_block):
    x = layers.ZeroPadding2D(((1, 0), (1, 0)))(inputs)
    x = darknet_con2d_bn_leaky(x, filters, 3, 2)
    for i in range(num_block):
        y = darknet_con2d_bn_leaky(x, filters // 2, 1, 1)
        y = darknet_con2d_bn_leaky(y, filters, 3, 1)
        # 残差
        x = layers.Add()([x, y])

    return x


# ---------------------------------------------------#
#   特征层->最后的输出
# ---------------------------------------------------#
def make_last_layer(inputs, filters, out_filters):
    x = darknet_con2d_bn_leaky(inputs, filters, 1, 1)
    x = darknet_con2d_bn_leaky(x, filters * 2, 3, 1)
    x = darknet_con2d_bn_leaky(x, filters, 1, 1)
    x = darknet_con2d_bn_leaky(x, filters * 2, 3, 1)
    x = darknet_con2d_bn_leaky(x, filters, 1, 1)

    y = darknet_con2d_bn_leaky(x, filters * 2, 3, 1)
    # 最后一层不需要激活函数
    outputs = darknet_conv2d(y, out_filters, 1, 1)

    return outputs, x


# ---------------------------------------------------#
#   yolo3的整个架构
#   输出3个stage
# ---------------------------------------------------#
def yolo3_body(inputs, num_anchors, num_classes):

    out_shape = num_anchors * (num_classes + 5)

    # darknet53
    y = darknet_con2d_bn_leaky(inputs, 32, 3, 1)
    y = darknet_res_block(y, 64, 1)
    y = darknet_res_block(y, 128, 2)
    y1 = darknet_res_block(y, 256, 8)
    y2 = darknet_res_block(y1, 512, 8)
    y3 = darknet_res_block(y2, 1024, 4)

    # stage1
    feature1, x3 = make_last_layer(y3, 512, out_shape)
    x3 = darknet_con2d_bn_leaky(x3, 256, 1, 1)
    x3 = layers.UpSampling2D(2)(x3)
    y2 = layers.Concatenate()([x3, y2])

    # stage2
    feature2, x2 = make_last_layer(y2, 256, out_shape)
    x2 = darknet_con2d_bn_leaky(x2, 128, 1, 1)
    x2 = layers.UpSampling2D(2)(x2)
    y1 = layers.Concatenate()([x2, y1])

    # stage3
    feature3, _ = make_last_layer(y1, 128, out_shape)

    return keras.Model(inputs, [feature1, feature2, feature3])


if __name__ == '__main__':
    inputs_img = keras.Input(shape=(416, 416, 3))

    yolo3_model = yolo3_body(inputs_img, 3, 80)

    yolo3_model.load_weights('../weights/yolo_weights.h5')

    # yolo3_model.summary()

    # print(len(yolo3_model.layers))

    from PIL import Image, ImageDraw, ImageFont
    from utils.util import letterbox_image, correct_boxes, nms
    import numpy as np
    import colorsys
    from net.yolo3 import yolo_decode
    import matplotlib.pyplot as plt

    street = Image.open('../img/street.jpg')
    h, w = street.size
    street_letter_box = letterbox_image(street, (416, 416))

    # plt.imshow(street_letter_box)
    # plt.show()

    street_letter_box = np.array(street_letter_box, 'float32').reshape((1, 416, 416, 3)) / 255.0
    # print(street_letter_box.size)
    features = yolo3_model(street_letter_box)
    # print(features[0])
    # for feature in features:
        # print(feature.shape)

    with open('../yolo_data/yolo_anchors.txt') as f:
        line = f.readline().split(',')
        anchors = [float(x) for x in line]

    with open('../yolo_data/coco_classes.txt') as f:
        lines = f.readlines()
        class_names = [class_name.strip() for class_name in lines]

    anchors = np.reshape(anchors, (-1, 2))
    num_classes = 80
    anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

    # street_cv2 = cv2.imread('../img/street.jpg')

    boxes_total = []
    scores_total = []

    # set font
    font = ImageFont.truetype(font='font/simhei.ttf',
                              size=np.floor(3e-2 * street.size[1] + 0.5).astype('int32'))
    thickness = (street.size[0] + street.size[1]) // 300

    # set different color
    hsv_tuples = [(x / len(class_names), 1., 1.)
                  for x in range(len(class_names))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    # print(colors)
    # shuffle color
    np.random.shuffle(colors)

    for i, feature in enumerate(features):
        box_xy, box_wh, confidence, class_prob = yolo_decode(feature, anchors[anchors_mask[i]], 80, (416, 416))
        # print(box_xy.shape)
        # print(confidence.shape)
        # print(class_prob.shape)
        box_xy = box_xy * 416
        box_wh = box_wh * 416
        # (1, 13, 13, 3, 2)
        # correct boxes in original image
        box_xy = np.array(box_xy, 'float32')
        box_wh = np.array(box_wh, 'float32')
        boxes = correct_boxes(box_xy, box_wh, (416, 416), (h, w))
        # scores = object_prob * class_prob
        scores = np.expand_dims(confidence, axis=-1) * class_prob
        boxes = np.reshape(boxes, newshape=(-1, 4))
        scores = np.reshape(scores, (-1, num_classes))
        print('the shape of boxes is{}'.format(boxes.shape))
        print('the shape of scores is {}'.format(scores.shape))

        boxes_total.append(boxes)
        scores_total.append(scores)

    boxes_total = np.concatenate(boxes_total, axis=0)
    scores_total = np.concatenate(scores_total, axis=0)

    # implement nms for each class
    for i in range(num_classes):
        score_per_class = scores_total[..., i:i+1]
        rectangles = np.concatenate([boxes_total, score_per_class], axis=-1)
        have_object = np.where(rectangles[..., 4] > 0.6)[0]
        # print(have_object)
        rectangles = rectangles[have_object]
        pick = nms(rectangles, threshold=0.3)
        # pick = tf.image.non_max_suppression(rectangles[..., 0:4], rectangles[..., 4], 20, 0.3)

        # print(pick)
        # boxes_pick = tf.gather(rectangles, pick)
        if pick:
            boxes_pick = rectangles[pick]
            for box in boxes_pick:
                # cv2.rectangle(street_cv2, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)

                # correct box
                x1, y1, x2, y2, _ = box
                left = int(max(0, x1))
                top = int(max(0, y1))
                right = int(min(street.size[0], x2))
                bottom = int(min(street.size[1], y2))

                label = '{} {:.2f}'.format(class_names[i], box[4])
                draw = ImageDraw.Draw(street)
                label_size = draw.textsize(label, font)
                label = label.encode('utf-8')
                print(label)

                if top - label_size[1] >= 0:
                    text_origin = np.array([left, top - label_size[1]])
                else:
                    text_origin = np.array([left, top + 1])

                for j in range(thickness):
                    draw.rectangle(
                        [left + j, top + j, right - j, bottom - j],
                        outline=colors[i])
                draw.rectangle(
                    [tuple(text_origin), tuple(text_origin + label_size)],
                    fill=colors[i])
                draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)

                # print(box)
                # print(i)

    # cv2.imshow('street', street_cv2)
    # cv2.waitKey(0)
    # cv2.imwrite('../img/street_result.jpg', street_cv2)

    plt.imshow(street)
    plt.show()

    # save image
    street.save('../img/street_result.jpg')
