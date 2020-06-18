# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:yolo3_loss.py
# software: PyCharm

import tensorflow as tf
from net.yolo3 import yolo_decode


# ---------------------------------------------------#
#   计算iou
#   输入的为归一化后的值
#   box1(13, 13, 3, 4)
#   box2(n,4)
# ---------------------------------------------------#
def iou(box1, box2):
    # (13,13,3,1,4)
    box1 = tf.expand_dims(box1, axis=-2)
    # (13,13,3,1,2)
    box1_xy = box1[..., :2]
    box1_wh = box1[..., 2:]
    # 左上角和右下角
    box1_min = box1_xy - box1_wh / 2
    box1_max = box1_xy + box1_wh / 2

    # (1,n,4)
    box2 = tf.expand_dims(box2, axis=0)
    box2_xy = box2[..., :2]
    box2_wh = box2[..., 2:]
    box2_min = box2_xy - box2_wh / 2
    box2_max = box2_xy + box2_wh / 2

    # (13,13,3,n,2)
    intersect = tf.maximum(tf.minimum(box1_max, box2_max) - tf.maximum(box1_min, box2_min), 0)
    # (13,13,3,n)
    intersect_area = intersect[..., 0] * intersect[..., 1]
    ious = intersect_area / (box1_wh[..., 0] * box1_wh[..., 1] + box2_wh[..., 0] * box2_wh[..., 1] - intersect_area)

    return ious


# --------------------------------------------------------------------------#
#   计算loss
#   param:
#       args:[stage1, stage2, stage3, stage1_true, stage2_true, stage3_true]
#       stage1:(n,13,13,3,(5+num_classes))
#       stage1_ture:(n,13,13,3,(5+num_classes)) 已经经过response分配的label标签
# --------------------------------------------------------------------------#
def yolo3_loss(args, anchors, num_classes, ignore_threshold, coord_scale, noobj_scale):
    num_stages = len(anchors) // 3

    yolo_outputs = args[:num_stages]
    y_true = args[num_stages:]

    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_stages == 3 else [[3, 4, 5], [1, 2, 3]]
    # input_shape 32*(h,w)
    input_shape = tf.cast(tf.shape(yolo_outputs[0])[1:3] * 32, tf.float32)
    grid_shape = [tf.cast(tf.shape(yolo_outputs[i])[1:3], tf.float32) for i in range(num_stages)]
    # batch_size
    batch = tf.shape(yolo_outputs[0])[0]
    batch_float = tf.cast(batch, tf.float32)
    # 计算每个stage的loss

    loss = 0

    for i in range(num_stages):
        # (n,13,13,3,1)
        # (n,26,26,3,1)
        # (n,52,52,3,1)
        object_mask = tf.cast(y_true[i][..., 4:5], tf.float32)
        # (n,13,13,3,4)
        boxes = tf.cast(y_true[i][..., 0:4], tf.float32)
        # (n,13,13,3,20)
        classes = tf.cast(y_true[i][..., 5:], tf.float32)
        # (n,13,13,3,1)
        noobj_mask = 1 - object_mask
        # raw_pred用来计算loss，pred_xy,pred_wh计算iou
        grid, raw_pred, pred_xy, pred_wh = yolo_decode(yolo_outputs[i], anchors[anchor_mask[i]], num_classes,
                                                       input_shape, cal_loss=True)
        # 真实
        # (n,13,13,3,4)
        pred_boxes = tf.concat([pred_xy, pred_wh], axis=-1)
        object_mask_boolean = tf.cast(object_mask, tf.bool)

        ignore_mask = tf.TensorArray(tf.float32, 1, dynamic_size=True)

        # 遍历每张图片计算ignore_mask
        for b in range(batch):
            # (n,4)
            true_boxes = tf.cast(tf.boolean_mask(boxes[b], object_mask_boolean[b, ..., 0]), tf.float32)
            # 计算IOU 计算ignore_mask
            # boxes(13,13,3,4) true_boxes(n,4)
            # (13,13,3,n)
            ious = iou(pred_boxes, true_boxes)
            # (13,13,3)
            ious_max = tf.reduce_max(ious, axis=-1)
            # 筛选iou小于阈值的bbox
            # (13,13,3)
            ignore_mask.write(b, tf.cast(ious_max < ignore_threshold, tf.float32))

        # (batch,13,13,3)
        ignore_mask = ignore_mask.stack()
        ignore_mask = tf.expand_dims(ignore_mask, axis=-1)

        # -------------------------- #
        #   将y_ture编码成raw_true
        #   grid:(1, h, w, 1, 2)
        # -------------------------- #
        true_xy = boxes[..., :2] * grid_shape[0][::-1] - grid
        # TODO 避免log0
        true_wh = tf.math.log(boxes[..., 2:] * input_shape[::-1] / anchors[anchor_mask[i]])
        true_wh = tf.keras.backend.switch(object_mask, true_wh, tf.keras.backend.zeros_like(true_wh))

        box_loss_scale = 2 - y_true[i][..., 2:3] * y_true[i][..., 3:4]

        # 计算loss
        # 中心点损失
        # print(true_xy.shape, raw_pred[..., 0:2].shape, object_mask.shape, box_loss_scale.shape)
        center_loss = coord_scale * object_mask * box_loss_scale *\
            tf.nn.sigmoid_cross_entropy_with_logits(true_xy, raw_pred[..., 0:2])
        # scale损失
        scale_loss = coord_scale * object_mask * \
            box_loss_scale * 0.5 * tf.square(true_wh - raw_pred[..., 2:4])
        # 置信度损失函数
        confidence_loss = object_mask * tf.nn.sigmoid_cross_entropy_with_logits(object_mask, raw_pred[..., 4:5]) \
            + noobj_scale * noobj_mask * ignore_mask * \
            tf.nn.sigmoid_cross_entropy_with_logits(object_mask, raw_pred[..., 4:5])
        # 类别损失函数
        class_loss = object_mask * tf.nn.sigmoid_cross_entropy_with_logits(classes, raw_pred[..., 5:])

        center_loss = tf.reduce_sum(center_loss) / batch_float
        scale_loss = tf.reduce_sum(scale_loss) / batch_float
        confidence_loss = tf.reduce_sum(confidence_loss) / batch_float
        class_loss = tf.reduce_sum(class_loss) / batch_float

        loss += center_loss + scale_loss + confidence_loss + class_loss

    return loss
