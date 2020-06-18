# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:util.py
# software: PyCharm

import numpy as np
import cv2
from PIL import Image
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb


# ---------------------------------------------------#
#   使用letter_box预处理图片，以防止图片失真
# ---------------------------------------------------#
def letter_box(img, target_size):
    h = img.shape[0]
    w = img.shape[1]

    th, tw = target_size
    scale = min(th / h, tw / w)
    h = int(h * scale)
    w = int(w * scale)
    # 填充0
    img = cv2.resize(img, (w, h))
    img = cv2.copyMakeBorder(img, (th - h) // 2, (th - h) // 2,
                             (tw - w) // 2, (tw - w) // 2, cv2.BORDER_CONSTANT, value=(128, 128, 128))
    return img


# ---------------------------------------------------#
#   非极大值抑制
#   params:
#       rects:(n,5)
# ---------------------------------------------------#
def nms(rects, threshold):
    rects = np.array(rects)
    if rects.size == 0:
        return rects
    # (n,)
    x1 = rects[:, 0]
    y1 = rects[:, 1]
    x2 = rects[:, 2]
    y2 = rects[:, 3]
    scores = rects[:, 4]
    # 排序
    # (n,)
    sort = np.argsort(scores)
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    # 进行筛选
    pick = []

    while len(sort) > 0:
        pick.append(sort[-1])
        # 计算IOU
        xx1 = np.maximum(x1[sort[0:-1]], x1[sort[-1]])
        yy1 = np.maximum(y1[sort[0:-1]], y1[sort[-1]])
        xx2 = np.minimum(x2[sort[0:-1]], x2[sort[-1]])
        yy2 = np.minimum(y2[sort[0:-1]], y2[sort[-1]])
        # 没有交叉区域
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)

        iou = (w * h) / (area[sort[0:-1]] + area[sort[-1]] - w * h)
        sort = sort[np.where(iou < threshold)]

    return pick


# ----------------------------------------------------------------#
#   yolo3的标签处理函数，将每个ground truth分配给一个cell和anchor box
#   1.进行iou计算
#   2.根据中心点找到responsible cell
#   params:
#       true_boxes:(m,n,5) dtype=list 经过数据增强预处理的ground truth
#       input_shape:输入形状
#       anchors:用来计算iou (9,2)
#       num_classes:
# ----------------------------------------------------------------#
def true_boxes_preprocess(true_boxes, input_shape, anchors, num_classes):

    true_boxes = np.array(true_boxes)

    assert ((true_boxes[..., 4] < num_classes).all()), '出现未知类别'

    num_layers = len(anchors) // 3
    # 先验框
    # 颗粒度细就用小尺寸先验框
    # 678为116,90,  156,198,  373,326
    # 345为30,61,  62,45,  59,119
    # 012为10,13,  16,30,  33,23
    anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_layers == 3 else [[3, 4, 5], [0, 1, 2]]
    true_boxes = np.array(true_boxes, dtype=np.float32)
    input_shape = np.array(input_shape, dtype=np.int32)

    # 中心点   长宽
    # (m,n,2)
    boxes_xy = (true_boxes[..., :2] + true_boxes[..., 2:4]) // 2
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., :2]
    # 归一化后用于计算responsible cell
    true_boxes[..., :2] = boxes_xy / input_shape
    true_boxes[..., 2:4] = boxes_wh / input_shape
    # (m,n,2)
    boxes_x_muti_y = boxes_wh[..., 0] * boxes_wh[..., 1]
    # (m,n)
    valid_mask = boxes_x_muti_y > 0

    batch = true_boxes.shape[0]
    # (num_layers,2)
    grids = [input_shape // {0: 32, 1: 16, 2: 8}[i] for i in range(num_layers)]

    anchors = np.expand_dims(anchors, 0)

    y_true = [np.zeros(shape=(batch, grids[k][0], grids[k][1], len(anchors_mask[k]), 5 + num_classes))
              for k in range(num_layers)]

    for m in range(batch):
        # 计算iou
        mask = valid_mask[m]
        # (n1,2)
        wh = boxes_wh[m][mask]
        # (n1, 1, 2)
        wh = np.expand_dims(wh, axis=-2)
        # (n1,4)
        xywh_class = true_boxes[m][mask]
        if len(wh) == 0:
            continue
        # (n1, 9, 2)
        inter_wh = np.maximum(np.minimum(wh / 2, anchors / 2) - np.maximum(-wh / 2, -anchors / 2), 0)
        # (n1, 9)
        inter_area = inter_wh[..., 0] * inter_wh[..., 1]
        box_area = wh[..., 0] * wh[..., 1]
        anchor_area = anchors[..., 0] * anchors[..., 1]
        # (n1,9)
        iou = inter_area / (box_area + anchor_area - inter_area)
        # (n1,)
        iou_max = np.argmax(iou, axis=-1)

        for t, anchor_index in enumerate(iou_max):
            for p in range(num_layers):
                if anchor_index in anchors_mask[p]:
                    # 分配cell
                    # x
                    j = np.floor(xywh_class[t, 0] * grids[p][1]).astype(np.int32)
                    i = np.floor(xywh_class[t, 1] * grids[p][0]).astype(np.int32)
                    index = anchors_mask[p].index(anchor_index)

                    categorical = xywh_class[t, 4].astype(np.int32)

                    y_true[p][m, i, j, index, 0:4] = xywh_class[t, 0:4]
                    y_true[p][m, i, j, index, 4] = 1
                    y_true[p][m, i, j, index, 5 + categorical] = 1

    return y_true


def rand(a, b):
    return np.random.rand() * (b - a) + a


"""
    随机数据增强：
        1.resize
        2.水平翻转
        3.光学扭曲
"""


def get_random_data(annotation_line, input_shape, random=True, max_boxes=100,
                    jitter=0.3, hue=0.1, sat=1.5, val=1.5):

    # [img_name, box1, box2, ...]
    lines = annotation_line.split()
    image = Image.open(lines[0])
    iw, ih = image.size
    h, w = input_shape
    # 将box转化为nparray
    # (n, 4)
    boxes = np.array([np.array(list(map(int, box.split(',')))) for box in lines[1:]])

    # 不进行数据增强
    if not random:
        scale = min(h / ih, w / iw)
        nw = int(iw * scale)
        nh = int(ih * scale)
        dx = (w - nw) // 2
        dy = (h - nh) // 2
        # letter box
        image = image.resize((nw, nh), Image.BICUBIC)
        new_img = Image.new('RGB', (w, h), (128, 128, 128))
        new_img.paste(image, (dx, dy))
        # 归一化
        img_data = np.array(np.array(new_img) / 255, dtype='float32')
        # 校正box
        # (x1, y1, x2, y2)
        # 设置最多选择max_boxes个box
        box_data = np.zeros((max_boxes, 5))
        if len(boxes) > 0:
            # shuffle
            np.random.shuffle(boxes)
            if len(boxes) > max_boxes:
                boxes = boxes[:max_boxes]
            boxes[..., [0, 2]] = boxes[..., [0, 2]] * scale + dx
            boxes[..., [1, 3]] = boxes[..., [1, 3]] * scale + dy
            box_data[0:len(boxes)] = boxes
        return img_data, box_data

    # 进行数据增强
    # resize
    aspect_ratio = w / h * rand(1 - jitter, 1 + jitter) / rand(1 - jitter, 1 + jitter)
    scale = rand(0.25, 2)
    resize_w = int(h * scale)
    resize_h = int(resize_w / aspect_ratio)
    image = image.resize((resize_w, resize_h), Image.BICUBIC)
    # place img
    # 具有剪切效果
    # place image
    dx = int(rand(0, w - resize_w))
    dy = int(rand(0, h - resize_h))
    new_image = Image.new('RGB', (w, h), (128, 128, 128))
    new_image.paste(image, (dx, dy))
    image = new_image

    # 翻转
    filp = rand(0, 1) < 0.5
    if filp:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)

    # 光学扭曲
    hue = rand(-hue, hue)
    sat = rand(1, sat) if rand(0, 1) < .5 else 1 / rand(1, sat)
    val = rand(1, val) if rand(0, 1) < .5 else 1 / rand(1, val)
    x = rgb_to_hsv(np.array(image) / 255.)
    x[..., 0] += hue
    x[..., 0][x[..., 0] > 1] -= 1
    x[..., 0][x[..., 0] < 0] += 1
    x[..., 1] *= sat
    x[..., 2] *= val
    x[x > 1] = 1
    x[x < 0] = 0
    image_data = hsv_to_rgb(x)  # numpy array, 0 to 1

    # 校正box
    box_data = np.zeros((max_boxes, 5))
    if len(boxes) > 0:
        np.random.shuffle(boxes)
        if len(boxes) > max_boxes:
            boxes = boxes[:max_boxes]
        boxes[..., [0, 2]] = boxes[..., [0, 2]] * resize_w / iw + dx
        boxes[..., [1, 3]] = boxes[..., [1, 3]] * resize_h / ih + dy
        if filp:
            # ---------------------------------------- #
            """这里要注意 flip之后校正 减去的是[2, 0]"""
            # ---------------------------------------- #
            boxes[..., [0, 2]] = w - boxes[..., [2, 0]]
        # 对超出图片范围的box进行校正
        # 对长宽小于1的box进行筛选
        boxes[:, 0:2][boxes[:, 0:2] < 0] = 0
        boxes[:, 2][boxes[:, 2] > w] = w
        boxes[:, 3][boxes[:, 3] > h] = h
        box_w = boxes[:, 2] - boxes[:, 0]
        box_h = boxes[:, 3] - boxes[:, 1]
        boxes = boxes[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box
        box_data[:len(boxes)] = boxes

    return image_data, box_data


def letterbox_image(image, size):
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    return new_image


def correct_boxes(box_xy, box_wh, input_shape, image_shape):
    """

    Args:
        box_xy: box_xy which model predicts in input coordinate
        box_wh: box_wh which model predicts in input coordinate
        input_shape: (416, 416)
        image_shape: (h, w)

    Returns:

    """
    h, w = image_shape
    input_h, input_w = input_shape
    scale = min(input_h / h, input_w / w)
    new_h = h * scale
    new_w = w * scale

    dx = (input_w - new_w) / 2
    dy = (input_h - new_h) / 2
    box_xy[..., 0] = box_xy[..., 0] - dx
    box_xy[..., 1] = box_xy[..., 1] - dy
    box_xy = box_xy / scale
    box_wh = box_wh / scale

    box_min = box_xy - box_wh / 2
    box_max = box_xy + box_wh / 2

    boxes = np.concatenate([box_min, box_max], axis=-1)

    return boxes
