# -*- coding:utf-8 -*-
# author:平手友梨奈ii
# e-mail:1353593259@qq.com
# datetime:1993/12/01
# filename:train.py
# software: PyCharm

import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import argparse
from utils.util import get_random_data
from utils.util import true_boxes_preprocess
from net.dark53 import yolo3_body
from net.yolo3_loss import yolo3_loss


def arguments():
    """
        调试超参：
            1.lr and warm up lr
            2.batch_size
            3.warm_batch_size
            4.epoch
            5.没有使用学习率下降
            6.adam优化器参数采取默认
            7.没有使用早停
        :return: args
    """
    # TODO 在训练中加入ReduceLROnPlateau 和 EarlyStopping
    parser = argparse.ArgumentParser(description='Train YOLO networks with random input shape.')
    parser.add_argument('--input_shape', default=(416, 416),
                        help="Input data shape for evaluation, use 320, 416, 608... ")
    parser.add_argument('--batch_size', default=4, type=int, help='Training mini-batch size')
    parser.add_argument('--warm_batch_size', default=32, type=int, help='Warm up mini-batch size')
    parser.add_argument('--epoches', type=int, default=50, help='Training epochs.')
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--warm_learning_rate', type=float, default=0.0001)
    parser.add_argument('--annotations_path', default='./voc2020/ImageSets/Main/train.txt')
    parser.add_argument('--anchors_path', default='./yolo_data/yolo_anchors.txt')
    parser.add_argument('--classes_path', default='./yolo_data/my_classes.txt')
    parser.add_argument('--pretrained_path', default='./weights/yolo_weights.h5')
    parser.add_argument('--ckpt_path', default='./weights/ckpt/')

    args = parser.parse_args()
    return args


def get_anchors(filepath):
    with open(filepath, 'r') as f:
        lines = f.readline()
    anchors = [float(x) for x in lines.split(',')]
    anchors = np.array(anchors).reshape(-1, 2)
    return anchors


def get_classes(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    classes = [cla.strip() for cla in lines]
    return classes


def data_generator(annotations, batch, input_shape, anchors, num_classes, random=True):
    """
        data generator
        读取train.txt
        生成数据 格式为(batch, 13, 13, 3, 25) (batch, 26, 26, 3, 25) (batch, 52, 52, 3, 25)
        true_boxes_preprocess
        读取lines 有点 不需要把图片数据全部读进内存
        -------------
        return:
            [input_img, y_true1, y_true2, y_true3]
    """
    num_data = len(annotations)
    # 计数器
    i = 0

    while True:
        input_img = []
        boxes = []
        for b in range(batch):
            if i == 0:
                np.random.shuffle(annotations)
            # img_data：模型的输入
            img_data, box_data = get_random_data(annotations[i], input_shape, max_boxes=100, random=random)
            input_img.append(img_data)
            boxes.append(box_data)
            i = (i + 1) % num_data
        # boxes: (batch, max_boxes, 5)
        input_img = np.array(input_img)
        boxes = np.array(boxes)
        y_true = true_boxes_preprocess(boxes, input_shape, anchors, num_classes)

        yield [input_img, *y_true]


def get_model(inputs_shape, anchors, num_classes, weights_path, pretrained=True, freezed=True):
    """
        建立yolo
        freezed=True时：
            冻结除了3个输出层之外的所有层数
    """
    inputs = keras.Input(shape=inputs_shape)
    num_anchors = len(anchors)
    yolo3_model = yolo3_body(inputs, num_anchors // 3, num_classes)
    print('create YOLOv4 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    # 是否载入预训练的权重
    if pretrained:
        yolo3_model.load_weights(weights_path)
        print('Load weights {}.'.format(weights_path))
    # 是否冻结除了最后三层之外的所有层
    if freezed:
        num_layers_freeze = 249
        for i in range(num_layers_freeze):
            yolo3_model.layers[i].trainable = False
        print('Freeze the first {} layers of total {} layers.'.format(num_layers_freeze, len(yolo3_model.layers)))
    return yolo3_model


@tf.function
def training(inputs, model, optimizer, y_trues, anchors, num_classes, ignore_threshold, coord_scale, noobj_scale):
    # 构建yolo3
    true1, true2, true3 = y_trues
    with tf.GradientTape() as tape:
        feature1, feature2, feature3 = model(inputs)
        args = tf.stack([feature1, feature2, feature3, true1, true2, true3], axis=0)
        losses = yolo3_loss(args, anchors, num_classes, ignore_threshold, coord_scale, noobj_scale)
    gradients = tape.gradient(losses, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return losses


def _main():
    """
            开始训练
        """
    args_ = arguments()
    # batch epoch
    BATCH_SIZE = args_.batch_size
    WARM_BATCH = args_.warm_batch_size
    EPOCHES = args_.epoches
    # lr
    lr = args_.learning_rate
    warm_lr = args_.warm_learning_rate
    inputs_shape_ = args_.input_shape
    image_shape = (None, None, 3)
    # file path
    annotation_file = args_.annotations_path
    anchor_file = args_.anchors_path
    classes_file = args_.classes_path
    weights_path = args_.pretrained_path
    # 得到anchors和classes
    anchors = get_anchors(anchor_file)
    classes = get_classes(classes_file)
    num_classes = len(classes)
    # 读取annotations
    # 计算steps per epoch
    with open(annotation_file, 'r') as f:
        annotation_lines = f.readlines()
        annotations = [line.strip() for line in annotation_lines]
    num_dataset = len(annotations)
    steps_per_epoch = num_dataset // BATCH_SIZE
    steps_per_epoch_warm = num_dataset // WARM_BATCH

    # optimizer
    # 默认使用adam论文中的建议参数
    # TODO 如果需要改变参数，修改这段代码
    optimizer = keras.optimizers.Adam(learning_rate=lr)
    optimizer_2 = keras.optimizers.Adam(learning_rate=warm_lr)

    # -------------------------- #
    #   热身 warm up
    #   freezed=True
    # -------------------------- #
    yolo_model = get_model(image_shape, anchors, num_classes, weights_path, pretrained=True, freezed=True)

    # 检查点 最多保存5个权重文件
    ckpt = tf.train.Checkpoint(yolo_model=yolo_model,
                               optimizer=optimizer,
                               optimizer_2=optimizer_2)
    ckpt_manager = tf.train.CheckpointManager(ckpt, args_.ckpt_path, max_to_keep=5, keep_checkpoint_every_n_hours=1)
    # if ckpt_manager.latest_checkpoint:
    #     ckpt.restore(ckpt_manager.latest_checkpoint)
    # print('恢复模型成功！')

    # 构建data generator
    data_gene1 = data_generator(annotations, WARM_BATCH, inputs_shape_, anchors, num_classes)

    for epoch in range(EPOCHES):
        # TODO 在第n个epoch的时候，下降学习率
        total_loss = 0
        for step in range(steps_per_epoch_warm):
            inputs_img, true1, true2, true3 = next(data_gene1)
            trues = [true1, true2, true3]
            losses = training(inputs_img, yolo_model, optimizer, trues, anchors, num_classes,
                              ignore_threshold=0.5, coord_scale=1, noobj_scale=1)
            total_loss += losses

        ckpt_manager.save()

        print('Epoch:{}----total losses:{}'.format(epoch, total_loss))
    yolo_model.save_weights('./weights/trained_weights_warm.h5')

    # ---------------------------------------- #
    #   fine tune, 如果warm up结果不好，就继续训练
    #   freezed=False
    # ---------------------------------------- #
    print('Start fine tune.')
    print('Unfreeze all of the layers.')

    for i in range(len(yolo_model.layers)):
        yolo_model.layers[i].trainable = True

    # 构建data generator
    data_gene2 = data_generator(annotations, BATCH_SIZE, inputs_shape_, anchors, num_classes)

    for epoch in range(EPOCHES):
        # TODO 在第n个epoch的时候，下降学习率
        total_loss = 0
        for step in range(steps_per_epoch):
            inputs_img, true1, true2, true3 = next(data_gene2)
            trues = [true1, true2, true3]
            losses = training(inputs_img, yolo_model, optimizer_2, trues, anchors, num_classes,
                              ignore_threshold=0.5, coord_scale=1, noobj_scale=1)
            total_loss += losses

        ckpt_manager.save()

        print('Epoch:{}----total losses:{}'.format(epoch, total_loss))
    yolo_model.save_weights('./weights/trained_weights_final.h5')

    print('training over!')


if __name__ == '__main__':
    print('开始训练！')
