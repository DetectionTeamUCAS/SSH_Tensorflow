# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, division

import tensorflow as tf
import random
import os
import sys
sys.path.append("../")

from libs.configs import cfgs


def make_anchors(base_anchor_size, anchor_scales, anchor_ratios,
                 featuremap_height, featuremap_width,
                 stride, name='make_anchors'):
    '''
    :param base_anchor_size:256
    :param anchor_scales:
    :param anchor_ratios:
    :param featuremap_height:
    :param featuremap_width:
    :param stride:
    :return:
    '''
    with tf.variable_scope(name):
        base_anchor = tf.constant([0, 0, base_anchor_size, base_anchor_size], tf.float32)  # [x_center, y_center, w, h]

        ws, hs = enum_ratios(enum_scales(base_anchor, anchor_scales),
                             anchor_ratios)  # per locations ws and hs

        x_centers = tf.range(featuremap_width, dtype=tf.float32) * stride
        y_centers = tf.range(featuremap_height, dtype=tf.float32) * stride

        x_centers, y_centers = tf.meshgrid(x_centers, y_centers)

        ws, x_centers = tf.meshgrid(ws, x_centers)
        hs, y_centers = tf.meshgrid(hs, y_centers)

        anchor_centers = tf.stack([x_centers, y_centers], 2)
        anchor_centers = tf.reshape(anchor_centers, [-1, 2])

        box_sizes = tf.stack([ws, hs], axis=2)
        box_sizes = tf.reshape(box_sizes, [-1, 2])
        # anchors = tf.concat([anchor_centers, box_sizes], axis=1)
        anchors = tf.concat([anchor_centers - 0.5*box_sizes,
                             anchor_centers + 0.5*box_sizes], axis=1)
        return anchors


def enum_scales(base_anchor, anchor_scales):

    anchor_scales = base_anchor * tf.constant(anchor_scales, dtype=tf.float32, shape=(len(anchor_scales), 1))

    return anchor_scales


def enum_ratios(anchors, anchor_ratios):
    '''
    ratio = h /w
    :param anchors:
    :param anchor_ratios:
    :return:
    '''
    ws = anchors[:, 2]  # for base anchor: w == h
    hs = anchors[:, 3]
    sqrt_ratios = tf.sqrt(tf.constant(anchor_ratios))

    ws = tf.reshape(ws / sqrt_ratios[:, tf.newaxis], [-1, 1])
    hs = tf.reshape(hs * sqrt_ratios[:, tf.newaxis], [-1, 1])

    return hs, ws


def shift_anchor(anchors, stride):
    shift_delta = [(stride // 2, 0), (0, stride // 2), (stride // 2, stride // 2)]
    anchor_merge = []
    anchor_shape = anchors.get_shape().as_list()
    for d in shift_delta:
        shift_x = tf.ones([anchor_shape[0], ]) * d[0]
        shift_y = tf.ones([anchor_shape[0], ]) * d[1]
        shift = tf.stack([shift_x, shift_y], axis=1)
        shift = tf.tile(shift, [1, 2])
        anchors_shift = anchors + shift
        anchor_merge.append(anchors_shift)
    return anchor_merge


def shift_jitter(anchors, stride):
    shift_delta = []
    anchor_merge = []
    anchor_shape = anchors.get_shape().as_list()
    for _ in range(4):
        shift_delta.append((round(random.uniform(0, stride // 2 - 1)),
                            round(random.uniform(0, stride // 2 - 1))))
    for d in shift_delta:
        shift_x = tf.ones([anchor_shape[0], ]) * d[0]
        shift_y = tf.ones([anchor_shape[0], ]) * d[1]
        shift = tf.stack([shift_x, shift_y], axis=1)
        shift = tf.tile(shift, [1, 2])
        anchors_shift = anchors + shift
        anchor_merge.append(anchors_shift)
    return anchor_merge


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    anchors = make_anchors(base_anchor_size=cfgs.BASE_ANCHOR_SIZE_LIST[0],
                           anchor_scales=[1.0],
                           anchor_ratios=cfgs.ANCHOR_RATIOS,
                           featuremap_height=5,
                           featuremap_width=5,
                           stride=[8],
                           name="make_anchors_for_m1")

    anchor_merge_1 = shift_jitter(anchors, 8)
    anchor_merge_2 = shift_anchor(anchors, 8)
    anchor_merge = [anchors] + [] + anchor_merge_2
    tmp = tf.reshape(tf.stack(anchor_merge, axis=1), [-1, 4])

    with tf.Session() as sess:
        anchors_, anchor_merge_ = sess.run([anchors, tmp])
        print(anchors_)
        print(anchors_.shape)
        # print('**'*20)
        # print(anchor_merge_)


