# -*-coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

from libs.networks import resnet
from libs.networks import mobilenet_v2
from libs.box_utils import encode_and_decode
from libs.box_utils import boxes_utils
from libs.box_utils import anchor_utils
from libs.configs import cfgs
from libs.losses import losses
from libs.box_utils import show_box_in_tensor
from libs.detection_oprations.proposal_target_layer import proposal_target_layer


class DetectionNetwork(object):

    def __init__(self, base_network_name, is_training):

        self.base_network_name = base_network_name
        self.is_training = is_training
        self.m1_num_anchors_per_location = len(cfgs.M1_ANCHOR_SCALES) * len(cfgs.ANCHOR_RATIOS)
        self.m2_num_anchors_per_location = len(cfgs.M2_ANCHOR_SCALES) * len(cfgs.ANCHOR_RATIOS)
        self.m3_num_anchors_per_location = len(cfgs.M3_ANCHOR_SCALES) * len(cfgs.ANCHOR_RATIOS)

    def build_base_network(self, input_img_batch):

        if self.base_network_name.startswith('resnet_v1'):
            return resnet.resnet_base(input_img_batch, scope_name=self.base_network_name, is_training=self.is_training)

        elif self.base_network_name.startswith('MobilenetV2'):
            return mobilenet_v2.mobilenetv2_base(input_img_batch, is_training=self.is_training)

        else:
            raise ValueError('Sry, we only support resnet or mobilenet_v2')

    def postprocess_ssh(self, rois, bbox_ppred, scores, img_shape, iou_threshold):
        '''

        :param rois:[-1, 4]
        :param bbox_ppred: [-1, (cfgs.Class_num+1) * 4]
        :param scores: [-1, cfgs.Class_num + 1]
        :return:
        '''

        with tf.name_scope('postprocess_fastrcnn'):

            if self.is_training:
                pre_nms_topN = cfgs.TOP_K_NMS_TRAIN
                post_nms_topN = cfgs.MAXIMUM_PROPOSAL_TARIN
            else:
                pre_nms_topN = cfgs.TOP_K_NMS_TEST
                post_nms_topN = cfgs.MAXIMUM_PROPOSAL_TEST

            rois = tf.stop_gradient(rois)
            scores = tf.stop_gradient(scores)
            bbox_ppred = tf.reshape(bbox_ppred, [-1, cfgs.CLASS_NUM + 1, 4])
            bbox_ppred = tf.stop_gradient(bbox_ppred)

            bbox_pred_list = tf.unstack(bbox_ppred, axis=1)
            score_list = tf.unstack(scores, axis=1)

            allclasses_boxes = []
            allclasses_scores = []
            categories = []
            for i in range(1, cfgs.CLASS_NUM + 1):

                # 1. decode boxes in each class
                tmp_encoded_box = bbox_pred_list[i]
                tmp_score = score_list[i]
                tmp_decoded_boxes = encode_and_decode.decode_boxes(encoded_boxes=tmp_encoded_box,
                                                                   reference_boxes=rois,
                                                                   scale_factors=cfgs.ROI_SCALE_FACTORS)
                # tmp_decoded_boxes = encode_and_decode.decode_boxes(boxes=rois,
                #                                                    deltas=tmp_encoded_box,
                #                                                    scale_factor=cfgs.ROI_SCALE_FACTORS)

                # 2. clip to img boundaries
                tmp_decoded_boxes = boxes_utils.clip_boxes_to_img_boundaries(decode_boxes=tmp_decoded_boxes,
                                                                             img_shape=img_shape)

                if pre_nms_topN > 0:
                    pre_nms_topN = tf.minimum(pre_nms_topN, tf.shape(tmp_decoded_boxes)[0])
                    tmp_score, top_k_indices = tf.nn.top_k(tmp_score, k=pre_nms_topN)
                    tmp_decoded_boxes = tf.gather(tmp_decoded_boxes, top_k_indices)

                # 3. NMS
                keep = tf.image.non_max_suppression(boxes=tmp_decoded_boxes,
                                                    scores=tmp_score,
                                                    max_output_size=cfgs.SSH_NMS_MAX_BOXES_PER_CLASS,
                                                    iou_threshold=iou_threshold)

                perclass_boxes = tf.gather(tmp_decoded_boxes, keep)
                perclass_scores = tf.gather(tmp_score, keep)

                allclasses_boxes.append(perclass_boxes)
                allclasses_scores.append(perclass_scores)
                categories.append(tf.ones_like(perclass_scores) * i)

            final_boxes = tf.concat(allclasses_boxes, axis=0)
            final_scores = tf.concat(allclasses_scores, axis=0)
            final_category = tf.concat(categories, axis=0)

            if self.is_training:
                '''
                in training. We should show the detecitons in the tensorboard. So we add this.
                '''
                kept_indices = tf.reshape(tf.where(tf.greater_equal(final_scores, cfgs.SHOW_SCORE_THRSHOLD)), [-1])

                final_boxes = tf.gather(final_boxes, kept_indices)
                final_scores = tf.gather(final_scores, kept_indices)
                final_category = tf.gather(final_category, kept_indices)

        return final_boxes, final_scores, final_category

    def context_module(self, feature_maps):
        with tf.variable_scope('context_module',
                               regularizer=slim.l2_regularizer(cfgs.WEIGHT_DECAY)):
            channels = feature_maps.get_shape().as_list()[-1]
            conv3x3_1 = slim.conv2d(inputs=feature_maps,
                                    num_outputs=channels // 2,
                                    kernel_size=[3, 3],
                                    trainable=self.is_training,
                                    weights_initializer=cfgs.INITIALIZER,
                                    activation_fn=tf.nn.relu,
                                    scope='conv3x3_1')
            conv3x3_2 = slim.conv2d(inputs=conv3x3_1,
                                    num_outputs=channels // 2,
                                    kernel_size=[3, 3],
                                    trainable=self.is_training,
                                    weights_initializer=cfgs.INITIALIZER,
                                    activation_fn=tf.nn.relu,
                                    scope='conv3x3_2')
            conv3x3_3 = slim.conv2d(inputs=conv3x3_1,
                                    num_outputs=channels // 2,
                                    kernel_size=[3, 3],
                                    trainable=self.is_training,
                                    weights_initializer=cfgs.INITIALIZER,
                                    activation_fn=tf.nn.relu,
                                    scope='conv3x3_3')
            conv3x3_4 = slim.conv2d(inputs=conv3x3_3,
                                    num_outputs=channels // 2,
                                    kernel_size=[3, 3],
                                    trainable=self.is_training,
                                    weights_initializer=cfgs.INITIALIZER,
                                    activation_fn=tf.nn.relu,
                                    scope='conv3x3_4')
            output = tf.concat([conv3x3_2, conv3x3_4], axis=3, name='concat')

            return output

    def detection_module(self, feature_maps, num_anchors_per_location, scope):
        with tf.variable_scope(scope,
                               regularizer=slim.l2_regularizer(cfgs.WEIGHT_DECAY)):
            channels = feature_maps.get_shape().as_list()[-1]
            context = self.context_module(feature_maps)
            conv3x3 = slim.conv2d(inputs=feature_maps,
                                  num_outputs=channels,
                                  kernel_size=[3, 3],
                                  trainable=self.is_training,
                                  weights_initializer=cfgs.INITIALIZER,
                                  activation_fn=tf.nn.relu,
                                  scope='conv3x3')
            concat_layer = tf.concat([conv3x3, context], axis=3, name='concat')
            cls_score = slim.conv2d(inputs=concat_layer,
                                    num_outputs=num_anchors_per_location * (cfgs.CLASS_NUM + 1),
                                    kernel_size=[1, 1],
                                    trainable=self.is_training,
                                    weights_initializer=cfgs.INITIALIZER,
                                    activation_fn=None,
                                    scope='cls_score')
            box_pred = slim.conv2d(inputs=concat_layer,
                                   num_outputs=num_anchors_per_location * (cfgs.CLASS_NUM + 1) * 4,
                                   kernel_size=[1, 1],
                                   trainable=self.is_training,
                                   weights_initializer=cfgs.INITIALIZER,
                                   activation_fn=None,
                                   scope='box_pred')
            return cls_score, box_pred

    def add_roi_batch_img_smry(self, img, rois, labels, name):
        positive_roi_indices = tf.reshape(tf.where(tf.greater_equal(labels, 1)), [-1])

        negative_roi_indices = tf.reshape(tf.where(tf.equal(labels, 0)), [-1])

        pos_roi = tf.gather(rois, positive_roi_indices)
        neg_roi = tf.gather(rois, negative_roi_indices)

        pos_in_img = show_box_in_tensor.only_draw_boxes(img_batch=img,
                                                        boxes=pos_roi)
        neg_in_img = show_box_in_tensor.only_draw_boxes(img_batch=img,
                                                        boxes=neg_roi)
        tf.summary.image('{}/pos_rois'.format(name), pos_in_img)
        tf.summary.image('{}/neg_rois'.format(name), neg_in_img)

    def build_whole_detection_network(self, input_img_batch, gtboxes_batch):

        if self.is_training:
            # ensure shape is [M, 5]
            gtboxes_batch = tf.reshape(gtboxes_batch, [-1, 5])
            gtboxes_batch = tf.cast(gtboxes_batch, tf.float32)

        img_shape = tf.shape(input_img_batch)

        # 1. build base network
        feature_stride8, feature_stride16 = self.build_base_network(input_img_batch)

        # feature_stride8 = tf.image.resize_bilinear(feature_stride8, [tf.shape(feature_stride8)[1] * 2,
        #                                                              tf.shape(feature_stride8)[2] * 2],
        #                                            name='upsampling_stride8')

        # 2. build rpn
        with tf.variable_scope('build_ssh',
                               regularizer=slim.l2_regularizer(cfgs.WEIGHT_DECAY)):

            ssh_max_pool = slim.max_pool2d(inputs=feature_stride16,
                                           kernel_size=[2, 2],
                                           scope='ssh_max_pool')

            cls_score_m3, box_pred_m3 = self.detection_module(ssh_max_pool,
                                                              self.m3_num_anchors_per_location,
                                                              'detection_module_m3')
            box_pred_m3 = tf.reshape(box_pred_m3, [-1, 4 * (cfgs.CLASS_NUM + 1)])
            cls_score_m3 = tf.reshape(cls_score_m3, [-1, (cfgs.CLASS_NUM + 1)])
            cls_prob_m3 = slim.softmax(cls_score_m3, scope='cls_prob_m3')

            cls_score_m2, box_pred_m2 = self.detection_module(feature_stride16,
                                                              self.m2_num_anchors_per_location,
                                                              'detection_module_m2')
            box_pred_m2 = tf.reshape(box_pred_m2, [-1, 4*(cfgs.CLASS_NUM + 1)])
            cls_score_m2 = tf.reshape(cls_score_m2, [-1, (cfgs.CLASS_NUM + 1)])
            cls_prob_m2 = slim.softmax(cls_score_m2, scope='cls_prob_m2')

            channels_16 = feature_stride16.get_shape().as_list()[-1]
            channels_8 = feature_stride8.get_shape().as_list()[-1]
            feature8_shape = tf.shape(feature_stride8)
            conv1x1_1 = slim.conv2d(inputs=feature_stride16,
                                    num_outputs=channels_16 // 4,
                                    kernel_size=[1, 1],
                                    trainable=self.is_training,
                                    weights_initializer=cfgs.INITIALIZER,
                                    activation_fn=tf.nn.relu,
                                    scope='conv1x1_1')
            upsampling = tf.image.resize_bilinear(conv1x1_1, [feature8_shape[1], feature8_shape[2]],
                                                  name='upsampling')

            conv1x1_2 = slim.conv2d(inputs=feature_stride8,
                                    num_outputs=channels_8 // 2,
                                    kernel_size=[1, 1],
                                    trainable=self.is_training,
                                    weights_initializer=cfgs.INITIALIZER,
                                    activation_fn=tf.nn.relu,
                                    scope='conv1x1_2')

            eltwise_sum = upsampling + conv1x1_2

            conv3x3 = slim.conv2d(inputs=eltwise_sum,
                                  num_outputs=channels_8 // 2,
                                  kernel_size=[3, 3],
                                  trainable=self.is_training,
                                  weights_initializer=cfgs.INITIALIZER,
                                  activation_fn=tf.nn.relu,
                                  scope='conv3x3')

            cls_score_m1, box_pred_m1 = self.detection_module(conv3x3,
                                                              self.m1_num_anchors_per_location,
                                                              'detection_module_m1')
            box_pred_m1 = tf.reshape(box_pred_m1, [-1, 4*(cfgs.CLASS_NUM + 1)])
            cls_score_m1 = tf.reshape(cls_score_m1, [-1, (cfgs.CLASS_NUM + 1)])
            cls_prob_m1 = slim.softmax(cls_score_m1, scope='cls_prob_m1')

        # 3. generate_anchors
        featuremap_height_m1, featuremap_width_m1 = tf.shape(feature_stride8)[1], \
                                                    tf.shape(feature_stride8)[2]
        featuremap_height_m1 = tf.cast(featuremap_height_m1, tf.float32)
        featuremap_width_m1 = tf.cast(featuremap_width_m1, tf.float32)

        anchors_m1 = anchor_utils.make_anchors(base_anchor_size=cfgs.BASE_ANCHOR_SIZE_LIST[0],
                                               anchor_scales=cfgs.M1_ANCHOR_SCALES,
                                               anchor_ratios=cfgs.ANCHOR_RATIOS,
                                               featuremap_height=featuremap_height_m1,
                                               featuremap_width=featuremap_width_m1,
                                               stride=[cfgs.ANCHOR_STRIDE[0]],
                                               name="make_anchors_for_m1")

        featuremap_height_m2, featuremap_width_m2 = tf.shape(feature_stride16)[1], \
                                                    tf.shape(feature_stride16)[2]
        featuremap_height_m2 = tf.cast(featuremap_height_m2, tf.float32)
        featuremap_width_m2 = tf.cast(featuremap_width_m1, tf.float32)

        anchors_m2 = anchor_utils.make_anchors(base_anchor_size=cfgs.BASE_ANCHOR_SIZE_LIST[0],
                                               anchor_scales=cfgs.M2_ANCHOR_SCALES,
                                               anchor_ratios=cfgs.ANCHOR_RATIOS,
                                               featuremap_height=featuremap_height_m2,
                                               featuremap_width=featuremap_width_m2,
                                               stride=[cfgs.ANCHOR_STRIDE[1]],
                                               name="make_anchors_for_m2")

        featuremap_height_m3, featuremap_width_m3 = tf.shape(ssh_max_pool)[1], \
                                                    tf.shape(ssh_max_pool)[2]
        featuremap_height_m3 = tf.cast(featuremap_height_m3, tf.float32)
        featuremap_width_m3 = tf.cast(featuremap_width_m3, tf.float32)

        anchors_m3 = anchor_utils.make_anchors(base_anchor_size=cfgs.BASE_ANCHOR_SIZE_LIST[0],
                                               anchor_scales=cfgs.M3_ANCHOR_SCALES,
                                               anchor_ratios=cfgs.ANCHOR_RATIOS,
                                               featuremap_height=featuremap_height_m3,
                                               featuremap_width=featuremap_width_m3,
                                               stride=[cfgs.ANCHOR_STRIDE[2]],
                                               name="make_anchors_for_m3")
        # refer to paper: Seeing Small Faces from Robust Anchor’s Perspective
        if cfgs.EXTRA_SHIFTED_ANCHOR:
            shift_anchors_m1 = anchor_utils.shift_anchor(anchors_m1, cfgs.ANCHOR_STRIDE[0])
            shift_anchors_m2 = anchor_utils.shift_anchor(anchors_m2, cfgs.ANCHOR_STRIDE[1])
            shift_anchors_m3 = anchor_utils.shift_anchor(anchors_m3, cfgs.ANCHOR_STRIDE[2])
        else:
            shift_anchors_m1, shift_anchors_m2, shift_anchors_m3 = [], [], []

        if cfgs.FACE_SHIFT_JITTER:
            jitter_anchors_m1 = anchor_utils.shift_jitter(anchors_m1, cfgs.ANCHOR_STRIDE[0])
            jitter_anchors_m2 = anchor_utils.shift_jitter(anchors_m2, cfgs.ANCHOR_STRIDE[1])
            jitter_anchors_m3 = anchor_utils.shift_jitter(anchors_m3, cfgs.ANCHOR_STRIDE[2])
        else:
            jitter_anchors_m1, jitter_anchors_m2, jitter_anchors_m3 = [], [], []

        anchors_m1 = [anchors_m1] + shift_anchors_m1 + jitter_anchors_m1
        anchors_m1 = tf.reshape(tf.stack(anchors_m1, axis=1), [-1, 4])

        anchors_m2 = [anchors_m2] + shift_anchors_m2 + jitter_anchors_m2
        anchors_m2 = tf.reshape(tf.stack(anchors_m2, axis=1), [-1, 4])

        anchors_m3 = [anchors_m3] + shift_anchors_m3 + jitter_anchors_m3
        anchors_m3 = tf.reshape(tf.stack(anchors_m3, axis=1), [-1, 4])

        if self.is_training:
            with tf.variable_scope('sample_ssh_minibatch_m1'):
                rois_m1, labels_m1, bbox_targets_m1, keep_inds_m1 = \
                    tf.py_func(proposal_target_layer,
                               [anchors_m1, gtboxes_batch, 'M1'],
                               [tf.float32, tf.float32, tf.float32, tf.int32])
                rois_m1 = tf.reshape(rois_m1, [-1, 4])
                labels_m1 = tf.to_int32(labels_m1)
                labels_m1 = tf.reshape(labels_m1, [-1])
                bbox_targets_m1 = tf.reshape(bbox_targets_m1, [-1, 4 * (cfgs.CLASS_NUM + 1)])
                self.add_roi_batch_img_smry(input_img_batch, rois_m1, labels_m1, 'm1')

            with tf.variable_scope('sample_ssh_minibatch_m2'):
                rois_m2, labels_m2, bbox_targets_m2, keep_inds_m2 = \
                    tf.py_func(proposal_target_layer,
                               [anchors_m2, gtboxes_batch, 'M2'],
                               [tf.float32, tf.float32, tf.float32, tf.int32])
                rois_m2 = tf.reshape(rois_m2, [-1, 4])
                labels_m2 = tf.to_int32(labels_m2)
                labels_m2 = tf.reshape(labels_m2, [-1])
                bbox_targets_m2 = tf.reshape(bbox_targets_m2, [-1, 4 * (cfgs.CLASS_NUM + 1)])
                self.add_roi_batch_img_smry(input_img_batch, rois_m2, labels_m2, 'm2')

            with tf.variable_scope('sample_ssh_minibatch_m3'):
                rois_m3, labels_m3, bbox_targets_m3, keep_inds_m3 = \
                    tf.py_func(proposal_target_layer,
                               [anchors_m3, gtboxes_batch, 'M3'],
                               [tf.float32, tf.float32, tf.float32, tf.int32])
                rois_m3 = tf.reshape(rois_m3, [-1, 4])
                labels_m3 = tf.to_int32(labels_m3)
                labels_m3 = tf.reshape(labels_m3, [-1])
                bbox_targets_m3 = tf.reshape(bbox_targets_m3, [-1, 4 * (cfgs.CLASS_NUM + 1)])
                self.add_roi_batch_img_smry(input_img_batch, rois_m3, labels_m3, 'm3')

        if not self.is_training:
            with tf.variable_scope('postprocess_ssh_m1'):
                final_bbox_m1, final_scores_m1, final_category_m1 = self.postprocess_ssh(rois=anchors_m1,
                                                                                         bbox_ppred=box_pred_m1,
                                                                                         scores=cls_prob_m1,
                                                                                         img_shape=img_shape,
                                                                                         iou_threshold=cfgs.M1_NMS_IOU_THRESHOLD)

            with tf.variable_scope('postprocess_ssh_m2'):
                final_bbox_m2, final_scores_m2, final_category_m2 = self.postprocess_ssh(rois=anchors_m2,
                                                                                         bbox_ppred=box_pred_m2,
                                                                                         scores=cls_prob_m2,
                                                                                         img_shape=img_shape,
                                                                                         iou_threshold=cfgs.M2_NMS_IOU_THRESHOLD)

            with tf.variable_scope('postprocess_ssh_m3'):
                final_bbox_m3, final_scores_m3, final_category_m3 = self.postprocess_ssh(rois=anchors_m3,
                                                                                         bbox_ppred=box_pred_m3,
                                                                                         scores=cls_prob_m3,
                                                                                         img_shape=img_shape,
                                                                                         iou_threshold=cfgs.M3_NMS_IOU_THRESHOLD)

            result_dict = {'final_bbox_m1': final_bbox_m1, 'final_scores_m1': final_scores_m1,
                           'final_category_m1': final_category_m1, 'final_bbox_m2': final_bbox_m2,
                           'final_scores_m2': final_scores_m2, 'final_category_m2': final_category_m2,
                           'final_bbox_m3': final_bbox_m3, 'final_scores_m3': final_scores_m3,
                           'final_category_m3': final_category_m3}
            return result_dict

        else:
            with tf.variable_scope('ssh_loss_m1'):

                if not cfgs.M1_MINIBATCH_SIZE == -1:

                    box_pred_m1 = tf.gather(box_pred_m1, keep_inds_m1)
                    cls_score_m1 = tf.gather(cls_score_m1, keep_inds_m1)
                    cls_prob_m1 = tf.reshape(tf.gather(cls_prob_m1, keep_inds_m1), [-1, (cfgs.CLASS_NUM + 1)])

                    bbox_loss_m1 = losses.smooth_l1_loss_rcnn(bbox_pred=box_pred_m1,
                                                              bbox_targets=bbox_targets_m1,
                                                              label=labels_m1,
                                                              num_classes=cfgs.CLASS_NUM + 1,
                                                              sigma=cfgs.M1_SIGMA)

                    cls_loss_m1 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=cls_score_m1,
                                                                                                labels=labels_m1))

            with tf.variable_scope('postprocess_ssh_m1'):
                final_bbox_m1, final_scores_m1, final_category_m1 = self.postprocess_ssh(rois=rois_m1,
                                                                                         bbox_ppred=box_pred_m1,
                                                                                         scores=cls_prob_m1,
                                                                                         img_shape=img_shape,
                                                                                         iou_threshold=cfgs.M2_NMS_IOU_THRESHOLD)

            with tf.variable_scope('ssh_loss_m2'):
                if not cfgs.M2_MINIBATCH_SIZE == -1:

                    box_pred_m2 = tf.gather(box_pred_m2, keep_inds_m2)
                    cls_score_m2 = tf.gather(cls_score_m2, keep_inds_m2)
                    cls_prob_m2 = tf.reshape(tf.gather(cls_prob_m2, keep_inds_m2), [-1, (cfgs.CLASS_NUM + 1)])

                    bbox_loss_m2 = losses.smooth_l1_loss_rcnn(bbox_pred=box_pred_m2,
                                                              bbox_targets=bbox_targets_m2,
                                                              label=labels_m2,
                                                              num_classes=cfgs.CLASS_NUM + 1,
                                                              sigma=cfgs.M2_SIGMA)

                    cls_loss_m2 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=cls_score_m2,
                                                                                                labels=labels_m2))

            with tf.variable_scope('postprocess_ssh_m2'):
                final_bbox_m2, final_scores_m2, final_category_m2 = self.postprocess_ssh(rois=rois_m2,
                                                                                         bbox_ppred=box_pred_m2,
                                                                                         scores=cls_prob_m2,
                                                                                         img_shape=img_shape,
                                                                                         iou_threshold=cfgs.M2_NMS_IOU_THRESHOLD)

            with tf.variable_scope('ssh_loss_m3'):
                if not cfgs.M3_MINIBATCH_SIZE == -1:

                    box_pred_m3 = tf.gather(box_pred_m3, keep_inds_m3)
                    cls_score_m3 = tf.gather(cls_score_m3, keep_inds_m3)
                    cls_prob_m3 = tf.reshape(tf.gather(cls_prob_m3, keep_inds_m3), [-1, (cfgs.CLASS_NUM + 1)])

                    bbox_loss_m3 = losses.smooth_l1_loss_rcnn(bbox_pred=box_pred_m3,
                                                              bbox_targets=bbox_targets_m3,
                                                              label=labels_m3,
                                                              num_classes=cfgs.CLASS_NUM + 1,
                                                              sigma=cfgs.M3_SIGMA)

                    cls_loss_m3 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=cls_score_m3,
                                                                                                labels=labels_m3))

            with tf.variable_scope('postprocess_ssh_m3'):
                final_bbox_m3, final_scores_m3, final_category_m3 = self.postprocess_ssh(rois=rois_m3,
                                                                                         bbox_ppred=box_pred_m3,
                                                                                         scores=cls_prob_m3,
                                                                                         img_shape=img_shape,
                                                                                         iou_threshold=cfgs.M3_NMS_IOU_THRESHOLD)

            result_dict = {'final_bbox_m1': final_bbox_m1, 'final_scores_m1': final_scores_m1,
                           'final_category_m1': final_category_m1, 'final_bbox_m2': final_bbox_m2,
                           'final_scores_m2': final_scores_m2,  'final_category_m2': final_category_m2,
                           'final_bbox_m3': final_bbox_m3, 'final_scores_m3': final_scores_m3,
                           'final_category_m3': final_category_m3}

            losses_dict = {'bbox_loss_m1': bbox_loss_m1, 'cls_loss_m1': cls_loss_m1,
                           'bbox_loss_m2': bbox_loss_m2, 'cls_loss_m2': cls_loss_m2,
                           'bbox_loss_m3': bbox_loss_m3, 'cls_loss_m3': cls_loss_m3}

            return result_dict, losses_dict


    def get_restorer(self):
        checkpoint_path = tf.train.latest_checkpoint(os.path.join(cfgs.TRAINED_CKPT, cfgs.VERSION))

        if checkpoint_path != None:
            if cfgs.RESTORE_FROM_RPN:
                print('___restore from rpn___')
                model_variables = slim.get_model_variables()
                restore_variables = [var for var in model_variables if not var.name.startswith('FastRCNN_Head')] + \
                                    [slim.get_or_create_global_step()]
                for var in restore_variables:
                    print(var.name)
                restorer = tf.train.Saver(restore_variables)
            else:
                restorer = tf.train.Saver()
            print("model restore from :", checkpoint_path)
        else:
            checkpoint_path = cfgs.PRETRAINED_CKPT
            print("model restore from pretrained mode, path is :", checkpoint_path)

            model_variables = slim.get_model_variables()
            # for var in model_variables:
            #     print(var.name)
            # print(20*"__++__++__")

            def name_in_ckpt_rpn(var):
                return var.op.name

            def name_in_ckpt_fastrcnn_head(var):
                '''
                Fast-RCNN/resnet_v1_50/block4 -->resnet_v1_50/block4
                Fast-RCNN/MobilenetV2/** -- > MobilenetV2 **
                :param var:
                :return:
                '''
                return '/'.join(var.op.name.split('/')[1:])

            nameInCkpt_Var_dict = {}
            for var in model_variables:
                if var.name.startswith('Fast-RCNN/'+self.base_network_name):  # +'/block4'
                    var_name_in_ckpt = name_in_ckpt_fastrcnn_head(var)
                    nameInCkpt_Var_dict[var_name_in_ckpt] = var
                else:
                    if var.name.startswith(self.base_network_name):
                        var_name_in_ckpt = name_in_ckpt_rpn(var)
                        nameInCkpt_Var_dict[var_name_in_ckpt] = var
                    else:
                        continue
            restore_variables = nameInCkpt_Var_dict
            for key, item in restore_variables.items():
                print("var_in_graph: ", item.name)
                print("var_in_ckpt: ", key)
                print(20*"___")
            restorer = tf.train.Saver(restore_variables)
            print(20 * "****")
            print("restore from pretrained_weighs in IMAGE_NET")
        return restorer, checkpoint_path

    def get_gradients(self, optimizer, loss):
        '''

        :param optimizer:
        :param loss:
        :return:

        return vars and grads that not be fixed
        '''

        # if cfgs.FIXED_BLOCKS > 0:
        #     trainable_vars = tf.trainable_variables()
        #     # trained_vars = slim.get_trainable_variables()
        #     start_names = [cfgs.NET_NAME + '/block%d'%i for i in range(1, cfgs.FIXED_BLOCKS+1)] + \
        #                   [cfgs.NET_NAME + '/conv1']
        #     start_names = tuple(start_names)
        #     trained_var_list = []
        #     for var in trainable_vars:
        #         if not var.name.startswith(start_names):
        #             trained_var_list.append(var)
        #     # slim.learning.train()
        #     grads = optimizer.compute_gradients(loss, var_list=trained_var_list)
        #     return grads
        # else:
        #     return optimizer.compute_gradients(loss)
        return optimizer.compute_gradients(loss)

    def enlarge_gradients_for_bias(self, gradients):

        final_gradients = []
        with tf.variable_scope("Gradient_Mult") as scope:
            for grad, var in gradients:
                scale = 1.0
                if cfgs.MUTILPY_BIAS_GRADIENT and './biases' in var.name:
                    scale = scale * cfgs.MUTILPY_BIAS_GRADIENT
                if not np.allclose(scale, 1.0):
                    grad = tf.multiply(grad, scale)
                final_gradients.append((grad, var))
        return final_gradients























