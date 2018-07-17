# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os, sys
sys.path.append("../")
import tensorflow as tf
import time
import cv2
import pickle
import numpy as np

from data.io.image_preprocess import short_side_resize_for_inference_data
from libs.configs import cfgs
from libs.networks import build_whole_network
from libs.val_libs import voc_eval
from libs.box_utils import draw_box_in_img
from libs.label_name_dict.label_dict import LABEl_NAME_MAP, NAME_LABEL_MAP
from help_utils import tools
from libs.box_utils import nms

os.environ["CUDA_VISIBLE_DEVICES"] = '1'


def merge_result(result_dict):
    final_bbox = [result_dict['final_bbox_m1'], result_dict['final_bbox_m2'], result_dict['final_bbox_m2']]
    final_scores = [result_dict['final_scores_m1'], result_dict['final_scores_m2'], result_dict['final_scores_m3']]
    final_category = [result_dict['final_category_m1'], result_dict['final_category_m2'],
                      result_dict['final_category_m3']]

    final_bbox = np.concatenate(final_bbox, axis=0)
    final_scores = np.concatenate(final_scores, axis=0)
    final_category = np.concatenate(final_category, axis=0)

    keep = nms.py_cpu_nms(final_bbox, final_scores, cfgs.FINAL_NMS_IOU_THRESHOLD)

    final_bbox = final_bbox[keep]
    final_scores = final_scores[keep]
    final_category = final_category[keep]

    return final_bbox, final_scores, final_category


def detect(det_net, src_dir, res_dir, draw_imgs):

    # 1. preprocess img
    img_plac = tf.placeholder(dtype=tf.uint8, shape=[None, None, 3])  # is RGB. not GBR
    img_batch = tf.cast(img_plac, tf.float32)
    img_batch = short_side_resize_for_inference_data(img_tensor=img_batch,
                                                     target_shortside_len=cfgs.IMG_SHORT_SIDE_LEN,
                                                     length_limitation=cfgs.IMG_MAX_LENGTH)
    img_batch = img_batch - tf.constant(cfgs.PIXEL_MEAN)
    img_batch = tf.expand_dims(img_batch, axis=0)  # [1, None, None, 3]

    result_dict = det_net.build_whole_detection_network(input_img_batch=img_batch,
                                                        gtboxes_batch=None)

    init_op = tf.group(
        tf.global_variables_initializer(),
        tf.local_variables_initializer()
    )

    restorer, restore_ckpt = det_net.get_restorer()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(init_op)
        if not restorer is None:
            restorer.restore(sess, restore_ckpt)
            print('restore model')

        sub_folders = os.listdir(src_dir)
        for sub_folder in sub_folders:

            folder_dir = os.path.join(src_dir, sub_folder)
            real_test_imgname_list = [os.path.join(folder_dir, img_name) for img_name in os.listdir(folder_dir)]

            tools.mkdir(os.path.join(res_dir, sub_folder))

            for i, a_img_name in enumerate(real_test_imgname_list):

                raw_img = cv2.imread(a_img_name)

                raw_h, raw_w = raw_img.shape[0], raw_img.shape[1]

                start = time.time()
                resized_img, result_dict_ = \
                    sess.run(
                        [img_batch, result_dict],
                        feed_dict={img_plac: raw_img[:, :, ::-1]}  # cv is BGR. But need RGB
                    )
                end = time.time()

                detected_boxes, detected_scores, detected_categories = merge_result(result_dict_)

                nake_name = a_img_name.split('/')[-1]

                xmin, ymin, xmax, ymax = detected_boxes[:, 0], detected_boxes[:, 1], \
                                         detected_boxes[:, 2], detected_boxes[:, 3]

                resized_h, resized_w = resized_img.shape[1], resized_img.shape[2]

                xmin = xmin * raw_w / resized_w
                xmax = xmax * raw_w / resized_w

                ymin = ymin * raw_h / resized_h
                ymax = ymax * raw_h / resized_h

                boxes = np.transpose(np.stack([xmin, ymin, xmax, ymax]))
                dets = np.hstack((detected_categories.reshape(-1, 1),
                                  detected_scores.reshape(-1, 1),
                                  boxes))

                show_indices = detected_scores >= cfgs.SHOW_SCORE_THRSHOLD
                show_scores = detected_scores[show_indices]
                show_boxes = boxes[show_indices]
                show_categories = detected_categories[show_indices]

                f = open(os.path.join(res_dir, sub_folder) + '/' + nake_name.split('.')[0] + '.txt', 'w')
                f.write('{}\n'.format(nake_name.split('.')[0]))
                # f.write('{}\n'.format(dets.shape[0]))
                # for inx in range(dets.shape[0]):
                #
                #     f.write('%d %d %d %d %.3f\n' % (int(dets[inx][2]),
                #                                     int(dets[inx][3]),
                #                                     int(dets[inx][4]) - int(dets[inx][2]),
                #                                     int(dets[inx][5]) - int(dets[inx][3]),
                #                                     dets[inx][1]))

                f.write('{}\n'.format(show_boxes.shape[0]))
                for inx in range(show_boxes.shape[0]):
                    f.write('%d %d %d %d %.3f\n' % (int(show_boxes[inx][0]),
                                                    int(show_boxes[inx][1]),
                                                    int(show_boxes[inx][2]) - int(show_boxes[inx][0]),
                                                    int(show_boxes[inx][3]) - int(show_boxes[inx][1]),
                                                    show_scores[inx]))

                if draw_imgs:
                    final_detections = draw_box_in_img.draw_boxes_with_label_and_scores(raw_img - np.array(cfgs.PIXEL_MEAN),
                                                                                        boxes=show_boxes,
                                                                                        labels=show_categories,
                                                                                        scores=show_scores)

                    tools.mkdir(cfgs.TEST_SAVE_PATH)
                    cv2.imwrite(cfgs.TEST_SAVE_PATH + '/' + nake_name,
                                final_detections)

                tools.view_bar('{} image cost {}s'.format(a_img_name, (end - start)), i + 1, len(real_test_imgname_list))


def inference(src_dir, res_dir, draw_imgs):

    faster_rcnn = build_whole_network.DetectionNetwork(base_network_name=cfgs.NET_NAME,
                                                       is_training=False)
    detect(det_net=faster_rcnn, src_dir=src_dir, res_dir=res_dir, draw_imgs=draw_imgs)


if __name__ == '__main__':

    # eval('/home/yjr/PycharmProjects/Faster-RCNN_TF/tools/inference_image')
    inference('/mnt/USBB/gaoxun/WIDER_VAL/images',
              '/mnt/USBB/gaoxun/SSH_ANCHOR/SSH_Tensorflow/eval_tools/pred',
              False)
















