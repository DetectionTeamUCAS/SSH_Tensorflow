# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import os
import tensorflow as tf

# ------------------------------------------------
VERSION = 'SSH_20180705'
NET_NAME = 'resnet_v1_101'  # 'MobilenetV2'
ADD_BOX_IN_TENSORBOARD = True

# ---------------------------------------- System_config
ROOT_PATH = os.path.abspath('../')
print(20*"++--")
print(ROOT_PATH)
GPU_GROUP = "3"
SHOW_TRAIN_INFO_INTE = 10
SMRY_ITER = 100
SAVE_WEIGHTS_INTE = 10000

SUMMARY_PATH = ROOT_PATH + '/output/summary'
TEST_SAVE_PATH = ROOT_PATH + '/tools/test_result'

if NET_NAME.startswith("resnet"):
    weights_name = NET_NAME
elif NET_NAME.startswith("MobilenetV2"):
    weights_name = "mobilenet/mobilenet_v2_1.0_224"
else:
    raise Exception('net name must in [resnet_v1_101, resnet_v1_50, MobilenetV2]')

PRETRAINED_CKPT = ROOT_PATH + '/data/pretrained_weights/' + weights_name + '.ckpt'
TRAINED_CKPT = os.path.join(ROOT_PATH, 'output/trained_weights')
EVALUATE_DIR = ROOT_PATH + '/output/evaluate_result_pickle/' + VERSION

# ------------------------------------------ Train config
RESTORE_FROM_RPN = False
IS_FILTER_OUTSIDE_BOXES = True
FIXED_BLOCKS = 1  # allow 0~3

M1_LOCATION_LOSS_WEIGHT = 1.
M1_CLASSIFICATION_LOSS_WEIGHT = 1.0

M2_LOCATION_LOSS_WEIGHT = 1.0
M2_CLASSIFICATION_LOSS_WEIGHT = 1.0

M3_LOCATION_LOSS_WEIGHT = 1.0
M3_CLASSIFICATION_LOSS_WEIGHT = 1.0

MUTILPY_BIAS_GRADIENT = None   # 2.0  # if None, will not multipy
GRADIENT_CLIPPING_BY_NORM = None   # 10.0  if None, will not clip

EPSILON = 1e-5
MOMENTUM = 0.9
LR = 0.001
DECAY_STEP = [50000, 70000]
MAX_ITERATION = 200000

# -------------------------------------------- Data_preprocess_config
DATASET_NAME = 'WIDER'  # 'ship', 'spacenet', 'pascal', 'coco'
PIXEL_MEAN = [123.68, 116.779, 103.939]  # R, G, B. In tf, channel is RGB. In openCV, channel is BGR
IMG_SHORT_SIDE_LEN = 1000
IMG_MAX_LENGTH = 3000
CLASS_NUM = 1

# --------------------------------------------- Network_config
BATCH_SIZE = 1
INITIALIZER = tf.random_normal_initializer(mean=0.0, stddev=0.01)
BBOX_INITIALIZER = tf.random_normal_initializer(mean=0.0, stddev=0.001)
WEIGHT_DECAY = 0.00005 if NET_NAME.startswith('Mobilenet') else 0.0001


# ---------------------------------------------Anchor config
BASE_ANCHOR_SIZE_LIST = [16]
ANCHOR_STRIDE = [16, 16, 32]
M1_ANCHOR_SCALES = [0.5, 1.0, 1.5]
M2_ANCHOR_SCALES = [0.5, 1.0, 1.5, 2.0, 4.0, 8.0]
M3_ANCHOR_SCALES = [0.5, 1.0, 1.5, 16.0, 32.0]
ANCHOR_RATIOS = [1.]
ROI_SCALE_FACTORS = [10., 10., 5.0, 5.0]
ANCHOR_SCALE_FACTORS = None

EXTRA_SHIFTED_ANCHOR = False
FACE_SHIFT_JITTER = False
NUM_SHIFT_JITTER = 4

# -------------------------------------------detection module config
M1_NMS_IOU_THRESHOLD = 0.1
M1_SIGMA = 1.0

M2_NMS_IOU_THRESHOLD = 0.2
M2_SIGMA = 1.0

M3_NMS_IOU_THRESHOLD = 0.3
M3_SIGMA = 1.0

TOP_K_NMS_TRAIN = 0
MAXIMUM_PROPOSAL_TARIN = 2000
TOP_K_NMS_TEST = 0
MAXIMUM_PROPOSAL_TEST = 300

M1_IOU_POSITIVE_THRESHOLD = 0.25
M1_IOU_NEGATIVE_THRESHOLD_UP = 0.1
M1_IOU_NEGATIVE_THRESHOLD_DOWN = 0.0   # 0.1 < IOU < 0.3 is negative
M1_MINIBATCH_SIZE = 256  # if is -1, that is train with OHEM
M1_POSITIVE_RATE = 0.25

M2_IOU_POSITIVE_THRESHOLD = 0.5
M2_IOU_NEGATIVE_THRESHOLD_UP = 0.3
M2_IOU_NEGATIVE_THRESHOLD_DOWN = 0.0   # 0.1 < IOU < 0.5 is negative
M2_MINIBATCH_SIZE = 256  # if is -1, that is train with OHEM
M2_POSITIVE_RATE = 0.25

M3_IOU_POSITIVE_THRESHOLD = 0.5
M3_IOU_NEGATIVE_THRESHOLD_UP = 0.3
M3_IOU_NEGATIVE_THRESHOLD_DOWN = 0.0   # 0.1 < IOU < 0.5 is negative
M3_MINIBATCH_SIZE = 256  # if is -1, that is train with OHEM
M3_POSITIVE_RATE = 0.25

SHOW_SCORE_THRSHOLD = 0.7  # only show in tensorboard
SSH_NMS_MAX_BOXES_PER_CLASS = 100

ADD_GTBOXES_TO_TRAIN = False
FINAL_NMS_IOU_THRESHOLD = 0.2

