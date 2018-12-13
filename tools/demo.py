#!/usr/bin/env python
# coding=utf-8
# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from model.config import cfg
from model.test import im_detect
from model.nms_wrapper import nms

from utils.timer import Timer
import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import argparse

from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1

# import sys
# reload(sys)
# sys.setdefaultencoding('utf-8')
from draw_boxes_and_labels_to_image import draw_boxes_and_labels_to_image

from matplotlib.font_manager import FontProperties


# CLASSES = ('__background__',
#            'aeroplane', 'bicycle', 'bird', 'boat',
#            'bottle', 'bus', 'car', 'cat', 'chair',
#            'cow', 'diningtable', 'dog', 'horse',
#            'motorbike', 'person', 'pottedplant',
#            'sheep', 'sofa', 'train', 'tvmonitor')

class_zh_en = {'痣': 'mole',
               '雀斑': 'freckles',
               '黄褐斑': 'chloasma',
               '痤疮': 'acne',
               '老年斑': 'age spots',
               '晒斑': 'sunburn',
               '褐青色痣': 'brown blue nevus',
               '炎症性色素沉着': 'inflammatory pigmentation',
               '太田痣': 'nevus of ota',
               '咖啡斑': 'coffee spot',
               '雀斑样痣': 'lentigo simplex',
               '湿疹': 'eczema',
               '特应性皮炎': 'atopic dermatitis',
               '接触性皮炎': 'contact dermatitis',
               '激素依赖性皮炎': 'hormone-dependent dermatitis',
               '敏感': 'sensitivity',
               '口周皮炎': 'perioral dermatitis',
               '单纯疱疹': 'herpes simplex',
               '带状疱疹': 'herpes zoster',
               '扁平疣': 'flat wart',
               '风疹': 'rubella',
               '色素痣': 'pigmented nevus',
               '斑痣': 'against',
               '脂溢性角化病': 'seborrheic keratosis',
               '汗管瘤': 'vulvar syringomas',
               '粟丘疹': 'millet pimples',
               '黑头粉刺': 'blackheads',
               '白头粉刺': 'whiteheads',
               '丘疹性痤疮': 'pimple acne',
               '脓包性痤疮': 'purulent acne',
               '硬结性痤疮': 'hard acne',
               '囊肿性痤疮': 'cystic acne',
               '炎症后色素沉着': 'yanzheng',
               '黑头': 'blackhead'

               }

# CLASSES = ('__background__',  # always index 0
#            u'老年斑', u'褐青色痣', u'雀斑', u'黄褐斑',
#            u'色素痣', u'晒斑', u'痤疮', u'炎症后色素沉着', u'黑头', u'痣', u'太田痣', u'敏感', u'雀斑样痣')

CLASSES = ('__background__',  # always index 0
           u'老年斑', u'褐青色痣', u'雀斑', u'黄褐斑',
           u'色素痣', u'晒斑', u'痤疮', u'炎症后色素沉着', u'黑头', u'太田痣', u'敏感')

# CLASSES = ('__background__',  # always index 0
#            u'老年斑', u'褐青色痣', u'雀斑', u'黄褐斑',
#            u'色素痣', u'晒斑', u'痤疮', u'炎症后色素沉着', u'黑头')

NETS = {'vgg16': ('vgg16_faster_rcnn_iter_70000.ckpt',),
        'res101': ('res101_faster_rcnn_iter_110000.ckpt',)}
DATASETS = {'pascal_voc': ('voc_2007_trainval',), 'pascal_voc_0712': (
    'voc_2007_trainval+voc_2012_trainval',), 'dianli': ('custom_train',)}


def vis_detections(im, class_name, dets, thresh=0.5):
  """Draw detected bounding boxes."""
  # 字符串转换
  if isinstance(class_name, unicode):
    class_name = class_name.encode(encoding='UTF-8')
  else:
    class_name = class_name.decode(encoding='UTF-8')
  class_name = class_zh_en[class_name]
  inds = np.where(dets[:, -1] >= thresh)[0]
  if len(inds) == 0:
    return

  im = im[:, :, (2, 1, 0)]
  fig, ax = plt.subplots(figsize=(12, 12))
  ax.imshow(im, aspect='equal')
  for i in inds:
    bbox = dets[i, :4]
    score = dets[i, -1]

    ax.add_patch(
        plt.Rectangle((bbox[0], bbox[1]),
                      bbox[2] - bbox[0],
                      bbox[3] - bbox[1], fill=False,
                      edgecolor='red', linewidth=3.5)
    )

    # print('++++++++++++')
    # print('{:s} {:.3f}'.format(class_name, score))
    ax.text(bbox[0], bbox[1] - 2,
            '{:s} {:.3f}'.format(class_name, score),
            bbox=dict(facecolor='blue', alpha=0.5),
            fontsize=14, color='white')
  # # 指定中文字体
  # myfont = FontProperties(
  #     fname='/usr/share/fonts/truetype/wqy/wqy-microhei.ttc')
  ax.set_title(('{} detections with '
                'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                thresh), fontsize=14)
  plt.axis('off')
  plt.tight_layout()
  plt.draw()


def vis_detections_byimg(im, class_name_list, dets_list, im_name, thresh=0.5):
  """Draw detected bounding boxes."""
  im = im[:, :, (2, 1, 0)]
  fig, ax = plt.subplots()
  ax.imshow(im, aspect='equal')
  for indx, dets in enumerate(dets_list):
    class_name = class_name_list[indx]
    # 字符串转换
    if isinstance(class_name, unicode):
      class_name = class_name.encode(encoding='UTF-8')
    else:
      class_name = class_name.decode(encoding='UTF-8')
    class_name = class_zh_en[class_name]
    inds = np.where(dets[:, -1] >= thresh)[0]

    if len(inds) == 0:
      continue

    # im = im[:, :, (2, 1, 0)]
    # fig, ax = plt.subplots(figsize=(12, 12))
    # ax.imshow(im, aspect='equal')
    for i in inds:
      bbox = dets[i, :4]
      score = dets[i, -1]

      ax.add_patch(
          plt.Rectangle((bbox[0], bbox[1]),
                        bbox[2] - bbox[0],
                        bbox[3] - bbox[1], fill=False,
                        edgecolor='red', linewidth=1.5)
      )

      # print('++++++++++++')
      # print('{:s} {:.3f}'.format(class_name, score))
      ax.text(bbox[0], bbox[1] - 2,
              '{:s} {:.3f}'.format(class_name, score),
              bbox=dict(facecolor='blue', alpha=0.5),
              fontsize=5, color='white')
    # plt.axis('off')
    # plt.draw()
    # # 指定中文字体
    # myfont = FontProperties(
    #     fname='/usr/share/fonts/truetype/wqy/wqy-microhei.ttc')
    # ax.set_title(('{} detections with '
    #               'p({} | box) >= {:.1f}').format(class_name, class_name,
    #                                               thresh), fontsize=14)
  ax.set_title('detections for %s' % im_name, fontsize=14)
  plt.axis('off')
  plt.tight_layout()
  # plt.draw()
  plt.savefig('/data/frcnn_result/20181126_2/result/%s' % im_name, dpi=1100)
  plt.close('all')


def demo(sess, net, image_name):
  """Detect object classes in an image using pre - computed object proposals."""

  # Load the demo image
  im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
  # im_file = os.path.join(
  #     cfg.DATA_DIR, 'VOCdevkit2007/VOC2007/JPEGImages', image_name)
  # im_file = os.path.join(
  #     cfg.DATA_DIR, '/data/frcnn_result/20181126_2/img', image_name)

  im = cv2.imread(im_file)
  # Detect all object classes and regress object bounds
  timer = Timer()
  timer.tic()
  scores, boxes = im_detect(sess, net, im)
  timer.toc()
  print('Detection took {:.3f}s for {:d} object proposals'.format(
      timer.total_time, boxes.shape[0]))

  # 定义cls和det的list
  cls_list = []
  dets_list = []
  # Visualize detections for each class
  CONF_THRESH = 0.8
  NMS_THRESH = 0.3
  for cls_ind, cls in enumerate(CLASSES[1:]):
    cls_ind += 1  # because we skipped background
    cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
    cls_scores = scores[:, cls_ind]
    # 通过类别筛选
    dets = np.hstack((cls_boxes,
                      cls_scores[:, np.newaxis])).astype(np.float32)
    # 针对某一类进行NMS
    keep = nms(dets, NMS_THRESH)
    dets = dets[keep, :]
    cls_list.append(cls)
    dets_list.append(dets)
    # vis_detections(im, cls, dets, thresh=CONF_THRESH)

  # 按图片画box
  # vis_detections_byimg(im, cls_list, dets_list,
  #                      image_name, thresh=CONF_THRESH)


def demo2(sess, net, image_path, out_path, image_name):
  """Detect object classes in an image using pre - computed object proposals."""

  # Load the demo image
  im_file = os.path.join(image_path, image_name)
  save_path = os.path.join(out_path, image_name)

  im = cv2.imread(im_file)

  # Detect all object classes and regress object bounds
  timer = Timer()
  timer.tic()
  # 提取300个proposal,boxes(300,40),scores(300,10)
  scores, boxes = im_detect(sess, net, im)

  timer.toc()
  print('Detection took {:.3f}s for {:d} object proposals'.format(
        timer.total_time, boxes.shape[0]))

  classes = []
  coords = []
  scores_ = []
  # Visualize detections for each class
  CONF_THRESH = 0.8
  NMS_THRESH = 0.3
  for cls_ind, cls in enumerate(CLASSES[1:]):
    if isinstance(cls, unicode):
      cls = cls.encode(encoding='UTF-8')
    else:
      cls = cls.decode(encoding='UTF-8')

    cls_ind += 1  # because we skipped background
    cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
    # scores = np.array(scores)
    cls_scores = scores[:, cls_ind]

    # 将每个类别的信息组合起来,dets(300,5)
    dets = np.hstack((cls_boxes,
                      cls_scores[:, np.newaxis])).astype(np.float32)

    # 针对某一类进行NMS,之后的dets(?,5)
    keep = nms(dets, NMS_THRESH)
    dets = dets[keep, :]

    # 对NMS后的dets根据阈值再次筛选
    inds = np.where(dets[:, -1] >= CONF_THRESH)[0]

    # 如果len为0,说明此类无有效检测,继续下一个类的处理
    if len(inds) == 0:
      continue

    for i in inds:
      bbox = dets[i, :4]
      score = dets[i, -1]
      coords.append(bbox)
      scores_.append(score)
      classes.append(class_zh_en[cls])

  # cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
  draw_boxes_and_labels_to_image(im[..., ::-1], coords, scores_, classes, is_center=False,
                                 is_rescale=False, save_name=save_path)


def parse_args():
  """Parse input arguments."""
  parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
  parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101]',
                      choices=NETS.keys(), default='res101')
  parser.add_argument('--dataset', dest='dataset', help='Trained dataset [pascal_voc pascal_voc_0712]',
                      choices=DATASETS.keys(), default='pascal_voc_0712')
  args = parser.parse_args()

  return args


if __name__ == '__main__':
  cfg.TEST.HAS_RPN = True  # Use RPN for proposals
  args = parse_args()

  # model path
  demonet = args.demo_net
  dataset = args.dataset
  # tfmodel = os.path.join('output', demonet, DATASETS[dataset][0], 'default',
  #                        NETS[demonet][0])
  # tfmodel = os.path.join('output_b1', demonet, 'voc_2007_trainval', 'default',
  #                        'res101_faster_rcnn_iter_70000.ckpt')
  # tfmodel = os.path.join('output', demonet, 'voc_2007_trainval', 'default',
  #                        'res101_faster_rcnn_iter_60000.                                      )

  tfmodel = os.path.join('output_109', 'res101', 'voc_2007_trainval', 'default',
                         'res101_faster_rcnn_iter_70000.ckpt')

  if not os.path.isfile(tfmodel + '.meta'):
    raise IOError(('{:s} not found.\nDid you download the proper networks from '
                   'our server and place them properly?').format(tfmodel + '.meta'))

  # set tf.ConfigProto()
  tfconfig = tf.ConfigProto(allow_soft_placement=True)
  tfconfig.gpu_options.allow_growth = True

  # init session
  sess = tf.Session(config=tfconfig)
  # load network
  if demonet == 'vgg16':
    net = vgg16()
  elif demonet == 'res101':
    net = resnetv1(num_layers=101)
  else:
    raise NotImplementedError
  net.create_architecture("TEST", 12,
                          tag='default', anchor_scales=[8, 16, 32])
  saver = tf.train.Saver()
  saver.restore(sess, tfmodel)

  print('Loaded network {:s}'.format(tfmodel))

  # im_names = ['000456.jpg', '000542.jpg', '001150.jpg',
  #             '001763.jpg', '004545.jpg']
  # im_names = ['11976.jpg', '11978.jpg', '16507.jpg',
  #             '16514.jpg', '16515.jpg', '16542.jpg', '16559.jpg']
  # im_names = ['1.jpg', '2.jpg']

  # for im_name in im_names:
  #     # for im_name in im_names:
  #   print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
  #   print('Demo for data/demo/{}'.format(im_name))
  #   # im_name = '%s.jpg' % im_name.strip( )
  #   demo(sess, net, im_name)

  with open('/home/ai002/dev/gitclone/tf-faster-rcnn/data/VOCdevkit2007/VOC2007/ImageSets/Main/test.txt') as f:
    n_list = f.readlines()
    for im_name in n_list:
      im_name = '%s.jpg' % im_name.strip()
      print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
      print('Demo for data/demo/{}'.format(im_name))
      image_path = '/home/ai002/dev/gitclone/tf-faster-rcnn/data/VOCdevkit2007/VOC2007/JPEGImages/'
      save_path = '/data/frcnn_result/20181205/result/'
      demo2(sess, net, image_path, save_path, im_name)

  # for parent, _, files in os.walk('/data/frcnn_result/20181126_2/img'):
  #   for im_name in files:
  #     # for im_name in im_names:
  #     print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
  #     print('Demo for data/demo/{}'.format(im_name))
  #     # im_name = '%s.jpg' % im_name.strip()
  #     demo(sess, net, im_name)

  # plt.show()
