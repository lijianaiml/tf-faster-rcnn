#!/usr/bin/env python
# coding=utf-8
import os
import imageio
import numpy as np
import tensorlayer as tl
import cv2
# from xml_helper import *

class_zh_en = {u'痣': 'mole',
               u'雀斑': 'freckles',
               u'黄褐斑': 'chloasma',
               u'痤疮': 'acne',
               u'老年斑': 'age spots',
               u'晒斑': 'sunburn',
               u'褐青色痣': 'brown blue nevus',
               u'炎症后色素沉着': 'inflammatory pigmentation',
               u'太田痣': 'nevus of ota',
               u'咖啡斑': 'coffee spot',
               u'雀斑样痣': 'lentigo simplex',
               u'湿疹': 'eczema',
               u'特应性皮炎': 'atopic dermatitis',
               u'接触性皮炎': 'contact dermatitis',
               u'激素依赖性皮炎': 'hormone-dependent dermatitis',
               u'敏感': 'sensitivity',
               u'口周皮炎': 'perioral dermatitis',
               u'单纯疱疹': 'herpes simplex',
               u'带状疱疹': 'herpes zoster',
               u'扁平疣': 'flat wart',
               u'风疹': 'rubella',
               u'色素痣': 'pigmented nevus',
               u'斑痣': 'against',
               u'脂溢性角化病': 'seborrheic keratosis',
               u'汗管瘤': 'vulvar syringomas',
               u'粟丘疹': 'millet pimples',
               u'黑头粉刺': 'blackheads',
               u'白头粉刺': 'whiteheads',
               u'丘疹性痤疮': 'pimple acne',
               u'脓包性痤疮': 'purulent acne',
               u'硬结性痤疮': 'hard acne',
               u'囊肿性痤疮': 'cystic acne',
               u'炎症性色沉着': 'yanzheng',
               u'炎症性色沉': 'yanzheng',
               u'黑头': 'blackhead'
               }


def get_xml_info(xml_path):
  coords = []
  classes_list = []
  cord_name_list = parse_xml(xml_path)
  for list_ in cord_name_list:
    coords.append(list_[:4])
    classes_list.append(class_zh_en[list_[-1]])

  return coords, classes_list


def draw_boxes_and_labels_to_image(
        image, coords, scores, classes_list, is_center=True, is_rescale=True, save_name=None
):
  """Draw bboxes and class labels on image. Return or save the image with bboxes, example in the docs of ``tl.prepro``.

  Parameters
  -----------
  image : numpy.array
      The RGB image [height, width, channel].
  classes : list of int
      A list of class ID (int).
  coords : list of int
      A list of list for coordinates.
          - Should be [x, y, x2, y2] (up-left and botton-right format)
          - If [x_center, y_center, w, h] (set is_center to True).
  scores : list of float
      A list of score (float). (Optional)
  classes_list : list of str
      for converting ID to string on image.
  is_center : boolean
      Whether the coordinates is [x_center, y_center, w, h]
          - If coordinates are [x_center, y_center, w, h], set it to True for converting it to [x, y, x2, y2] (up-left and botton-right) internally.
          - If coordinates are [x1, x2, y1, y2], set it to False.
  is_rescale : boolean
      Whether to rescale the coordinates from pixel-unit format to ratio format.
          - If True, the input coordinates are the portion of width and high, this API will scale the coordinates to pixel unit internally.
          - If False, feed the coordinates with pixel unit format.
  save_name : None or str
      The name of image file (i.e. image.png), if None, not to save image.

  Returns
  -------
  numpy.array
      The saved image.

  References
  -----------
  - OpenCV rectangle and putText.
  # skimage.draw.rectangle>`__.
  - `scikit-image <http://scikit-image.org/docs/dev/api/skimage.draw.html

  """
  # if len(coords) != len(classes):
  #     raise AssertionError("number of coordinates and classes are equal")

  # if len(scores) > 0 and len(scores) != len(classes):
  #     raise AssertionError("number of scores and classes are equal")

  # don't change the original image, and avoid error https://stackoverflow.com/questions/30249053/python-opencv-drawing-errors-after-manipulating-array-with-numpy
  image = image.copy()

  imh, imw = image.shape[0:2]
  thick = int((imh + imw) // 430)

  for i, _v in enumerate(coords):
    if is_center:
      x, y, x2, y2 = tl.prepro.obj_box_coord_centroid_to_upleft_butright(
          coords[i])
    else:
      x, y, x2, y2 = coords[i]

    if is_rescale:  # scale back to pixel unit if the coords are the portion of width and high
      x, y, x2, y2 = tl.prepro.obj_box_coord_scale_to_pixelunit(
          [x, y, x2, y2], (imh, imw))

    cv2.rectangle(
        image,
        (int(x), int(y)),
        (int(x2), int(y2)),  # up-left and botton-right
        [0, 255, 0],
        thick
    )

    # cv2.putText(
    #     image,
    #     classes_list[classes[i]] +
    #     ((" %.2f" % (scores[i])) if (len(scores) != 0) else " "),
    #     (int(x), int(y)),  # button left
    #     0,
    #     1.5e-3 * imh,  # bigger = larger font
    #     [0, 0, 256],  # self.meta['colors'][max_indx],
    #     int(thick / 2) + 1
    # )  # bold
    cv2.putText(
        image,
        classes_list[i] +
        ((" %.2f" % (scores[i])) if (len(scores) != 0) else " "),
        (int(x), int(y)),  # button left
        0,
        1.5e-3 * imh,  # bigger = larger font
        [0, 0, 256],  # self.meta['colors'][max_indx],
        int(thick / 2) + 1
    )  # bold

  if save_name is not None:
    # cv2.imwrite('_my.png', image)
    save_image(image, save_name)
  # if len(coords) == 0:
  #     tl.logging.info("draw_boxes_and_labels_to_image: no bboxes exist, cannot draw !")
  return image


def save_image(image, image_path='_temp.png'):
  """Save a image.

  Parameters
  -----------
  image : numpy array
      [w, h, c]
  image_path : str
      path

  """
  try:  # RGB
    imageio.imwrite(image_path, image)
  except Exception:  # Greyscale
    imageio.imwrite(image_path, image[:, :, 0])


if __name__ == '__main__':
  # with open('/home/ai002/dev/gitclone/tf-faster-rcnn/data/VOCdevkit2007/VOC2007/ImageSets/Main/test.txt') as f:
  #     n_list = f.readlines()
  #     for name in n_list:
  #         im_name = '%s.jpg' % name.strip()
  #         xml_name = '%s.xml' % name.strip()
  #         im_path = '/home/ai002/dev/gitclone/tf-faster-rcnn/data/VOCdevkit2007/VOC2007/JPEGImages/%s' % im_name
  #         xml_path = '/home/ai002/dev/gitclone/tf-faster-rcnn/data/VOCdevkit2007/VOC2007/Annotations/%s' % xml_name
  #         coords, classes_list = get_xml_info(xml_path)
  #         # image = cv2.imread(im_path)
  #         image = imageio.imread(im_path)
  #         print('Draw Box for %s' % im_name)
  #         draw_boxes_and_labels_to_image(image, coords, [], classes_list, is_center=False,
  #                                        is_rescale=False, save_name='/data/frcnn_output_test/%s' % im_name)

  for parent, _, files in os.walk('/home/ai002/dev/gitclone/tf-faster-rcnn/data/VOCdevkit2007/VOC2007/JPEGImages'):
    for file in files:
      im_name = file
      im_path = os.path.join(parent, file)
      xml_path = os.path.join(
          '/home/ai002/dev/gitclone/tf-faster-rcnn/data/VOCdevkit2007/VOC2007/Annotations', file[:-4] + '.xml')
      coords, classes_list = get_xml_info(xml_path)
      # image = cv2.imread(im_path)
      image = imageio.imread(im_path)
      print('Draw Box for %s' % im_name)
      try:
        draw_boxes_and_labels_to_image(image, coords, [], classes_list, is_center=False,
                                       is_rescale=False, save_name='/data/frcnn_result/20181205/ground_truth/%s' % im_name)
      except Exception as e:
        pass
      continue
