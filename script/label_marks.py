# -*- coding:utf-8 -*-
import xml.etree.ElementTree as ET
# import pickle
from collections import defaultdict
import os
import cv2
import sys

class_zh_en = {'痣': 'mole',
               # '雀斑': 'freckles',
               '黄褐斑': 'chloasma'
               # ,
               # '痤疮': 'acne',
               # '老年斑': 'age spots',
               # '晒斑': 'sunburn',
               # '褐青色痣': 'brown blue nevus',
               # '炎症性色素沉着': 'inflammatory pigmentation',
               # '太田痣': 'nevus of ota',
               # '咖啡斑': 'coffee spot',
               # '雀斑样痣': 'lentigo simplex',
               # '湿疹': 'eczema',
               # '特应性皮炎': 'atopic dermatitis',
               # '接触性皮炎': 'contact dermatitis',
               # '激素依赖性皮炎': 'hormone-dependent dermatitis',
               # '敏感': 'sensitivity',
               # '口周皮炎': 'perioral dermatitis',
               # '单纯疱疹': 'herpes simplex',
               # '带状疱疹': 'herpes zoster',
               # '扁平疣': 'flat wart',
               # '风疹': 'rubella',
               # '色素痣': 'pigmented nevus',
               # '斑痣': 'against',
               # '脂溢性角化病': 'seborrheic keratosis',
               # '汗管瘤': 'vulvar syringomas',
               # '粟丘疹': 'millet pimples',
               # '黑头粉刺': 'blackheads',
               # '白头粉刺': 'whiteheads',
               # '丘疹性痤疮': 'pimple acne',
               # '脓包性痤疮': 'purulent acne',
               # '硬结性痤疮': 'hard acne',
               # '囊肿性痤疮': 'cystic acne'
               }


def get_color(x, idmax):
  colors = [[255, 0, 255], [0, 0, 255], [0, 255, 255],
            [0, 255, 0], [255, 255, 0], [255, 0, 0]]
  x = (x * 123457) % idmax
  ratio = float(5.0 * x / idmax)
  i = int(ratio)
  j = i + 1
  ratio = float(ratio - i)
  r = int((1 - ratio) * colors[i][0] + ratio * colors[j][0])
  g = int((1 - ratio) * colors[i][1] + ratio * colors[j][1])
  b = int((1 - ratio) * colors[i][2] + ratio * colors[j][2])
  return [r, g, b]


def mark_image(xml_file, image_file, out_dir):
  if not os.path.exists(image_file):
    print('image file does not exists.')
    return
  tree = ET.parse(xml_file)
  xml_root = tree.getroot()
  im = cv2.imread(image_file)
  file_name = image_file.split('/')[-1]

  _keys = list(class_zh_en.keys())
  objs = xml_root.findall('object')
  for obj in objs:

    cls = obj.find('name').text
    cls = cls.lower()
    cls = cls.strip('\r')
    cls = cls.strip('\n')

    if cls in _keys:
      xmlbox = obj.find('bndbox')
      xmin = int(xmlbox.find('xmin').text)
      ymin = int(xmlbox.find('ymin').text)
      xmax = int(xmlbox.find('xmax').text)
      ymax = int(xmlbox.find('ymax').text)

      color = get_color(_keys.index(cls), 4)
      cv2.rectangle(im, (xmin, ymin), (xmax, ymax), color, 5)
      cv2.putText(im, class_zh_en[cls], (xmin, ymin - 20),
                  cv2.FONT_HERSHEY_PLAIN, 4,
                  color, thickness=5, lineType=2)

  if len(objs) > 0:
    out_img_file = os.path.join(out_dir, file_name)
    cv2.imwrite(out_img_file, im)


def mark_label_on_img(annotation_dir, images_list, out_dir):

  if os.path.isfile(images_list):

    with open(images_list, 'r') as in_file:
      file_list = in_file.readlines()
      for file in file_list:
        file = file.strip()
        if os.path.isfile(file):
          xml_file = os.path.join(annotation_dir, file.replace('.jpg', '.xml'))
          mark_image(xml_file, file, out_dir)
  elif os.path.isdir(images_list):

    for root, dirs, files in os.walk(annotation_dir):
      for file in files:
        if file.endswith('.xml'):
          xml_file = os.path.join(annotation_dir, file)
          image_file = os.path.join(images_list, file.replace('.xml', '.jpg'))
          mark_image(xml_file, image_file, out_dir)
  else:
    print(images_list, ' is not a directory or a images_list file.')


def count_label_info(annotation_dir, out_file):
  label_cnt_dict = defaultdict(int)
  for root, dirs, files in os.walk(annotation_dir):
    for file in files:
      if file.endswith('.xml'):
        xml_file = os.path.join(root, file)
        tree = ET.parse(xml_file)
        xml_root = tree.getroot()
        objs = xml_root.findall('object')
        for i, obj in enumerate(objs):
          cls = obj.find('name').text
          if not cls:
            continue
          cls = cls.lower()
          cls = cls.strip('\r')
          cls = cls.strip('\n')
          cls = cls.strip(' ')
          label_cnt_dict[cls] += 1
        pass
  sorted(label_cnt_dict.items(), key=lambda x: x[1], reverse=True)
  with open(out_file, 'w') as out:
    for key, value in label_cnt_dict.items():
      out.write('%s:%d\n' % (key, value))


if __name__ == '__main__':
  op = sys.argv[1]
  if op == 'count':
    count_label_info(sys.argv[2], sys.argv[3])
  elif op == 'mark':
    mark_label_on_img(sys.argv[2], sys.argv[3], sys.argv[4])
  else:
    print('usage: python3 label_marks.py count annotation_dir out_file')
    print('usage: python3 label_marks.py mark '
          'annotation_dir images_list out_dir')
