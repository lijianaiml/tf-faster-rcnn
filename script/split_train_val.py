import os
import random


def split_train_val(name_path, out_path):
  '''
  输入：
      name_path:合并后xml文件夹（包含所有类别的xml）
      out_path:输出的trainval.txt和train.txt的路径
  输出：
      随机分配的train和val和test数据（7:1:2）
  '''
  random.seed(2018)
  for filename in os.listdir(name_path):
    filename = filename[:-4]
    i = random.randint(1, 11)
    with open(out_path + "trainval.txt", "a") as f:
      f.write(filename)
      f.write('\n')
    if i == 1:
      with open(out_path + "val.txt", "a") as f:
        f.write(filename)
        f.write('\n')
    elif i == 2 or i == 3:
      with open(out_path + "test.txt", "a") as f:
        f.write(filename)
        f.write('\n')
    else:
      with open(out_path + "train.txt", "a") as f:
        f.write(filename)
        f.write('\n')


if __name__ == '__main__':
  split_train_val(
      "/home/ai002/dev/gitclone/tf-faster-rcnn/data/VOCdevkit2007/VOC2007/Annotations", "/home/ai002/Downloads/data/output_set/")
