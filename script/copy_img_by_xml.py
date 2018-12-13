import os
import shutil


def copy_img_by_xml(name_path, img_path, out_path):
  '''
  输入：
      name_path:默认为xml文件路径
      img_path:图像路径
      out_path:输出文件路径
  '''

  for filename in os.listdir(name_path):
    filename = filename[:-4]
    path = img_path + '/' + filename + '.jpg'
    print(path)
    shutil.copy(path, out_path)


if __name__ == '__main__':
  copy_img_by_xml(
      "/data/face_data/20181129_14/Annotations", "/data/201808/images", "/data/face_data/20181129_14/JPEGImages")
