import os
import shutil
import xml.etree.ElementTree as ET
import sys

dic_all = {}


def listDir(root_dir, out_dir):
  '''
  输入：
      root_dir:xml文件根目录，可包含不同类别的xml文件夹，不同文件夹的xml文件可同名
      out_dir:合并后的xml输出路径
  输出：

  '''
  for filename in os.listdir(root_dir):
    pathname = os.path.join(root_dir, filename)
    pathname_output = os.path.join(out_dir, filename)
    if os.path.isfile(pathname):
      if filename in dic_all:
        merge(dic_all.get(
            filename), pathname)  # 文件重复先执行merge再拷贝
      else:
        # 文件不重复直接拷贝
        shutil.copy(pathname, out_dir)

      # 更新dic_all为输出文件夹中的融合文件
      dic_all[filename] = pathname_output
    else:
      listDir(pathname, out_dir)


def merge(outputfname, nwefname):
  # 解析output文件夹对应的文件
  tree_out = ET.parse(outputfname)
  tree_new = ET.parse(nwefname)
  root_out = tree_out.getroot()
  root_new = tree_new.getroot()
  for obj in root_new.findall('object'):
    if obj is not None:
      root_out.append(obj)

  tree_out.write(outputfname, "utf-8")


if __name__ == "__main__":
  listDir("/home/ai002/Downloads/data/CUSTOM",
          "/home/ai002/Downloads/data/output")
  # print('usage: python3 merge.py  xml_path_bemerged xml_path_out')
  # listDir(sys.argv[1], sys.argv[2])
