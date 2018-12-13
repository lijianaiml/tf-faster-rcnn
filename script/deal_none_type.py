import os
from xml_helper import parse_xml

root_path = '/home/ai002/Downloads/data/output'
# pic_path = '/home/ai002/dev/gitclone/tf-faster-rcnn/data/VOCdevkit2007/VOC2007/JPEGImages'
for parent, _, files in os.walk(root_path):
  for file in files:
    xml_path = os.path.join(parent, file)
    coords = parse_xml(xml_path)
    if coords is None:
      img_path = pic_path + file[:-4] + 'jpg'
      print()
      if os.path.exists(xml_path):
        print('delete xmlfile :%s' % file)
        os.remove(xml_path)
      # if os.path.exists(img_path):
      #   print('delete imgfile :%s' % img_path)
      #   os.remove(img_path)

# list_ = []
# for parent, _, files in os.walk(root_path):
#   for file in files:
#     list_.append(file[:-4])

# list__ = []
# for parent, _, files in os.walk(pic_path):
#   for file in files:
#     if file[:-4] not in list_:
#       list__.append(file[:-4])

# with open('/data/nonetype.txt', 'w') as out:
#   for item in list__:
#     out.writelines(item + '\n')
