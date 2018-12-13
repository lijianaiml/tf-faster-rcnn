# 源码解析
## 目录
### ./data
各种数据，包括运行demo和train的pre-trained models，coco和PASCAL的数据或其软链接，demo图片，自定义数据
### ./output
框架的训练输出（比如：res101_faster_rcnn_iter_70000.ckpt）的存放目录
### ./experiments
训练及测试的脚本
### ./lib
各种库
常用库：datasets（框架的IO操作）
未完待续...
### ./tools
各种工具，包括demo.py
### ./tensorboard
tb输出目录
### ./script
自定义的数据预处理脚本
## 常用函数
#### numpy.hstack(tup)
返回按行堆叠的数组，与numpy.vstack相反。对pixel数据（h,w,c）最有意义？
#### EasyDict
from easydict import EasyDict
EasyDict的作用：可以使得以属性的方式去访问字典的值
## demo
##### ./tools/demo.py
运行demo之前需要下载pre-trained models
运行：./data/scripts/fetch_faster_rcnn_models.sh 会自动下载模型到./data/voc_2007_trainval+voc_2012_trainval
为model创建软连接，连接到./output/(参考git)
## train
##### ./experiments/scripts/train_faster_rcnn.sh 0 pascal_voc res101

### 注意:
1. 训练之前需要下载pre-trained models及其weights，目前支持VGG16和ResnetV1.下载后放在./data/imagenet_weights
注:
2. 训练前需要将./data/cache及./output(可提前备份)删除
3. 如果训练数据的种类变化,需要修改./lib/datasets/pascal_voc.py中的_CLASS种类
4. 运行demo.py,加载的模型需要与net.create_archeitecture()方法的类别对应起来,否则会报saver.restore()错误