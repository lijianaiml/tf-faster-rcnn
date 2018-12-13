# Train Diary
## day1 
2018.11.19 18:00 -- 2018.11.20 10:00
### parameter
traindata:2712
iter:62180
ALES '[8,16,32]' ANCHOR_RATIOS '[0.5,1,2]' TRAIN.STEPSIZE '[50000]
### record
增加镜像增强数据训练,未完成训练,62180/70000处停止,保存60000处模型
iter: 60000 / 70000, total loss: 0.353293 lr: 0.000100
iter: 62180 / 70000, total loss: 0.352573 lr: 0.000100

## day2
使用服务器训练100000轮,其他超参数未调,
loss:0.45

原因可能是使用了融合数据,新融合的数据有问题,手动数据增强后feed给网络

数据备份:/data/face_data/20181129_14
对应输出:服务器109

#day3
使用未融合数据,train:val:test=4:1:1,

loss:0.39  MAP:0.4607

数据备份:/data/face_data/20181129_14
对应输出:/home/ai002/dev/gitclone/tf-faster-rcnn/output_b3

#day4
train:val:test=7:1:2,
C.TRAIN.BATCH_SIZE = 256

loss: 0.448641   MAP:0.5076

数据备份:/data/face_data/20181201(只更新了imageset,其他引用20181129_14)
对应输出:/home/ai002/dev/gitclone/tf-faster-rcnn/output_b4

#day4
train:val:test=7:1:2,
C.TRAIN.BATCH_SIZE = 128
根据训练类别更改了PASCAL_voc.py的类别代码

loss: 0.3933   MAP:0.6047

数据备份:/data/face_data/20181201(只更新了imageset,其他引用20181129_14)
对应输出:/output(109)

#day 5 
train:val:test=7:1:2,
C.TRAIN.BATCH_SIZE = 256,
C.TRAIN.GAMMA = 0.08

loss: 0.460663  MAP:0.5265

数据备份:/data/face_data/20181201(只更新了imageset,其他引用20181129_14)
对应输出:/output_b5









