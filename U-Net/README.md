# U-Net-liver
基于UNet的肝脏CT分割（数字图像处理大作业）
### 环境的配置：
pytorch==1.2.0
python==3.6.12
skimage==0.19.1
opencv-python==4.4
其他库按需求安装
### train函数所需时间
大约50分钟，可以先执行train()再执行test_1()函数
train()函数的模型保存在weight_19.pth当中
#### 如果报错，可以注释或者取消注释第85行，之后执行
#### train_img
当中保存着测试集数据的效果，图片从左到右分别为：原肝脏CT切片，人工标注的mask，训练出的模型预测效果
