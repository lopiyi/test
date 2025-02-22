# 测试环境

| |版本号|
|----------------|--------------------------------|
|VS|`2019`|
|Qt|`5.15` |
|opencv|`4.8` |
|cuda|`11.8`|
|tensorrt|`8.5.3.1`|
|openvino|`2023.3.0`|

# 如何编译
用vs打开后修改属性表中的包含目录为自己的目录。

## 如何使用GPU编译

 1. 配置好cuda和tensorrt的属性表。
 2. 修改customview.h中关于CPU的宏定义为false，移除PPOcrVino.h和PPOcrVino.cpp。
 3. GPU模式首次识别会生成模型，需要比较长的时间，耐心等待。
 
## 如何使用CPU编译
 1. 配置好openvino的属性表
 2. 修改customview.h中关于CPU的宏定义为true，移除PPOcrTrt.h和PPOcrTrt.cu。

# 软件使用
打开软件后加载图像，鼠标滚轮放大缩小，按住鼠标左键可拖动，同时按住Ctrl和鼠标左键可画框，松开鼠标后识别框内的文字
