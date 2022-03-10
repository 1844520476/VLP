# VLP

## 项目介绍
VLP是一个多模态的视觉语言项目：

1.CLIP:clip原生程序，clip模型是基于自然语言监督信号的迁移视觉模型

2.label:文本标签

3.Input：待识别的图片

4.notebook：自动标注相关调研，gpt-neo的demo，clip的colab实现以及相关的ipynb笔记

5.Weights：模型权重

## 模块
1.Detect.py:检测程序
2.ZeroShot.py:（针对Cifar10）的zero-shot程序
3.dehaze.py:去雾模块


ps.如果您觉得配置环境实在不好搞，不要怕，以下是配置好的conda环境（前提是您要安装有Anaconda3）:
clip.zip（百度网盘链接：）

1.解压clip.zip到Anaconda3\envs中

2.将Anaconda3\envs\clip\python.exe配置为项目的python解释器地址

3.运行程序前记得切换环境到clip

## 【初稿】3.10日更新

1.增加dehaze功能（现可直接调用，也可单独使用）

2.增加（针对Cifar10）的zero-shot功能

3.解决了detect重复2调用性能下降的问题（detect函数也已经封装完成）
