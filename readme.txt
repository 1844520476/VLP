NTVT是我随便取得，因为clip模型是基于自然语言监督信号的迁移视觉模型
CLIP:clip原生程序
datadet_label:文本标签
Input：待识别的图片
notebook：自动标注相关调研，以及clip的colab实现
Weights：模型权重（bpe_simple_vocab_16e6）
main.py:主程序

如果您觉得配置环境实在不好搞，不要怕，以下是配置好的conda环境（前提是您要安装有Anaconda3）:
clip.zip（百度网盘链接：）
1.解压clip.zip到Anaconda3\envs中
2.将Anaconda3\envs\clip\python.exe配置为项目的python解释器地址
3.直接运行main.py或在终端中（记得切换环境到clip）输入python main.py

3.10日更新
1.增加dehaze功能（现可直接调用，也可单独使用）
2.增加（针对Cifar10）的zero-shot功能
3.解决了detect重复2调用性能下降的问题（detect函数也已经封装完成）