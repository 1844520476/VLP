# python解释器地址
# ! D:\DLSoftware\Anaconda3\envs\clip\python.exe
"""
1.CLIP源代码：https://github.com/OpenAI/CLIP.
2.请在终端中运行下面两行代码（如果是在ipynb环境下运行：请在pip前加上！）
    pip install ftfy regex tqdm
    pip install git+https://github.com/openai/CLIP.git
3.国内镜像源：(1)清华镜像:pip install [The Package You Want to Download] -i https://pypi.tuna.tsinghua.edu.cn/simple
4.刘同学于2022.02增补
"""

import time
import datetime

from prettytable import PrettyTable
import torch
import CLIP
from PIL import Image

from Label.emotional import *
from Label.imagenet import *
from Label.coco128 import *
from Label.cifar10 import *
from dehaze import quwu


# 主程序
def main(WeightsPath):
    print(f'-----------------------------欢迎使用基于自然语言监督信号的迁移视觉模型识别系统--------------------------------')

    # 待测数据集标签选择
    label_exist = (emotion_label, coco128_label, imagenet_label, cifar10_label)
    label_dict = {'0': 'emotion', '1': 'coco128', '2': 'imagenet', '3': 'cifar10'}
    # 打印所有已存在的标签
    print(f'可选择的【文本】标签：')
    for i in label_dict:
        print(f'[{int(i) + 1}]:{label_dict[i]}_label')
        # label_dict[i] += '_dict'# list indices must be integers or slices, not str(注意此类错误)

    # 选择标签
    def LabelNum():
        while True:
            Label_num = int(input('请输入对应的【数字编号】：'))
            Label_num -= 1
            if 0 <= Label_num < len(label_dict):
                Label = label_exist[Label_num]
                print(f'\n您选择的【文本标签详情】如下:\n{Label}')
                return Label, Label_num
                break

    Label, Label_num = LabelNum()
    # 初始化存储概率信息的字典
    dict_prob = {}

    def TextAdd():
        """
        使用add可以方便的对文本信息进行统一修改，
        使格式与训练数据集中的文本图像对更匹配
        """
        text_add = input('\n需要对【文本前后增补】吗？[y/n]:')
        if text_add == 'y':
            print(f'示例：add1 + Label_list[i] + add2（记得打前后空格）')
            add1 = input('add1:')
            add2 = input('add2:')
            return add1, add2
        else:
            add1 = ''
            add2 = ''
            return add1, add2

    # 对原始标签进行增补操作
    add1, add2 = TextAdd()

    # 测试并选择设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 可选模型
    model_dict = {0: 'RN50', 1: 'RN101', 2: 'RN50x4', 3: 'RN50x16', 4: 'RN50x64', 5: 'ViT-B/32', 6: 'ViT-B/16',
                  7: 'ViT-L/14'}

    # 模型选择
    def modelchose():
        while True:
            print(f'\n可选模型：{model_dict}')
            model_num = int(input('请输入您中意的模型的【数字编号】:'))
            if 0 <= Label_num < len(model_dict):
                model = model_dict[model_num]
                return model
                break

    # 模型选择
    model = modelchose()
    print(f'本次识别使用的【网络模型】为：{model}'
          f'\n.................................网络加载中............................................')

    # 加载模型与预处理
    model, preprocess = CLIP.load(model,
                                  device=device, download_root=WeightsPath)

    # 确定待配对文本信息
    Label_list = list(Label.values())
    # 提前定义列表
    text_input = []

    for i in range(len(Label_list)):
        # 将统一修改后的文本添加到新链表中
        new_label_list = add1 + Label_list[i] + add2
        text_input.append(new_label_list)

    # 图像识别函数
    def detect(img_path):
        # 计时器：开始计时
        start_time = time.time()

        # 加载待识别图片
        image = preprocess(Image.open(img_path)). \
            unsqueeze(0).to(device)

        # 读取文本链表
        text = CLIP.tokenize(text_input).to(device)

        # 开始推理过程
        '''
        torch.no_grad() 是一个上下文管理器，被该语句 wrap 起来的部分将不会track 梯度
        即：被此上下文管理器包裹起来计算部分：可以执行计算，但该计算不会在反向传播中被记录
        '''
        with torch.no_grad():
            # 提取文本特征信息到向量空间
            model.encode_image(image)
            # 提取图像特征信息到向量空间
            model.encode_text(text)
            # 计算语义向量与图片特征向量
            logits_per_image, logits_per_text = model(image, text)
            # 计算两者的相似度（计算结果以链表形式按顺序存储）
            probs = logits_per_image.softmax(dim=-1). \
                cpu().numpy()
            probs = probs.tolist()
            # 结束计时
            Time = time.time() - start_time

            # 后处理，打印对应概率
            for i in range(len(text_input)):
                # 以一维list存储了对应text_input的预测概率
                prob = probs[0]
                # 将文本和相似度分别存储为key和value
                prob[i] = round(prob[i] * 100, 3)
                dict_prob[text_input[i]] = prob[i]

            # 最有可能的预测结果
            MAX = sorted(dict_prob,
                         key=dict_prob.get,
                         reverse=True)[0]

            print(f'\n最有可能的结果是【{MAX}】,'
                  f'有{dict_prob[MAX]:.2f}%的可能性.'
                  f'\n(检测过程耗时{Time:.2f}秒)\n')  # 预测耗时

            # 画个表格，让输出结果直观漂亮点
            table = PrettyTable(['序号',
                                 '类别',
                                 '预测概率'])

            # 确定最大显示列数
            def length():
                while True:
                    n = int(input('显示top-n的预测情况[请输入整数以确定n]：'))
                    if n <= 0:
                        print('请输入大于0的数字')
                    elif n > len(label_exist[Label_num]):
                        print(f'请输入小于文本标签长度的数字'
                              f'(标签长度为【{len(label_exist[Label_num])}】)')
                    else:
                        print(f'\n将显示top-{n}的预测概率')
                        return n
                        break

            # 按value的大小排序
            table_length = length()
            print(f'\n针对【图片:{img}】的【识别结果】为:')

            # 按概率大小的顺序翻译
            dict_cn = {}
            for i in range(table_length):
                prob_max_i = sorted(dict_prob,
                                    key=dict_prob.get,
                                    reverse=True)[i]
                probability = str(dict_prob[prob_max_i]) + '%'

                # 将英文标签转化为中文
                label_cn_dict = {0: emotional_dict, 1: '', 2: '', 3: cifar10_dict}  # 标签的中文翻译（以字典形式存储）
                label_cn_dict = label_cn_dict[Label_num]  # 根据标签选择确定对应的中文翻译字典
                if label_cn_dict != '':
                    # 将翻译后的key映射回翻译前的key
                    text_i = text_input.index(prob_max_i)
                    prob_max_i = Label_list[text_i]

                    prob_max_i_cn = label_cn_dict[prob_max_i] + '(' + prob_max_i + ')'
                    dict_cn[prob_max_i] = prob_max_i_cn
                else:
                    prob_max_i_cn = prob_max_i
                    dict_cn[prob_max_i] = prob_max_i_cn

                # 画表格
                table.add_row([i + 1,
                               prob_max_i_cn,
                               probability])

            # 打印预测概率top-n表格
            print(f'{table}')

    while True:
        # 图片选择或退出系统
        photo = input('\n(输入exit可退回重选文本标签)'
                      '\n请输入【图片名称】以选择【待检测图片】：')
        print('\n.................................检测中............................................')
        # TODO 功能1：完善图片检索功能与验证机制
        if photo != 'exit':
            # 图片地址
            img = photo + '.jpg'
            img_path = r'Input/emotion/' + img

            if input('是否去雾:[y/n]') == 'y':
                img_path = quwu(photo)
            # TODO 功能二：增加目标检测算法与裁切功能：1.去雾算法去除雾气干扰2.目标检测算法检测车辆位置，随后将裁切的图片交给Clip 3.Clip进行细分类
            detect(img_path)
        else:
            break


# 本地时间
def localTime():
    # 打印系统时间
    define = '%m月%d日 %H时%M分'
    system_time = datetime.datetime.now().strftime(define)
    print(f'现在是北京时间：{system_time}')


# 执行主程序
if __name__ == '__main__':
    while True:
        localTime()
        # 模型存储地址
        weightspath = r'weights'
        # 主程序
        main(weightspath)
        # 退出与否
        exit = input('是否退出系统:[y/n]')
        if exit == 'y':
            print(f'-----------------------------期待您再次使用系统，再见>_<！--------------------------------')
            break
        else:
            print('即将返回主界面')
