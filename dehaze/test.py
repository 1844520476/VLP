# import numpy as np
# import torch
# from torch import nn
# from d2l import torch as d2l
#
# avail = torch.cuda.is_available()
# count = torch.cuda.device_count()
# print(f'{count}')
# print(f'{avail}')
#
# print(torch.__version__)
#
# #不行，view只能转化成总元素不变
# # a = torch.ones(1,3,32,32)
# # b = a.view(1,3,224,224)  # 将一个32x32的Tensor变成224x224的Tensor
# # print(a)
# # print(b)
#
# # a = np.array([3, 4, 5])
# # b = np.array([0, 0, 0])
# #
# # t1 = torch.from_numpy(a)
# # t2 = torch.from_numpy(b)
# # print('t1:', t1)
# # print('------------------------')
# # print('t2:', t2)
# #
# # res = torch.cat((t1, t2), 0)
# # print(res)
#
# """
# 考虑用转置卷积实现
# """
#
# # def trans_conv(X, K):
# #     h, w = K.shape
# #     Y = torch.zeros((X.shape[0] + h - 1, X.shape[1] + w - 1))
# #     for i in range(X.shape[0]):
# #         for j in range(X.shape[1]):
# #             Y[i: i + h, j: j + w] += X[i, j] * K
# #     return Y
# #
# # num = 7
# # for i in range(num):
# #     X = trans_conv(X, K)
#
# # X = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
# # num = 7
# # for i in range(num):
# #     n=i+2
# #     # 生成转置卷积核
# #     a = torch.arange(start=1, end=n+1, step=1)
# #     b = torch.arange(start=n+1, end=2*n+1, step=1)
# #     K = torch.stack([a, b], dim=0)
# #     #转化为图片对应维度（batch_size,RGB,H,W）
# #     X, K = X.reshape(1, 1, n, n), K.reshape(1, 1, n, n)
# #     tconv = nn.ConvTranspose2d(n, n, kernel_size=n, bias=False)
# #     tconv.weight.data = K
# #     X = tconv(X)
# #
# # print(f'K:{K}\nX:{X}')
#
# # X = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
# #
# # a = torch.arange(start=0., end=2., step=1)
# # b = torch.arange(start=2., end=4., step=1)
# # K = torch.stack([a, b], dim=0)
# # print(f"K1:{K}")
# # K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
# # print(f"K2:{K}")
#
# # X, K = X.reshape(1, 1, 2, 2), K.reshape(1, 1, 2, 2)
# #
# # tconv = nn.ConvTranspose2d(2, 2, kernel_size=2, bias=False)
# # tconv.weight.data = K
# #
# # Y = tconv(X)
# # print(f'Y:{Y}')
#
#
#
# # def tconv32to224(X):
# #     a = torch.arange(start=0., end=2., step=1)
# #     b = torch.arange(start=2., end=4., step=1)
# #     K = torch.stack([a, b], dim=0)
# #     print(f"K:{K}")
# #     X, K = X.reshape(1, 1, 2, 2), K.reshape(1, 1, 2, 2)
# #     tconv = nn.ConvTranspose2d(2, 2, kernel_size=2, bias=False)
# #     tconv.weight.data = K
# #     return tconv(X)
#
#
#
#
# # a = torch.arange(start=0., end=32., step=1)
# # b = torch.arange(start=32., end=65., step=1)
# # X = torch.stack([a, b], dim=0)
# #
# # print(f'X1:{X}')
# #
# # def tconv32to224(X):
# #     a = torch.arange(start=0., end=7., step=1)
# #     b = torch.arange(start=7., end=14., step=1)
# #     K = torch.stack([a, b], dim=0)
# #     print(f"K:{K}")
# #     X, K = X.reshape(1, 1, 32, 32), K.reshape(1, 1, 7, 7)
# #     tconv = nn.ConvTranspose2d(2, 2, kernel_size=7, bias=False)
# #     tconv.weight.data = K
# #     return tconv(X)
# # print(f'X2:{tconv32to224(X)}')
#
# X = torch.randn(1, 3, 2, 2)
# # print(f'X1:{X}')
# # def tconv32to224(X):
# #     K = torch.ones(1, 3, 2, 2)
# #     print(f"K:{K}")
# #     tconv = nn.ConvTranspose3d(1,1,kernel_size=2,padding=2, bias=False)
# #     tconv.weight.data = K
# #     return tconv(X)
# # print(f'X2:{tconv32to224(X)}')
# print(f'X:{X}')
# upsam = nn.Upsample(scale_factor=7, mode='bilinear')
# imgs = upsam(X)
# print(f"imgs:{imgs}")
from dehaze310 import quwu

quwu('2')