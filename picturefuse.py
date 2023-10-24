# -*- coding: utf-8 -*-
import cv2
import numpy as np

# 这里图像的尺寸必须为2的n次幂
# A = cv2.imread('C:/Users/HP/Desktop/apple.png')
# A = cv2.resize(A, (1024, 1024), interpolation=cv2.INTER_CUBIC)
# B = cv2.imread('C:/Users/HP/Desktop/orange.png')
# B = cv2.resize(B, (1024, 1024), interpolation=cv2.INTER_CUBIC)
#
# A = cv2.imread('./000111.jpg')
# A = cv2.resize(A, (1024, 1024), interpolation=cv2.INTER_CUBIC)
# B = cv2.imread('./000112.jpg')
# B = cv2.resize(B, (1024, 1024), interpolation=cv2.INTER_CUBIC)
# # 生成8层的高斯金字塔gpA
# G = A.copy()
# gpA = [G]
#
# for i in range(7):
#     # 进行7次高斯模糊+下采样
#     G = cv2.pyrDown(G)
#     # 把每次高斯模糊+下采样的结果送给gpA
#     gpA.append(G)
#
# # 生成8层的高斯金字塔gpB
# G = B.copy()
# gpB = [G]
# for i in range(7):
#     # 进行7次高斯模糊+下采样
#     G = cv2.pyrDown(G)
#     # 把每次高斯模糊+下采样的结果送给gpB
#     gpB.append(G)
#
# # 把两个高斯金字塔进行合并
# LR = []
# # zip(lpA,lpB)把两个高斯金字塔各层的两个图像组合成一个元组，然后各元组构成一个大zip
# # 对于各元组中的两个图像
# for la, lb in zip(gpA, gpB):
#     # 取la或lb的尺寸皆可
#     rows, cols, dpt = la.shape
#     # 利用np.hstack将这两个图像“一半一半”地拼接起来
#     # 取la的左边一半和lb的右边一半拼成一个融合后的图，结果赋给ls
#     lr = np.hstack((la[:, 0:cols // 2], lb[:, cols // 2:]))
#     # 两个拉普拉斯金字塔各层图像融合后的结果赋给LS
#     LR.append(lr)
#
# # 用融合后的拉普拉斯金字塔重构出最终图像
# # 初始化ls为融合后拉普拉斯金字塔的最高层
# # 下面的循环结束后ls就是要求的最终结果图像
# lr = LR[7]
# for i in range(6, -1, -1):
#     # 每层图像先上采样，再和当前层的下一层图像相加，结果再赋给ls
#     lr = cv2.pyrUp(lr)
#     lr = cv2.add(lr, LR[i])
# # ---------------------------------------------------------------------------------------------------
# # 生成8层拉普拉斯金字塔
# # 从顶层开始构建
# # 顶层即高斯金字塔的顶层
# lpA = [gpA[7]]
# # 7 6 5 4 3 2 1
# for i in range(7, 0, -1):
#     # 从顶层开始，不断上采样
#     GE = cv2.pyrUp(gpA[i])
#     # 用下一层的高斯减去上层高斯的上采样
#     L = cv2.subtract(gpA[i - 1], GE)
#     # 结果送给拉普拉斯金字塔
#     lpA.append(L)
#
# lpB = [gpB[7]]
# for i in range(7, 0, -1):
#     GE = cv2.pyrUp(gpB[i])
#     L = cv2.subtract(gpB[i - 1], GE)
#     lpB.append(L)
#
# # 把两个拉普拉斯金字塔进行合并
# LS = []
# # zip(lpA,lpB)把两个拉普拉斯金字塔各层的两个图像组合成一个元组，然后各元组构成一个大zip
# # 对于各元组中的两个图像
# for la, lb in zip(lpA, lpB):
#     # 取la或lb的尺寸皆可
#     rows, cols, dpt = la.shape
#     # 利用np.hstack将这两个图像“一半一半”地拼接起来
#     # 取la的左边一半和lb的右边一半拼成一个融合后的图，结果赋给ls
#     ls = np.hstack((la[:, 0:cols // 2], lb[:, cols // 2:]))
#     # 两个拉普拉斯金字塔各层图像融合后的结果赋给LS
#     LS.append(ls)
#
# # 用融合后的拉普拉斯金字塔重构出最终图像
# # 初始化ls为融合后拉普拉斯金字塔的最高层
# # 下面的循环结束后ls就是要求的最终结果图像
# ls = LS[0]
# for i in range(1, 8):
#     # 每层图像先上采样，再和当前层的下一层图像相加，结果再赋给ls
#     ls = cv2.pyrUp(ls)
#     ls = cv2.add(ls, LS[i])
#
# with_pyramid = lr + ls
#
# # 不用金字塔融合，直接生硬地连接两幅原始图像
# without_pyramid = np.hstack((A[:, :cols // 2], B[:, cols // 2:]))
#
# # 对比一下结果
# cv2.imshow("with_pyramid", with_pyramid)
# cv2.imshow("without_pyramid", without_pyramid)
#
# # 按任意键关闭所有窗口
# cv2.waitKey()
# cv2.destroyAllWindows()

import cv2
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os
import matplotlib.pyplot as plt
import re
from glob import glob
from slic_segmentation import *

save_path ="./VOCROSD/ROSDJPEGImagestest"# 保存文件的路径

path1 = "G:/1.course/myproject/3saliency/Saliency-Extraction-master/VOCROSD/JPEGImages"
path2 = "G:/1.course/myproject/3saliency/Saliency-Extraction-master/VOCROSD/JPEGImages1"
# path1 = "G:/1.course/myproject/3saliency/faster_rcnngai/VOCdevkit/DIOR/JPEGImages-test"
# path2 = "G:/1.course/myproject/3saliency/faster_rcnngai/VOCdevkit/DIOR/JPEGImages-saliency-trainval"
frames1 = glob(os.path.join(path1, '*.jpg'))
frames2 = glob(os.path.join(path2, '*.jpg'))
def get_saliency_rbd(img_path1,img_path2):
    img1 = cv2.imread(img_path1)
    img2 = cv2.imread(img_path2)

    result = cv2.addWeighted(img1, 0.7, img2, 0.3, 0)
    return result

if __name__ == '__main__':
#     # 批量操作
#     #
#     # AA = []
#     # # AA =list()
#     # BB = []
#     # # BB = list()
#     #
    for i, frame1 in enumerate(frames1):
        pattern = re.compile(r'([^<>/\\\|:""\*\?]+)\.\w+$')
        data = pattern.findall(frame1)
        data_now = data[0]
        Newdir = os.path.join(path1, str(data_now) + '.jpg')
        # img = cv2.imread(Newdir)
        # cv2.waitKey(200)


    for i, frame2 in enumerate(frames2):
        pattern1 = re.compile(r'([^<>/\\\|:""\*\?]+)\.\w+$')
        data1 = pattern1.findall(frame2)
        data_now1 = data1[0]
        Newdir1 = os.path.join(path2, str(data_now1) + '.jpg')
        # img1 = cv2.imread(Newdir1)
        cv2.waitKey(100)


        rbd = get_saliency_rbd(frames1[i], frames2[i]).astype('uint8')

    # image_np = save_saliency_rbd(rbd)
        name = str(data_now1) + ".jpg"
        cv2.imwrite(save_path + '/' + name, rbd)
        print(frames1[i])
        print(frames2[i])
        print(name)


# 显示图像
# cv2.imshow("src1", src1)
# cv2.imshow("src2", src2)
# cv2.imshow("result", result)

# 等待显示
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
#
#
# # 读取图片
# src1 = cv2.imread('./00003.jpg')
# src2 = cv2.imread('./000033.jpg')
#
# # 图像融合
# result = cv2.addWeighted(src1, 0.7, src2, 0.3, 0)
#
# # 显示图像
# cv2.imshow("src1", src1)
# cv2.imshow("src2", src2)
# cv2.imshow("result", result)
#
# # 等待显示
# cv2.waitKey(0)
# cv2.destroyAllWindows()
