from sklearn.decomposition import PCA
import matplotlib.pylab as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


class k_means:

    def __init__(self, k, data):
        self.k = k
        self.data = data
        self.center_points = torch.randn(k, data.shape[1])
        self.clusters = {}
        for i in range(k):
            self.clusters[i] = []

        # 做数据分类

    def doCluster(self):
        for i in self.data:
            distance = []
            for j in self.center_points:
                distance.append(self.computer_O_diatance(i, j))
            # 求最小距离下标，分类
            max_d = max(distance)
            max_index = distance.index(max_d)
            # 把这个点分到对应的cluster
            self.clusters[max_index].append(i)
        # while !converge:

        print('do cluster')
        print(len(self.clusters[0]))
        print(len(self.clusters[1]))

        # 计算欧式距离

    def computer_O_diatance(self, data_point, center_point):
        re = data_point @ center_point.numpy()
        return re

    def reassign_center(self):
        # centers = self.
        return 0
