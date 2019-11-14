from sklearn.decomposition import PCA
import matplotlib.pylab as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


class myPCA:
    def __init__(self, n_components, X, y):
        self.n_components = n_components
        self.eigenvalues = None  # 特征值
        self.cumsum_eigenvalues = None
        self.eigenvectors = None  # 特征向量
        self.w = 28  # 图片宽度
        self.h = 28  # 图片高度
        self.lower_dimensional_data = None  # pca后的低纬度数据
        self.X = X  # 训练数据集X
        self.y = y  # 训练数据集label y
        self.recon_img = None  # pca 还原的图片
        self.pca = None

    def doPCA(self):
        pca = PCA(n_components=self.n_components)
        self.pca = pca
        self.lower_dimensional_data = pca.fit_transform(self.X)
        self.recon_img = pca.inverse_transform(self.lower_dimensional_data)
        print(self.lower_dimensional_data.shape)
        self.eigenvectors = pca.components_
        self.eigenvalues = pca.explained_variance_
        self.cumsum_eigenvalues = torch.cumsum(torch.from_numpy(self.eigenvalues), dim=0)

    def plot(self):
        plt.bar(range(self.n_components), self.eigenvalues, alpha=0.5, align='center', label='individual variances')
        plt.step(range(self.n_components), self.cumsum_eigenvalues, where='mid', label='cumulative variances')

    def showEigenVectorsAsImg(self, number):
        eigenvectors_img = self.eigenvectors.reshape(self.n_components, self.w, self.h)
        plt.figure(figsize=(self.n_components, 5))
        for index, item in enumerate(eigenvectors_img[:number]):
            plt.subplot(1, number, index + 1)
            plt.imshow(item)

    def showReconImg(self, number):
        indices_list = torch.randint(0, self.X.shape[0], (number,))
        for index, item in enumerate(indices_list):
            x_img = self.X[item]
            y_img = self.y[item]
            recon_img = self.pca.inverse_transform(self.lower_dimensional_data[item])
            # show recon img
            plt.subplot(2, number, index + 1)
            plt.title(str(item))
            plt.imshow(recon_img.reshape(self.h, self.w))
            # show origin img
            plt.subplot(2, number, index + number + 1)
            plt.title(str(item))
            plt.imshow(x_img.reshape(self.h, self.w))


class k_means:
    def __init__(self, k, data):
        self.k = k
        self.data = data
        self.center_points = torch.randn(k, data.shape[1])
        self.clusters = {}
        for i in range(k):
            self.clusters[k] = []
        print(self.center_points)

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
        print('do cluster')

    # 计算欧式距离
    def computer_O_diatance(self, data_point, center_point):
        re = torch.mul(torch.from_numpy(data_point), center_point)
        return re


if __name__ == '__main__':

    # prepare data
    mnist_data = datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor())
    train_loader_pca = torch.utils.data.DataLoader(mnist_data, batch_size=4096, shuffle=False)
    X = None
    y = None
    for data in train_loader_pca:
        X, y = data
    X = X.view(-1, 28 * 28)  # X.shape : 4096 * 784
    choose_choose_digits = torch.randint(0, 10, (2,))
    print("the 2 digit we choose:", choose_choose_digits)
    i_list = []
    for i, item in enumerate(y):
        if item in choose_choose_digits:
            i_list.append(i)
    X_2 = X[i_list]
    y_2 = y[i_list]
    # 4b
    my_pca_4 = myPCA(2, X_2, y_2)
    my_pca_4.doPCA()
    my_pca_4.showReconImg(10)

    # 要用的数据在这里
    data = my_pca_4.lower_dimensional_data
    print(data.shape)
    print(data[1].shape)
    e_vectors = my_pca_4.eigenvectors
    print(e_vectors.shape)
    kk = k_means(2, data)
    kk.doCluster()
