import os
import torch
import torch.utils.data as tud
import numpy as np
import pandas as pd
import cv2 as cv


# 创建数据和标记对照表
def data_label(path):
    df_label = pd.read_csv('initial_data\\label.csv', header=None)  # 读取所有的label数据，保存为列向量
    files = os.listdir(path)  # 获取path目录下的所有文件名
    path_list = []
    label_list = []
    for f in files:
        if os.path.splitext(f)[1] == '.jpg':  # 分离文件名和扩展名
            path_list.append(f)
            index = int(os.path.splitext(f)[0])  # 文件名等于在label中的index
            label_list.append(df_label.iat[index, 0])

    # 写入到csv中
    path_s = pd.Series(path_list)  # 序列化
    label_s = pd.Series(label_list)  # 序列化
    df = pd.DataFrame()  # 实例化对象
    df['path'] = path_s  # header
    df['label'] = label_s  # header
    df.to_csv(path + '\\dataset.csv', index=False, header=False)


# 得到训练集和验证集的数据
train_path = 'D:\\SentimentClassification\\train'
validate_path = 'D:\\SentimentClassification\\validate'
data_label(train_path)
data_label(validate_path)

print('按任意键继续------')
input()
print('-----------------')


class dataset(tud.Dataset):
    # 初始化
    def __init__(self, root):
        super(dataset, self).__init__()

        self.root = root
        df_path = pd.read_csv(root + '\\dataset.csv', header=None, usecols=[0])
        df_label = pd.read_csv(root + '\\dataset.csv', header=None, usecols=[1])

        self.path = np.array(df_path)[:, 0]
        self.label = np.array(df_label)[:, 0]

    # 重写getitem方法
    def __getitem__(self, item):
        face = cv.imread(self.root + '\\' + self.path[item])  # 读取图片
        face_gray = cv.cvtColor(face, cv.COLOR_BGR2GRAY)  # 转换为灰度图

        # 高斯模糊，todo
        # face_Gus = cv.GaussianBlur(face_gray, (3, 3), 0)

        # 直方图均衡化
        face_hist = cv.equalizeHist(face_gray)

        # 像素值标准化到[0,1]区间
        face_normalized = face_hist.reshape(1, 48, 48) / 255.0

        # 转化为训练需要的类型
        face_tensor = torch.from_numpy(face_normalized).type('torch.FloatTensor')

        # label直接利用索引得到
        label = self.label[item]

        return face_tensor, label

    # 重写len方法
    def __len__(self):
        return self.path.shape[0]


# 实例化
train_set = dataset(root='D:\\SentimentClassification\\train')
train_loader = tud.DataLoader(train_set, batch_size=10)  # 利用加载器加载，可分批
for images, labels in train_loader:
    print(len(labels))
