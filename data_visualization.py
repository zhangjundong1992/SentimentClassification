import pandas as pd
import cv2 as cv
import numpy as np


# 将训练集的x和y拆分出来
def separate_xy():
    path = 'initial_data\\train.csv'
    df = pd.read_csv(path)
    df_y = df[['label']]
    df_x = df[['feature']]
    df_y.to_csv('initial_data\\label.csv', index=False, header=False)
    df_x.to_csv('initial_data\\data.csv', index=False, header=False)


separate_xy()
print('按任意键继续------')
input()
print('-----------------')


# 将x可视化
def visualize():
    path = './/image//'
    data = np.loadtxt('initial_data\\data.csv')
    for i in range(data.shape[0]):
        face_array = np.reshape(data[i, :], (48, 48))
        cv.imwrite(path + '//' + '{}.jpg'.format(i), face_array)


visualize()
