import os
import pandas as pd


def data_label(path):
    df_label = pd.read_csv('label.csv', header=None)  # 读取所有的label数据，保存为列向量
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


train_path = 'D:\\SentimentClassification\\train'
validate_path = 'D:\\SentimentClassification\\validate'
data_label(train_path)
data_label(validate_path)
