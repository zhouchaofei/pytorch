import os

import matplotlib.pyplot as plt
import numpy as np


def get_listdir(file_dir):
    """
    获取一个文件夹下面的所有文件
    :param file_dir:文件夹路径
    :return:所有文件路径的集合
    """
    txt_paths = []
    for root, dir_names, file_names in os.walk(file_dir):
        for file_name in file_names:
            txt_paths.append(os.path.join(root, file_name))
    return txt_paths

def plots(txt_names):
    for txt in txt_names:
        with open(txt, 'r') as file:
            lines = file.readlines()

        x = []
        y = []
        for line in lines:
            data = line.split(' ')
            x.append(float(data[1]))
            y.append(float(data[2]))

        fig, ax = plt.subplots()

        ax.plot(x, y, 'o-')

        # ax.set_title('航迹图')
        # ax.set_xlabel('X-经度')
        # ax.set_ylabel('y-维度')

        plt.axis('off')  # 去坐标轴
        plt.xticks([])  # 去刻度
        plt.yticks([])  # 去刻度

        str1 = '/'
        str2 = '.'
        tempPathFirst = txt.split('/')[1:-3]
        tempPathSecond = txt.split('/')[-2]
        tempPathLast = txt.split('/')[-1]
        tempPathLast = tempPathLast.split('.')[0] + '.jpg'
        imgPath = os.path.join('/', str1.join(tempPathFirst), 'image', tempPathSecond)
        if not os.path.exists(imgPath):
            os.makedirs(imgPath)
        imgPath = os.path.join(imgPath, tempPathLast)
        plt.savefig(imgPath)
        # plt.show()

def main():
    data_root = '/data/zcf/code/pytorch/d2l/dataset'
    data_classification = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13']
    # data_classification = ['0', '1']
    data_paths = []
    for i in data_classification:
        data_paths.append(os.path.join(data_root, i))
    print(data_paths)
    for idx, file_dir in enumerate(data_paths):
        txt_names = get_listdir(file_dir)
        txt_names.sort()
        print(txt_names)
        plots(txt_names)


if __name__ == '__main__':
    main()
