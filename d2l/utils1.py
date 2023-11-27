import os
import math
import numpy as np
import numpy.linalg as la

def processData(istrain):
    data_paths = []
    # 统计每个TXT的行数
    COUNTS = []
    if istrain:
        data_root = '/data/zcf/code/pytorch/d2l/dataset'
    else:
        data_root = '/data/zcf/code/pytorch/d2l/test'
    data_classification = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13']
    # data_classification = ['0', '1']
    for i in data_classification:
        data_paths.append(os.path.join(data_root, i))
    print(data_paths)

    for idx, file_dir in enumerate(data_paths):
        txt_names = get_listdir(file_dir)
        txt_names.sort()
        print(txt_names)
        # 处理数据，并写入到新文件中
        COUNT = processFile(txt_names, istrain)
        COUNTS.append(COUNT)
        # txt_path = predataset(txt_names, istrain)
        # txt_paths.append(txt_path)
    # print(txt)
    print(COUNTS)
    # return COUNTS


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

def processFile(txt_names, istrain):
    """
    处理数据，计算方向角、加速度、曲率。并为每一行添加标签，将输入整合入一个TXT文件
    :param txt_names: 一个分类下的文件集合
    :param istrain: 是否训练
    :return: 每一个分类下面各txt数据的行数
    """
    # txt_paths = []
    # 统计每一个txt的数量
    COUNT =[]
    # 获取每个txt文件中的全部数据
    for txt_name in txt_names:
        # 把处理后的数据放到新的txt文件里面
        str3 = '/'
        str1 = ' '
        str2 = '\n'
        tempPathFirst = txt_name.split('/')[1:-3]
        # 获取数据的类别
        tempPathSecond = txt_name.split('/')[-2]
        newTxtName = 'new.txt'
        if istrain:
            newTxtPath = os.path.join('/', str3.join(tempPathFirst), 'preTxt')
        else:
            newTxtPath = os.path.join('/', str3.join(tempPathFirst), 'pretestTxt')
        if not os.path.exists(newTxtPath):
            os.makedirs(newTxtPath)
        newTxtName = os.path.join(newTxtPath, newTxtName)

        with open(txt_name, 'r') as file:
            temp = []
            for line in file:
                line = line.strip('\n')
                line = line.rstrip()
                words = line.split( )
                words = [words[0], float(words[1]), float(words[2]), float(words[3]), float(words[4])]
                temp.append(words)
        data = calculate(temp)
        # 将计算之后的数据输入写入到新文件里面
        templist = []
        for d in data:
            for i in range(len(d)):
                d[i] = str(d[i])
            d.append(str(tempPathSecond))
            if is_float_zero(float(d[0])):
                continue
            templist.append(str1.join(d))
        # 如果txt文件处理之后的数据为空
        if len(templist) == 0 or len(templist) == 1:
            continue
        # txt_paths.append(newTxtName)
        COUNT.append(len(templist))
        # if os.path.exists(newTxtName):
        #     os.remove(newTxtName)
        newfile = open(newTxtName, 'a')
        if os.path.getsize(newTxtName) != 0:
            newfile.write('\n')
        newfile.write(str2.join(templist))
        newfile.close()
        # print(data)
    # return txt_paths
    return COUNT


def is_float_zero(value, tolerance=1e-10):
    return abs(value) < tolerance

def calculate(data_list):
    new_datalist = []
    z_list = [1, 10, 1, 0.1, 1]
    for i in range(1, len(data_list) - 1):
        forWord = data_list[i - 1]
        word = data_list[i]
        nextWord = data_list[i+1]

        if(i == 1):
            data_list[0].append(round(calculate_normal(word[-1] - forWord[-1], z_list[2]), 3))
            time = float(word[0][1:]) - float(forWord[0][1:])
            speed = (word[3] - forWord[3]) * 0.5144444
            a = speed / time
            a = calculate_normal(a, z_list[3])
            data_list[0].append(round(a, 5))
            data_list[0].append(0)

            data_list[0][3] = round(calculate_normal(data_list[0][3], z_list[0]), 3)
            data_list[0][4] = round(calculate_normal(data_list[0][4], z_list[1]), 3)
            new_datalist.append(data_list[0][3:])


        # 计算转向角
        corner = nextWord[-1] - word[-1]
        corner = calculate_normal(corner, z_list[2])
        # 讲计算结果保留三位小数点并保存在列表中
        data_list[i].append(round(corner, 3))

        # 计算两个航迹点之间的时间间隔
        t1 = word[0][1:]
        t2 = nextWord[0][1:]
        timeDiff = float(t2) - float(t1)

        # 计算两个航迹点之间的速度差
        v1 = word[3]
        v2 = nextWord[3]
        # 这里的速度单位是 节  转化为m/s
        speedDiff = (v2 - v1) * 0.5144444

        # 计算加速度
        a = speedDiff / timeDiff
        a = calculate_normal(a, z_list[3])
        data_list[i].append(round(a, 3))


        # 计算曲率  参考https://zhuanlan.zhihu.com/p/72083902
        kappa = curvature([forWord[1], word[1], nextWord[1]], [forWord[2], word[2], nextWord[2]])
        kappa = calculate_normal(kappa, z_list[4])
        data_list[i].append(round(kappa, 3))

        data_list[i][3] = round(calculate_normal(data_list[i][3], z_list[0]), 3)
        data_list[i][4] = round(calculate_normal(data_list[i][4], z_list[1]), 3)
        new_datalist.append(data_list[i][3:])

        if(i == len(data_list) - 2):
            data_list[-1].append(0)
            data_list[-1].append(0)
            data_list[-1].append(0)
            new_datalist.append(data_list[-1][3:])

    return new_datalist


def calculate_normal(x, y):
    # 计算表达式 z = 10 * log10((|x| + 1) / y)
    z = 10 * math.log10((abs(x) + 1) / y)
    return z

# 计算曲率
def curvature(x, y):
    t_a = la.norm([x[1] - x[0], y[1] - y[0]])
    t_b = la.norm([x[2] - x[1], y[2] - y[1]])

    M = np.array([
        [1, -t_a, t_a ** 2],
        [1, 0, 0],
        [1, t_b, t_b ** 2]
    ])

    if(t_a == 0 or t_b == 0):
        return 0

    a = np.matmul(la.inv(M), x)
    b = np.matmul(la.inv(M), y)

    kappa = 2 * (a[2] * b[1] - b[2] * a[1]) / (a[1] ** 2. + b[1] ** 2.) ** (1.5)
    return kappa

if __name__ == '__main__':
    processData(istrain=True)



