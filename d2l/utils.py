import os
import math
import numpy as np
import numpy.linalg as la


def get_listdir(file_dir):
    txt_path = []
    for root, dir_names, file_names in os.walk(file_dir):
        for file_name in file_names:
            txt_path.append(os.path.join(root, file_name))
    return txt_path

def predataset(txt_names, istrain):
    txt_paths = []
    # 获取每个txt文件中的全部数据
    for txt_name in txt_names:
        # 把处理后的数据放到新的txt文件里面
        str3 = '/'
        str1 = ' '
        str2 = '\n'
        tempPathFirst = txt_name.split('/')[1:-3]
        tempPathSecond = txt_name.split('/')[-2]
        tempPathLast = txt_name.split('/')[-1]
        if istrain:
            newTxtPath = os.path.join('/', str3.join(tempPathFirst), 'predataset', tempPathSecond)
        else:
            newTxtPath = os.path.join('/', str3.join(tempPathFirst), 'pretest', tempPathSecond)
        if not os.path.exists(newTxtPath):
            os.makedirs(newTxtPath)
        newTxtName = os.path.join(newTxtPath, tempPathLast)

        newfile = open(newTxtName, 'w')
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
        txt_paths.append(newTxtName)
        newfile.write(str2.join(templist))
        newfile.close()
        # print(data)
    return txt_paths

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

def preprocess_data(istrain):
    data_paths = []
    txt_paths = []
    if istrain:
        data_root = '/data/zcf/code/pytorch/d2l/dataset'
    else:
        data_root = '/data/zcf/code/pytorch/d2l/test'
    train_classification = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13']
    for i in train_classification:
        data_paths.append(os.path.join(data_root, i))
    print(data_paths)

    for idx, data_path in enumerate(data_paths):
        txt_names = get_listdir(data_path)
        txt_names.sort()
        print(txt_names)
        txt_path = predataset(txt_names, istrain)
        txt_paths.append(txt_path)
    print(txt_paths)
    return txt_paths






if __name__ == '__main__':
    preprocess_data(istrain=False)



