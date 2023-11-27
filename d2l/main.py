import numpy as np
import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import argparse
# from model import ResNet18,ResBlock
from model import ResNet
import os
from utils import preprocess_data

parser = argparse.ArgumentParser(description='PyTorch Training for Mnist multi-classifier')

''' Train setting '''
parser.add_argument('--epochs', default=25, type=int) # 训练的一共次数
parser.add_argument('-b', '--batch-size', default=1000, type=int) # Batchsize 显卡越好可以越大
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float) # 学习率
parser.add_argument('-ch', '--in_channels', default=1, type=int)  # 输入维度
parser.add_argument('-n', '--num_classes', default=14, type=int)  # 最终分类数
parser.add_argument('--train', default=False, type=bool)  # 训练或者测试
parser.add_argument('--resume', default='None', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')


class DiabetsDataset(Dataset):
    def __init__(self, filepath):
        xy = np.loadtxt(filepath, delimiter=' ', dtype=np.float32)
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:, :-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])

    def __getitem__(self, item):
        return self.x_data[item], self.y_data[item]

    def __len__(self):
        return self.len


def main(args):
    ''' Dataset'''
    # txt_paths = preprocess_data(args.train)
    txt_paths = [['/data/zcf/code/pytorch/d2l/predataset/14/107291.txt', '/data/zcf/code/pytorch/d2l/predataset/14/108376.txt', '/data/zcf/code/pytorch/d2l/predataset/14/109008.txt'],
                ['/data/zcf/code/pytorch/d2l/predataset/14/124872.txt', '/data/zcf/code/pytorch/d2l/predataset/14/130618.txt', '/data/zcf/code/pytorch/d2l/predataset/14/131880.txt']]
    # txt_paths = [['/data/zcf/code/pytorch/d2l/pretest/0/432590.txt', '/data/zcf/code/pytorch/d2l/pretest/0/432928.txt', '/data/zcf/code/pytorch/d2l/pretest/0/439847.txt'],
    #              ['/data/zcf/code/pytorch/d2l/pretest/1/603287.txt', '/data/zcf/code/pytorch/d2l/pretest/1/607171.txt', '/data/zcf/code/pytorch/d2l/pretest/1/607180.txt']]
    for id in range(len(txt_paths)):
        for txt_path in txt_paths[id]:
            print(txt_path)
            dataset = DiabetsDataset(txt_path)
            if(dataset.len < args.batch_size):
                train_loader = DataLoader(dataset=dataset,
                                          batch_size=dataset.len,
                                          shuffle=True,
                                          num_workers=4)
            else:
                train_loader = DataLoader(dataset=dataset,
                                          batch_size=args.batch_size,
                                          shuffle=True,
                                          num_workers=4)

            test_dataset = DiabetsDataset(txt_path)
            if(test_dataset.len < args.batch_size):
                test_loader = DataLoader(dataset=test_dataset,
                                          batch_size=test_dataset.len,
                                          shuffle=False)
            else:
                test_loader = DataLoader(dataset=test_dataset,
                                          batch_size=args.batch_size,
                                          shuffle=False)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            in_channels = args.in_channels
            num_classes = args.num_classes
            model = ResNet(in_channels, num_classes)
            model.to(device)

            criterion = torch.nn.CrossEntropyLoss()  # 交叉熵损失函数
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)  # Adam优化器收敛更快

            if args.train:  # 若训练
                # if id != 0:
                #     args.resume = "./checkpoint/checkpoint.pth"
                #     checkpoint = torch.load(args.resume)
                #     model_dict = model.state_dict()
                #     for k, v in checkpoint['state_dict'].items():
                #         if k in model_dict and v.shape == model_dict[k].shape:
                #             model_dict[k] = v
                #         else:
                #             print('\tMismatched layers: {}'.format(k))
                #     model.load_state_dict(model_dict)
                args.resume = "./checkpoint/checkpoint.pth"
                checkpoint = torch.load(args.resume)
                model_dict = model.state_dict()
                for k, v in checkpoint['state_dict'].items():
                    if k in model_dict and v.shape == model_dict[k].shape:
                        model_dict[k] = v
                    else:
                        print('\tMismatched layers: {}'.format(k))
                model.load_state_dict(model_dict)
                for epoch in range(args.epochs):
                    training_loss = 0.0
                    best_tloss = 100.0
                    for idx, data in enumerate(train_loader, 0):
                        # 获得一个批次的数据和标签
                        images, labels = data
                        if labels.shape[0] == 1:
                            print("维度数据为1")
                            continue
                        images = torch.unsqueeze(images, dim=1)
                        labels = labels.view(-1)
                        labels = labels.long()

                        images = images.to(device)
                        labels = labels.to(device)
                        optimizer.zero_grad()

                        # 获得模型预测结果（Batch,14）
                        outputs = model(images)

                        # 交叉熵代价函数outputs(64,10),target（64）
                        loss = criterion(outputs, labels)

                        loss.backward()  # 回传损失并清空的书写顺序
                        optimizer.step()

                        training_loss += loss.item()

                        if idx % 2 == 1:  # 每2轮打印一次
                            print('Epoch: %d, Iter: %5d, loss: %.3f' % (epoch + 1, idx + 1, training_loss / 2))

                            if best_tloss > training_loss:
                                best_tloss = training_loss

                                checkpoint = {
                                    'epoch': epoch + 1,
                                    'iter': idx,
                                    'state_dict': model.state_dict(),
                                    'best_loss': training_loss
                                }
                                filename = 'checkpoint.pth'
                                res_path = os.path.join("./checkpoint/", filename)
                                print('Save best checkpoint to {}'.format(res_path))
                                torch.save(checkpoint, res_path)

                            training_loss = 0.0

            if not args.train:  # 若测试
                print('test')
                args.resume = "./checkpoint/checkpoint.pth"
                checkpoint = torch.load(args.resume)
                model_dict = model.state_dict()
                for k, v in checkpoint['state_dict'].items():
                    if k in model_dict and v.shape == model_dict[k].shape:
                        model_dict[k] = v
                    else:
                        print('\tMismatched layers: {}'.format(k))
                model.load_state_dict(model_dict)

                correct = 0
                total = 0
                with torch.no_grad():
                    for data in test_loader:
                        images, labels = data
                        if labels.shape[0] == 1:
                            print("维度数据为1")
                            continue
                        images = torch.unsqueeze(images, dim=1)
                        labels = labels.view(-1)
                        labels = labels.long()

                        images = images.to(device)
                        labels = labels.to(device)
                        outputs = model(images)

                        _, predicted = torch.max(outputs.data, dim=1)  # 输出最大概率的下标
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()  # 张量之间的比较运算
                print('total_nums: ', total)
                print('correct_nums: ', correct)
                print('accuracy on test set: %d %% ' % (100 * correct / total))


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)