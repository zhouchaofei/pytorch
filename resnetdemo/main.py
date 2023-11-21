import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.optim as optim
import argparse
from model import ResNet18,ResBlock
import os

parser = argparse.ArgumentParser(description='PyTorch Training for Mnist multi-classifier')

''' Train setting '''
parser.add_argument('--epochs', default=5, type=int) # 训练的一共次数
parser.add_argument('-b', '--batch-size', default=32, type=int) # Batchsize 显卡越好可以越大
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float) # 学习率
parser.add_argument('-n', '--num_classes', default=14, type=int)  # 最终分类数
parser.add_argument('--train', default=True, type=bool)  # 训练或者测试
parser.add_argument('--resume', default='None', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')


def main(args):
    ''' Dataset'''
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]) # 归一化,均值和方差
    train_dataset = datasets.MNIST(root='./dataset/mnist/', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
    test_dataset = datasets.MNIST(root='./dataset/mnist/', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ResNet18(ResBlock,args.num_classes)
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()  # 交叉熵损失函数
    optimizer = optim.Adam(model.parameters(),lr = args.lr, weight_decay=1e-5) # Adam优化器收敛更快

    #args.train = False # 测试

    if args.train:  #若训练
        for epoch in range(args.epochs):
            training_loss = 0.0
            best_tloss = 100.0
            for idx, data in enumerate(train_loader, 0):
                # 获得一个批次的数据和标签
                inputs, target = data
                inputs = inputs.to(device)
                target = target.to(device)
                optimizer.zero_grad()

                # 获得模型预测结果（Batch,10）
                outputs = model(inputs)

                # 交叉熵代价函数outputs(64,10),target（64）
                loss = criterion(outputs, target)

                loss.backward()     # 回传损失并清空的书写顺序
                optimizer.step()

                training_loss += loss.item()

                if idx % 20 == 19: # 每20轮打印一次
                    print('Epoch: %d, Iter: %5d, loss: %.3f' % (epoch + 1, idx + 1, training_loss / 20))

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

    if not args.train:  #若测试

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
                outputs = model(images)
                _, predicted = torch.max(outputs.data, dim=1) # 输出最大概率的下标
                total += labels.size(0)
                correct += (predicted == labels).sum().item()  # 张量之间的比较运算
        print('total_nums: ' , total)
        print('correct_nums: ' , correct)
        print('accuracy on test set: %d %% ' % (100 * correct / total))

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)