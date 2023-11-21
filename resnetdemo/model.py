import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self,in_features,out_features,stride=[1,1],padding=1):
        super(ResBlock,self).__init__()
        self.layer = nn.Sequential(     # 每个残差块中包含两次卷积
            nn.Conv2d(in_features, out_features, kernel_size=3, stride=stride[0], padding=padding, bias=False),  # 第一次卷积
            nn.BatchNorm2d(out_features),
            nn.ReLU(inplace=True),  # 原地替换,节省内存开销
            nn.Conv2d(out_features, out_features, kernel_size=3, stride=stride[1], padding=padding, bias=False),   # 第二次卷积
            nn.BatchNorm2d(out_features)
        )

        # shortcut 部分
        self.shortcut = nn.Sequential()
        if stride[0] != 1 or in_features != out_features:   # 当输出维度和输入维度不一样时，不能直接相加，需要转换到一样的维度
            self.shortcut = nn.Sequential(
                # 卷积核为1 进行升降维
                nn.Conv2d(in_features, out_features, kernel_size=1, stride=stride[0], bias=False),
                nn.BatchNorm2d(out_features)
            )

    def forward(self, x):
        out = self.layer(x)
        out += self.shortcut(x)    #Resnet核心，残差连接
        out = F.relu(out)
        return out

class ResNet18(nn.Module):
    def __init__(self,ResBlock,num_classes,in_futures=64):
        super(ResNet18, self).__init__()
        self.in_features = in_futures

        # 开始进行卷积特征提取操作
        self.conv1 = nn.Sequential(  # Resnet18的第一个卷积是独立的，后续卷积重复率较高
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.conv2 = self._make_layer(ResBlock, 64, [[1, 1], [1, 1]])
        self.conv3 = self._make_layer(ResBlock, 128, [[2, 1], [1, 1]])
        self.conv4 = self._make_layer(ResBlock, 256, [[2, 1], [1, 1]])
        self.conv5 = self._make_layer(ResBlock, 512, [[2, 1], [1, 1]])

        #最后的pooling和Linear全连接层分类
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_features, strides):  #这个函数主要是用来，重复同一个残差块
        layers = []
        for stride in strides:
            layers.append(block(self.in_features, out_features, stride))
            self.in_features = out_features
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)

        out = self.avgpool(out)
        out = out.reshape(x.shape[0], -1)
        out = self.fc(out)
        return out


## 打印查看网络
#res18 = ResNet18(ResBlock,10,64)
#print(res18)
