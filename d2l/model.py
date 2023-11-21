import torch
from torchsummary import summary

class Bottlrneck(torch.nn.Module):
    def __init__(self,In_channel,Med_channel,Out_channel,downsample=False):
        super(Bottlrneck, self).__init__()
        self.stride = 1
        if downsample == True:
            self.stride = 2

        self.layer = torch.nn.Sequential(
            torch.nn.Conv1d(In_channel, Med_channel, 1, self.stride),
            torch.nn.BatchNorm1d(Med_channel),
            torch.nn.ReLU(),
            torch.nn.Conv1d(Med_channel, Med_channel, 3, padding=1),
            torch.nn.BatchNorm1d(Med_channel),
            torch.nn.ReLU(),
            torch.nn.Conv1d(Med_channel, Out_channel, 1),
            torch.nn.BatchNorm1d(Out_channel),
            torch.nn.ReLU(),
        )

        if In_channel != Out_channel:
            self.res_layer = torch.nn.Conv1d(In_channel, Out_channel,1,self.stride)
        else:
            self.res_layer = None

    def forward(self,x):
        if self.res_layer is not None:
            residual = self.res_layer(x)
        else:
            residual = x
        return self.layer(x)+residual


class ResNet(torch.nn.Module):
    def __init__(self, in_channels=1, num_classes=14):
        super(ResNet, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels,64,kernel_size=7,stride=2,padding=3),
            torch.nn.MaxPool1d(3,2,1),

            Bottlrneck(64,64,256,False),
            Bottlrneck(256,64,256,False),
            Bottlrneck(256,64,256,False),
            #
            Bottlrneck(256,128,512, True),
            Bottlrneck(512,128,512, False),
            Bottlrneck(512,128,512, False),
            Bottlrneck(512,128,512, False),
            #
            Bottlrneck(512,256,1024, True),
            Bottlrneck(1024,256,1024, False),
            Bottlrneck(1024,256,1024, False),
            Bottlrneck(1024,256,1024, False),
            Bottlrneck(1024,256,1024, False),
            Bottlrneck(1024,256,1024, False),
            #
            Bottlrneck(1024,512,2048, True),
            Bottlrneck(2048,512,2048, False),
            Bottlrneck(2048,512,2048, False),

            torch.nn.AdaptiveAvgPool1d(1)
        )
        self.classifer = torch.nn.Sequential(
            torch.nn.Linear(2048, num_classes)
        )

    def forward(self,x):
        x = self.features(x)
        x = x.view(-1,2048)
        x = self.classifer(x)
        return x

if __name__ == '__main__':
    x = torch.randn(size=(3,1,224))
    # x = torch.randn(size=(1,64,224))
    # model = Bottlrneck(64,64,256,True)
    model = ResNet(in_channels=1)

    output = model(x)
    print(f'输入尺寸为:{x.shape}')
    print(f'输出尺寸为:{output.shape}')
    # print(model)
    # summary(model,(1,224),device='cpu')

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# class ResBlock(nn.Module):
#     def __init__(self,in_features,out_features,stride=[1,1],padding=1):
#         super(ResBlock,self).__init__()
#         self.layer = nn.Sequential(     # 每个残差块中包含两次卷积
#             nn.Conv2d(in_features, out_features, kernel_size=3, stride=stride[0], padding=padding, bias=False),  # 第一次卷积
#             nn.BatchNorm2d(out_features),
#             nn.ReLU(inplace=True),  # 原地替换,节省内存开销
#             nn.Conv2d(out_features, out_features, kernel_size=3, stride=stride[1], padding=padding, bias=False),   # 第二次卷积
#             nn.BatchNorm2d(out_features)
#         )
#
#         # shortcut 部分
#         self.shortcut = nn.Sequential()
#         if stride[0] != 1 or in_features != out_features:   # 当输出维度和输入维度不一样时，不能直接相加，需要转换到一样的维度
#             self.shortcut = nn.Sequential(
#                 # 卷积核为1 进行升降维
#                 nn.Conv2d(in_features, out_features, kernel_size=1, stride=stride[0], bias=False),
#                 nn.BatchNorm2d(out_features)
#             )
#
#     def forward(self, x):
#         out = self.layer(x)
#         out += self.shortcut(x)    #Resnet核心，残差连接
#         out = F.relu(out)
#         return out
#
# class ResNet18(nn.Module):
#     def __init__(self,ResBlock,num_classes,in_futures=64):
#         super(ResNet18, self).__init__()
#         self.in_features = in_futures
#
#         # 开始进行卷积特征提取操作
#         self.conv1 = nn.Sequential(  # Resnet18的第一个卷积是独立的，后续卷积重复率较高
#             nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
#             nn.BatchNorm2d(64),
#             nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         )
#         self.conv2 = self._make_layer(ResBlock, 64, [[1, 1], [1, 1]])
#         self.conv3 = self._make_layer(ResBlock, 128, [[2, 1], [1, 1]])
#         self.conv4 = self._make_layer(ResBlock, 256, [[2, 1], [1, 1]])
#         self.conv5 = self._make_layer(ResBlock, 512, [[2, 1], [1, 1]])
#
#         #最后的pooling和Linear全连接层分类
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(512, num_classes)
#
#     def _make_layer(self, block, out_features, strides):  #这个函数主要是用来，重复同一个残差块
#         layers = []
#         for stride in strides:
#             layers.append(block(self.in_features, out_features, stride))
#             self.in_features = out_features
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         out = self.conv1(x)
#         out = self.conv2(out)
#         out = self.conv3(out)
#         out = self.conv4(out)
#         out = self.conv5(out)
#
#         out = self.avgpool(out)
#         out = out.reshape(x.shape[0], -1)
#         out = self.fc(out)
#         return out


## 打印查看网络
#res18 = ResNet18(ResBlock,10,64)
#print(res18)
