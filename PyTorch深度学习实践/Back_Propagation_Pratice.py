import torch
import matplotlib.pyplot as plt
import core.urils as utils

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w1 = torch.Tensor([0.5])
w2 = torch.Tensor([1.0])
b = torch.Tensor([1.0])
w1.requires_grad = True
w2.requires_grad = True
b.requires_grad = True

def forward(x):
    return x ** 2 * w1 + x * w2 + b

def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2

print("predict (before training)", 4, forward(4).item())

epoch_list = []
loss_list = []

for epoch in range(50):
    for x, y in zip(x_data, y_data):
        l = loss(x, y)
        l.backward()
        print('\tgrad:', x, y, w1.grad.item(), w2.grad.item(), b.grad.item(),
              w1.data, w2.data, b.data)
        w1.data = w1.data - 0.02 * w1.grad.data
        w2.data = w2.data - 0.01 * w2.grad.data
        b.data = b.data - 0.01 * b.grad.data

        w1.grad.data.zero_()
        w2.grad.data.zero_()
        b.grad.data.zero_()
    print("progress:", epoch, l.item(), l.data)
    epoch_list.append(epoch)
    loss_list.append(l.item())
print('predict (after training)', 4, forward(4).item())

plt.plot(epoch_list, loss_list)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()