import torch
import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = torch.Tensor([1.0])
w.requires_grad = True

def forward(x):
    return w * x

def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2

print("predict (before training)", 4, forward(4).item())

epoch_list = []
loss_list = []

for epoch in range(10):
    for x, y in zip(x_data, y_data):
        l = loss(x, y)
        l.backward()
        print('\tgrad:', x, y, w.grad.item(), w.grad.data, w.data)
        w.data = w.data - 0.01 * w.grad.data

        w.grad.data.zero_()
    print("progress:", epoch, l.item(), l.data)
    epoch_list.append(epoch)
    loss_list.append(l.item())
print('predict (after training)', 4, forward(4).item())

plt.plot(epoch_list, loss_list)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()