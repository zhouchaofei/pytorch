import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

def forward(x):
    return x * w + b

def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)

w_list = []
b_list = []
mse_list = []

for w in np.arange(0.0, 4.1, 0.1):
    print('w=', w)
    for b in np.arange(-2.0, 2.1, 0.1):
        print('b=', b)
        l_sum = 0
        for x_val, y_val in zip(x_data, y_data):
            y_pred_val = forward(x_val)
            loss_val = loss(x_val, y_val)
            l_sum += loss_val
            print('\t', x_val, y_val, y_pred_val, loss_val)
        print('MSE=', l_sum/3)
        b_list.append(b)
        mse_list.append(l_sum/3)
    w_list.append(w)

b_r = np.array(b_list)
b_re = b_r.reshape(41, 41)
w, b = np.meshgrid(w_list, b_re[0])
m_r = np.array(mse_list)
m = m_r.reshape((41, 41))

# Plot the surface.
surf = ax.plot_surface(w, b, m, cmap=cm.coolwarm, rcount=100, ccount=100,
                       # rstride=1,  # rstride（row）指定行的跨度
                       # cstride=1,  # cstride(column)指定列的跨度
                       linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(0, 40)
ax.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
ax.zaxis.set_major_formatter('{x:.02f}')

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
