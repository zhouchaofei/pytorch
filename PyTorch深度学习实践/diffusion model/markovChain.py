import numpy as np

matrix = np.matrix([[0.9, 0.075, 0.025],
                    [0.15, 0.8, 0.05],
                    [0.25, 0.25, 0.5]], dtype=float)

vector1 = np.matrix([[0.3, 0.4, 0.3]], dtype=float)

for i in range(100):
    vector1 = vector1 * matrix
    print('Courrent round: {}'.format(i+1))
    print(vector1)