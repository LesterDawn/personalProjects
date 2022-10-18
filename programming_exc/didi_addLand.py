import numpy as np

m = int(input())
n = int(input())
add_cnt = int(input())

coordinates = np.zeros([m, n])
k = np.array([[1, -1]])


def corr2d(X, K):
    h, w = K.shape
    Y = np.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
    return Y


for i in range(add_cnt):
    island = 0
    s = input()
    s = list(map(int, s.split()))
    coordinates[s[0]][s[1]] = 1
    print(coordinates)
    print(corr2d(coordinates, k))
    print(corr2d(coordinates, k.transpose()))
