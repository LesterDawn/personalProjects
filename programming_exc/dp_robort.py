import numpy as np


def route_num(m: int, n: int):
    arr2D = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            if i == 0 or j == 0:
                arr2D[i][j] = 1
            else:
                arr2D[i][j] = arr2D[i-1][j] + arr2D[i][j-1]

    return arr2D[m-1][n-1]


print(int(route_num(1, 3)))

