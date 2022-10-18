import numpy as np


def can_jump_dp(A: list):
    n = len(A)
    f = np.zeros(n)
    f[0] = 1

    for j in range(1, n):
        f[j] = 0
        # last jump is from i to j
        for i in range(0, n):
            if f[i] == 1 and i + A[i] >= j:
                f[j] = 1
                break
    return f[n - 1] == 1


def can_jump_greedy(A: list):
    n = len(A)
    i = 0
    while i <= n - 1:
        if i > 0:
            i = A[i]


A = [2, 3, 1, 1, 4]
print(can_jump_dp(A))


