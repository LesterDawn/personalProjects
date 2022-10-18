import sys


def coin_change(arr, M):
    n = len(arr)
    f = [0 for i in range(0, M + 1)]
    f.append(0)

    for i in range(1, M + 1):
        f[i] = sys.maxsize
        for j in range(0, n):
            if i >= arr[j] and f[i - arr[j]] != sys.maxsize:
                f[i] = min(f[i - arr[j]] + 1, f[i])

    if f[M] == sys.maxsize:
        f[M] = -1
    return f[M]


arr = [2, 5, 7]
print(coin_change(arr, 27))
