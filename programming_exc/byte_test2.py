import sys

s = input().split()
a, b = int(s[0]), int(s[1])

bin_a, bin_b = bin(a).replace('0b', ''), bin(b).replace('0b', '')
n_a, n_b = len(list(bin_a)), len(list(bin_b))
res = ['0'] * (n_a + n_b)

for i in range(len(res)-1, -1, -1):
    if i % 2 == 0:
        if n_b > 0:
            res[i] = bin_b[n_b - 1]
            n_b -= 1
    else:
        if n_b > 0:
            res[i] = bin_a[n_a - 1]
            n_a -= 1

print(eval('0b' + ''.join(res)))