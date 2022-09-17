from math import sqrt

while True:
    str = input()
    if not str:
        break
    n, m = int(str.split()[0]), int(str.split()[1])
    sum = 0
    for i in range(m):
        sum += n
        n = sqrt(n)
    print('{:.2f}'.format(sum))
