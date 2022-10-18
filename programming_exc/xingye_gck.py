x, y = 5, 654320

prod = x * y
a = x
cnt = 0
while a <= y:
    b = prod / a
    if b % x == 0:
        cnt += 1
    a += x
print(cnt)