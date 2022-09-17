nums = list(map(int, input().split()))
a, b, m, x = nums[0], nums[1], nums[2], nums[3]

xlist = [x]
while 1:
    x = (a * x + b) % m
    if not xlist.__contains__(x):
        xlist.append(x)
    else:
        break
print(len(xlist))