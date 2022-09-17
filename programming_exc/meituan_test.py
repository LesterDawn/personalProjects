s1 = list(map(int, input().split()))
nums_x = list(map(int, input().split()))
nums_y = list(map(int, input().split()))
nums_x.reverse()
nums_y.reverse()
tmp = nums_x.copy()
tmp.extend(nums_y)
nums_all = sorted(tmp)

n, x, y = s1[0], s1[1], s1[2]
opt_cnt = 0

while len(nums_all) > 0:
    if len(nums_x) > 0 and nums_x[0] == min(nums_all):
        nums_x.pop(0)
        nums_all.pop(0)
        opt_cnt += 1
    elif len(nums_y) > 0 and nums_y[0] == min(nums_all):
        nums_y.pop(0)
        nums_all.pop(0)
        opt_cnt += 1
    else:
        if nums_x.__contains__(min(nums_all)):
            nums_y.insert(0, nums_x.pop(0))
            opt_cnt += 1
        elif nums_y.__contains__(min(nums_all)):
            nums_x.insert(0, nums_y.pop(0))
            opt_cnt += 1
print(opt_cnt)

