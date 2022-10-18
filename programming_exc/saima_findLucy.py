n = 7
nums = [1,2,3,4,5,6,6]
cnt = 0
if n == 1:
    print(1)
else:
    for i in range(n):
        flag = 1
        if i == n - 1:
            if nums[i] == nums[i-1]:
                flag = 0
        for j in range(i + 1, n):
            if nums[j] % nums[i] == 0:
                flag = 0
                break
        if flag == 1:
            cnt += 1
    print(cnt)
