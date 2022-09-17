import math
nk = input().split()
n = int(nk[0])
k = int(nk[1])

nums = list(map(int, input().split()))
nums.sort()

print('({},{})'.format(nums[math.ceil(k / n) % n - 1], nums[(k % n)-1]))
