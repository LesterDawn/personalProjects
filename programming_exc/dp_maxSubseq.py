def max_product(nums) -> int:
    # write your code here
    n = len(nums)
    max_i = [nums[0]]
    min_i = [nums[0]]

    for i in range(1, n):
        if nums[i] > 0:
            max_i.append(max(max_i[i - 1] * nums[i], nums[i]))
            min_i.append(min(min_i[i - 1] * nums[i], nums[i]))
        else:
            max_i.append(max(min_i[i - 1] * nums[i], nums[i]))
            min_i.append(min(max_i[i - 1] * nums[i], nums[i]))
    return max(max_i)


nums = [-2, -3, 4, 2, 2, -1]

print(max_product(nums))
