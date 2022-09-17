nums = []
while 1:
    try:
        s = input()
        nums.append(s)
    except:
        break

for i in range(len(nums)):
    negSymble = False
    if nums[i].__contains__('-'):
        nums[i] = nums[i].replace('-', '')
        negSymble = True
    for k in range(len(nums[i])):
        if len(nums[i]) == 1:
            break
        if nums[i][0] != '0' or nums[i][1] == '.':
            break
        else:
            nums[i] = nums[i][1:]
    dotLeft = nums[i]
    dotRight = '.00'
    if nums[i].__contains__('.'):
        dotLeft = nums[i].split('.')[0]
        dotRight = nums[i].split('.')[1]
        if len(dotRight) > 2:
            dotRight = dotRight[0:2]
        elif len(dotRight) < 2:
            dotRight += '0'
        dotLeft = list(dotLeft)
        for j in range(len(dotLeft) - 3, 0, -3):
            dotLeft[j] = ',' + dotLeft[j]
        dotLeft = ''.join(dotLeft)
        dotRight = '.' + dotRight
    else:
        dotLeft = list(nums[i])
        for j in range(len(dotLeft) - 3, 0, -3):
            dotLeft[j] = ',' + dotLeft[j]
        dotLeft = ''.join(dotLeft)
    if negSymble:
        if dotLeft + dotRight == '0.00':
            nums[i] = '$' + dotLeft + dotRight
        else:
            nums[i] = '(' + '$' + dotLeft + dotRight + ')'
    else:
        nums[i] = '$' + dotLeft + dotRight

for num in nums:
    print(num)
