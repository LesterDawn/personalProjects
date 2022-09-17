


userNum = int(input())
likeVal = list(map(int, input().split()))
quiryNum = int(input())
for i in range(quiryNum):
    quiry = list(map(int, input().split()))
    x, y, val = quiry[0], quiry[1], quiry[2]
    cnt = 0
    quiryRange = sorted(likeVal[x - 1: y])
    l, r, m = 0, len(quiryRange) - 1, 0
    while l != r:
        m = int((l + r) / 2)
        if quiryRange[m] > val:
            r = m - 1
        elif quiryRange[l] < val:
            l = m + 1
        else:
            break
    tmp = m
    if quiryRange[tmp] == val:
        while quiryRange[tmp] == val and tmp >= 0:
            tmp -= 1
            cnt += 1
        tmp = m
        while quiryRange[tmp] == val and tmp < len(quiryRange):
            tmp += 1
            cnt += 1
        cnt -= 1
    print(cnt)
