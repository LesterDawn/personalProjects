starPos = []
n = int(input())

for i in range(n):
    s = input()
    starPos.append(list(map(int, s.strip().split())))

if n < 5:
    print(0)
else:
    xPos, yPos = [], []
    for star in starPos:
        xPos.append(star[0])
        yPos.append(star[1])
    minX, maxX = sorted(xPos)[0], sorted(xPos)[n - 1]
    minY, maxY = sorted(yPos)[0], sorted(yPos)[n - 1]

    luckyCnt = 0
    for i, x in enumerate(xPos):
        upCnt, downCnt = 0, 0
        if minX < x < maxX and maxY > yPos[i] > minY:
            for j, y in enumerate(yPos):
                if y > yPos[i] and xPos[j] == x:
                    upCnt += 1
                if y < yPos[i] and xPos[j] == x:
                    downCnt += 1
            if upCnt >= 1 and downCnt >= 1:
                luckyCnt += 1
    print(luckyCnt)
