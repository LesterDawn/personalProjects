T = int(input())
if T == 0:
    print(0)
else:
    for i in range(T):
        data = list(input())
        if len(data) == 0:
            print(0)
        else:
            shiftCnt = 0
            for j in range(len(data)):
                if j == 0 and data[j].isupper():
                    shiftCnt += 1
                if not data[j-1].isupper() and data[j].isupper():
                    shiftCnt += 1
            print(shiftCnt+len(data))