from queue import Queue
n = int(input())
p1, p2, p3, p4, p5 = Queue(), Queue(), Queue(), Queue(), Queue()

for i in range(n):
    str_ = list(input().split())
    event = str_[0]
    if event == 'a':
        num, p = int(str_[1]), int(str_[2])
        if p == 1:
            p1.put(num)
        elif p ==2:
            p2.put(num)
        elif p == 3:
            p3.put(num)
        elif p == 4:
            p4.put(num)
        else:
            p5.put(num)
    else:
        if p1:
            print(p1.get())
            continue
        if p2:
            print(p2.get())
            continue
        if p3:
            print(p3.get())
            continue
        if p4:
            print(p4.get())
            continue
        if p2:
            print(p5.get())
            continue