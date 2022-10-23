"""
1. 多列表同时排序
"""
# end, start, sal = (list(t) for t in zip(*sorted(zip(end, start, sal))))
"""
2. 创建二维列表
"""
# record = [[''] * n2 for i in range(n1)]
s = 'A@6,B@7,C@8,D@9,E@10'

a = list(s.split(','))
name = []
age = []
for val in a:
    name.append(val.split('@')[0])
    age.append(int(val.split('@')[1]))
age, name = (list(t) for t in zip(*sorted(zip(age, name))))
for i, val in enumerate(a):
    print('第' + str(i+1) + '个: [姓名: ' + name[i] + ', 年龄: ' + str(age[i]) + '岁]')


