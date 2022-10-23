def LCS(s1: str, s2: str) -> str:
    # write code here
    res = ''
    n1, n2 = len(s1), len(s2)
    record = [[''] * (n2 + 1) for i in range(n1 + 1)]
    s1 = ' ' + s1
    s2 = ' ' + s2
    for i in range(1, n1 + 1):
        for j in range(1, n2 + 1):
            if s1[i] == s2[j]:
                record[i][j] = record[i - 1][j - 1] + s1[i]
            else:
                record[i][j] = record[i][j - 1] if len(record[i][j - 1]) >= len(record[i - 1][j]) else record[i - 1][j]
    for l in record:
        if len(max(l, key=len)) >= len(res):
            res = max(l, key=len)
    return res if res else '-1'

s1 = "1a1a31"
s2 = "1a231"
print(LCS(s1, s2))
