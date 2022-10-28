def LCS(s1: str, s2: str) -> str:  # 最长子序列
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


def LCS1(str1: str, str2: str) -> str:  # 最长子串
    # write code here
    n1, n2 = len(str1), len(str2)
    res = [[''] * (n2 + 1) for _ in range(n1 + 1)]
    str1, str2 = ' ' + str1, ' ' + str2
    for i in range(1, n1 + 1):
        for j in range(1, n2 + 1):
            if str2[j] == str1[i]:
                res[i][j] = res[i - 1][j - 1] + str2[j]
            else:
                res[i][j] = ''
    lcs = ''
    for ele in res:
        if len(max(ele, key=len)) > len(lcs):
            lcs = max(ele, key=len)
    return lcs


s1, s2 = "abcdefg", "ab1cdefgabc1defg"
print(LCS1(s1, s2))
