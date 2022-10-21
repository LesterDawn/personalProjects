def LCS(s1: str, s2: str) -> str:
    # write code here
    res = ''
    start, curr = 0, 0
    n1, n2 = len(s1), len(s2)
    s1, s2 = s1 if n1 >= n2 else s2, s2 if n1 >= n2 else s1
    n1, n2 = len(s1), len(s2)
    while True:
        if curr == n1:
            break

        start = min(start, n2 - 1)
        end = min(n1, curr + 1)
        s2_curr = min(curr, n2 - 1)
        if s1[curr] in s2[start:end]:
            res += s1[curr]
            start = curr
        elif s2[s2_curr] in s1[start:curr + 1]:
            res += s2[s2_curr]
            start = curr
        if s2[s2_curr] == s1[curr]:
            start = curr + 1
        curr += 1

    return res if len(res) > 0 else -1

s1 = "1a1a31"
s2 = "1a231"
print(LCS(s1, s2))
