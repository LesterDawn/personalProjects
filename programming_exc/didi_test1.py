nk = list(map(int, input().split()))
n, k = nk[0], nk[1]
weights = sorted(list(map(int, input().split())))

cnt = 0
max_w, min_w = max(weights), min(weights)
s = max_w - min_w
for i, w in enumerate(weights):
    if w <= s:
        cnt += 1

print(cnt)
