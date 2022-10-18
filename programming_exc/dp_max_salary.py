def max_salary(sal: list, start: list, end: list):
    n = len(sal)
    # sort by end date
    end, start, sal = (list(t) for t in zip(*sorted(zip(end, start, sal))))

    prev = [0 for i in range(n)]
    opt = [0 for i in range(n)]

    for i in range(n):  # calculate prev ()
        for j in range(i, -1, -1):  # from latest date to earliest date
            if end[j] <= start[i]:
                prev[i] = j + 1
                break

    opt[0] = sal[0]
    for i in range(1, n):
        do_i = sal[i] + opt[prev[i] - 1]
        not_do_i = opt[i - 1]
        opt[i] = max(do_i, not_do_i)

    return max(opt)


sal = [1, 5, 8, 4, 6, 3, 4, 2]
start = [3, 1, 0, 4, 3, 5, 8, 6]
end = [5, 4, 6, 7, 8, 9, 11, 10]
print(max_salary(sal, start, end))
