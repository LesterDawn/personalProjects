def getString(data: str) -> str:
    # write code here
    list_data = sorted(list(data))
    set_data = list(set(list_data))
    s = ''
    for ele in set_data:
        s += ele
    return s


data = 'aaccbb'

print(getString(data))
