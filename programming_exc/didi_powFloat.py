while True:
    try:
        s = input()
        R, n = s.split()[0], int(s.split()[1])
        digit_num = 0
        new_R = int(R.split('.')[0])
        if float(R) % 1.0 > 0:
            digit_num = len(R.split('.')[1]) * n
            new_R = int(R.split('.')[0] + R.split('.')[1])

        pow_R = str(pow(new_R, n))
        dot_left = pow_R[:len(pow_R) - digit_num]
        if float(R) % 1.0 > 0:
            dot_right = pow_R[len(pow_R) - digit_num:]
        else:
            dot_right = '0'
        if float(R) >= 1:
            pow_R = dot_left + '.' + dot_right
        else:
            for i in range(digit_num - len(dot_right)):
                dot_right = '0' + dot_right
            pow_R = '0.' + dot_right

        print(pow_R)
    except:
        break


