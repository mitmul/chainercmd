def write_left_to_right(left, right):
    if isinstance(left, dict):
        for key in left:
            print('key:', key, left[key])
            write_left_to_right(left[key], right[key])
    elif isinstance(left, list):
        for i in range(len(left)):
            if left[i] is not None:
                print('list:', left, right)
                write_left_to_right(left[i], right[i])
    else:
        print('else:', left, right)
