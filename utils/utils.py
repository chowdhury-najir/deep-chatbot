def pad_or_trim_sequence(seq, length, token, keep_last=False):

    tmp = seq[:]

    if len(tmp) == length:
        return tmp

    if len(tmp) > length:
        if keep_last:
            return tmp[:length-1] + [tmp[length-1]]
        else:
            return tmp[:length]

    padding = []
    for _ in range(length-len(seq)):
        padding.append(token)

    tmp += padding

    return tmp
