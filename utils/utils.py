import math
import tensorflow as tf


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


def safe_exp(value):
    """Exponentiation with catching of overflow error."""
    try:
        ans = math.exp(value)
    except OverflowError:
        ans = float("inf")
    return ans

def get_translation(nmt_outputs, sent_id, tgt_eos):
    """Given batch decoding outputs, select a sentence and turn to text."""
    if tgt_eos: tgt_eos = tgt_eos.encode("utf-8")
    # Select a sentence
    output = nmt_outputs[sent_id, :].tolist()

    # If there is an eos symbol in outputs, cut them at that point.
    if tgt_eos and tgt_eos in output:
        output = output[:output.index(tgt_eos)]

    print(output)
