import numpy as np


def and_gate(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    result = np.sum(x * w) + b
    return h(result)


def h(result):
    if result < 0:
        return 0
    else:
        return 1
