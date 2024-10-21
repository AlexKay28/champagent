import os
import math


def min_max_restriction(v, min_v, max_v):
    return max(min(v, max_v), min_v)


def dist(p1, p2):
    return math.sqrt(sum(abs(v1 - v2) ** 2 for v1, v2 in zip(p1, p2)))
