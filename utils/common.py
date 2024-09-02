
from inspect import isfunction

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def exists(val):
    return val is not None