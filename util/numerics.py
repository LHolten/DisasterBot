

def clip(x: float, lower_cap: float = -1, higher_cap: float = 1):
    """Returns a clipped value in the range [lower_cap, higher_cap]"""
    if x < lower_cap:
        return lower_cap
    elif x > higher_cap:
        return higher_cap
    else:
        return x


def sign(x: float):
    """Retuns 1 if x > 0 else -1. > instead of >= so that sign(False) returns -1"""
    return 1 if x > 0 else -1
