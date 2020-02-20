from typing import Callable


def fix_generator(func: Callable):
    def result(self):
        gen = func(self)
        next(gen)
        return gen

    return result
