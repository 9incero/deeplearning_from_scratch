from memory_profiler import profile
import numpy as np
from steps.step16 import Variable, square

"""
(py39) (base) ➜  deeplearning_from_scratch git:(yoonjae) ✗ python -m memory_profiler step17-1p.py        
Filename: step17-1p.py

Line #    Mem usage    Increment  Occurrences   Line Contents
=============================================================
     6     59.6 MiB     59.6 MiB           1   @profile
     7                                         def run():
     8    135.5 MiB      0.0 MiB        1001       for i in range(1000):
     9    135.5 MiB     18.9 MiB        1000           x = Variable(np.random.randn(10000))
    10    135.5 MiB     57.0 MiB        1000           y = square(square(square(x)))
"""


@profile
def run():
    for i in range(1000):
        x = Variable(np.random.randn(10000))
        y = square(square(square(x)))


if __name__ == "__main__":
    run()
