from memory_profiler import profile
import numpy as np
from steps.step17 import Variable, square

"""
(py39) (base) ➜  deeplearning_from_scratch git:(yoonjae) ✗ python -m memory_profiler step17-2p.py
Filename: step17-2p.py

Line #    Mem usage    Increment  Occurrences   Line Contents
=============================================================
     6     60.3 MiB     60.3 MiB           1   @profile
     7                                         def run():
     8     64.6 MiB      0.0 MiB        1001       for i in range(1000):
     9     64.6 MiB      0.7 MiB        1000           x = Variable(np.random.randn(10000))
    10     64.6 MiB      3.6 MiB        1000           y = square(square(square(x)))
"""


@profile
def run():
    for i in range(1000):
        x = Variable(np.random.randn(10000))
        y = square(square(square(x)))


if __name__ == "__main__":
    run()
