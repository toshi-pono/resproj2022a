from math import log


from typing import Final
import math

# previous study [36]
C: Final[float] = 0.3


def calc_lnka(fb: float) -> float:
    fb = fb * 0.99 + 0.005
    return C * math.log(fb / (1.0 - fb))
