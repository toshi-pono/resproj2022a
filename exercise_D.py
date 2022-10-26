import math
from typing import Final

C: Final[float] = 0.3


def calc_lnka(fb: float) -> float:
    """
    Calculate ln(Ka) from PPB(fb)
    See Materials and methods in the paper https://doi.org/10.1186/s12859-018-2529-z

    Parameters
    ----------
    fb : float
        PPB(fb)
    """
    fb = fb * 0.99 + 0.005
    return C * math.log(fb / (1.0 - fb))


def toPPB(lnKa: float) -> float:
    return math.exp(lnKa / C) / (1 + math.exp(lnKa / C))
