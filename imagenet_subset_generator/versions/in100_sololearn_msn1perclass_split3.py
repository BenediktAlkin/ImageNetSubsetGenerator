from .in1k_msn1perclass_split3 import FILES as _FILES
from .in100_sololearn import CLASSES as _CLASSES

INFO = [
    "in1k_msn1perclass_split3 with only the classes of in100_sololearn"
]

FILES = [f for f in _FILES if f.split("_")[0] in _CLASSES]