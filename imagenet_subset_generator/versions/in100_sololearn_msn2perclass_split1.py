from .in1k_msn2perclass_split1 import FILES as _FILES
from .in100_sololearn import CLASSES as _CLASSES

FILES = [f for f in _FILES if f.split("_")[0] in _CLASSES]