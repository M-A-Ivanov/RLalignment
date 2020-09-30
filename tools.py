import watchdog
import numpy as np
import os


def rawToImage(path):
    offset = abs(2 ** 21 - os.path.getsize(path))
    with open(path, 'r') as f:
        f.seek(offset)
        return np.fromfile(f, dtype=np.uint16).reshape((1024, 1024))


