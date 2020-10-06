import watchdog
import numpy as np
import os


def rawToImage(path):
    offset = abs(2 ** 21 - os.path.getsize(path))
    with open(path, 'r') as f:
        f.seek(offset)
        return np.fromfile(f, dtype=np.uint16).reshape((1024, 1024))


def stepChanges(modules):
    """This is a lame workaround for how much each module should change each step.
    Its a dictionary where each key is a module name and the value si the (arbitrary) step change"""
    changes = dict()
    for module in modules.values():
        if module in ['ISTIGA', 'ISTIGB', 'ILUDX', 'ILUDY', 'CL3DX',
                      'CL3DY', 'CL2DX', 'CL2DY', 'DSTIGA', 'DSTIGB',
                      'FLDY', 'ILDX', 'ILDY', 'P1DX', 'P1DY', 'P2AX',
                      'P2AY']:
            changes[module] = 0.5

        elif module in ['OSTIGA', 'OSTIGB']:
            changes[module] = 4

        elif module in ['MOBJ']:
            changes[module] = 2

        elif module in ['CL1',
                      'CL2', 'CL3', 'TL', 'FL', 'IL', 'P1', 'P2']:
            changes[module] = 1

        elif module in ['FLDX', 'OBJDX','OBJDY']:
            changes[module] = 0.2

        elif module in ['SELO']:
            changes[module] = 0

        elif module in ['SELI']:
            changes[module] = 0.01

        elif module in ['STV']:
            changes[module] = 0.1

    return changes




