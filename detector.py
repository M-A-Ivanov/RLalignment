import numpy as np


class Detector(object):
    def __init__(self, simulation, position, n):
        self.beam = simulation.positions_all
        self.position = position
        self.n = n

    def detect(self):
        detection = []
        for n in range(0, np.shape(self.beam)[0]):
            x_beam = np.einsum('j, ij->i', self.n, np.array(self.beam[n][:][:]))
            x_detector = self.n.dot(self.position)
            arg = np.argmin(abs(x_beam - x_detector))
            detection.append(self.beam[n][arg][1:])  # we only care about y and z

        return detection
