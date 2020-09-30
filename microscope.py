from all_lenses import LensGroup
import numpy as np


class ColumnType:
    imaging = 1
    illumination = 2
    test = 0


class IlluminationColumn(LensGroup):
    def __init__(self):
        super(IlluminationColumn, self).__init__()
        self.elements = []

    def build(self):
        self.complete_arangement('illumination')


class ImagingColumn(LensGroup):
    def __init__(self):
        super(ImagingColumn, self).__init__()
        self.elements = []

    def build(self):
        self.complete_arangement('imaging')


class Microscope(LensGroup):
    def __init__(self):
        super(Microscope, self).__init__()
        self.elements = []
        self.detector_readings = []

    def build_microscope(self):
        for column in self.elements:
            column.build()

    def getB(self, pos):
        for column in self.elements:
            return column.getB(pos)

    def read_detector(self, simulation):
        for column in self.elements:
            column.read_detector(simulation)
            self.detector_readings.append(column.detector_readings)

