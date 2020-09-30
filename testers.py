from all_lenses import *


class SimpleColumn(LensGroup):
    def __init__(self):
        super(SimpleColumn, self).__init__()
        self.elements = []

    def build(self):
        self.complete_arangement('test')


class LensTester(LensGroup):
    def __init__(self, lens):
        super(LensTester, self).__init__()
        self.elements = []
        self.lens = lens
        self.set_up()

    def set_up(self):
        self.append(Space(d=self.lens.R))
        self.append(self.lens)
        self.append(Space(d=self.lens.R))

    def build(self):
        self.complete_arangement('test')


class PositionTester(LensGroup):
    def __init__(self):
        super(PositionTester, self).__init__()
        self.elements = []
        self.R = 10
        self.position_results = []
        self.set_up()
        self.build()
        self.create_positions()
        self.test_positions()

    def set_up(self):
        self.append(Space(d=self.R))
        self.append(Deflector(d=5, I=1000, r=self.R))
        self.append(Space(d=self.R))
        self.append(Coil(d=5, I=1000, r=self.R))
        self.append(Space(d=self.R))
        self.append(Deflector(d=5, I=1000, r=self.R))
        self.append(Space(d=self.R))

    def build(self):
        self.complete_arangement('test')

    def create_positions(self):
        start = self.elements[0].beginning
        end = self.elements[-1].ending
        self.x_coordinates = np.linspace(start, end, 100)
        y_coordinates = np.linspace(-self.R, self.R, 100)
        z_coordinates = np.linspace(-self.R, self.R, 100)
        # X, Y, Z = np.mgrid(self.x_coordinates, y_coordinates, z_coordinates)

    def test_positions(self):
        lens_label = ["space", "deflector", "space", "coil", "space", "deflector", "space"]
        for i in range(len(self.x_coordinates)):
            for j in range(len(self.elements)):
                if self.elements[j].is_beam_here(self.x_coordinates[i]):
                    self.position_results.append(lens_label[j])

        # assert len(lens_label) == len(self.x_coordinates)


