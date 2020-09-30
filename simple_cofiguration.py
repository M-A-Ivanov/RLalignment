import all_lenses as lens
from microscope import Microscope
from testers import SimpleColumn
from simulator import Simulate
from visualizer import Visualizer
import numpy as np

import matplotlib.pyplot as plt

from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import pyqtgraph.exporters
import sys
import the_watchdog as watch
import time
from skimage import io
import cv2


class Environment:

    def __init__(self, I_coil, I_deflector_y, I_deflector_z):
        self.I_deflector_y = I_deflector_y
        self.I_deflector_z = I_deflector_z
        self.I_coil = I_coil
        self.n_actions = 6

        self.I_deflector_y_init = I_deflector_y
        self.I_deflector_z_init = I_deflector_z
        self.I_coil_init = I_coil

        self.Test_microscope = None
        self.simulation = None

        self.scatter = None
        self.label = None
        self.indicators = ['-', '-', '-']
        self.qt_application = None
        self.file_number = 0

        self.create_system()
        self.state = self.get_state()
        self.initial_spot_spread = self.output_tf()
        self.goal = 0.5 * self.initial_spot_spread
        self.counter = 0

        ''' Specifically for the output required by the new RL algorithm. Need to modify how this works to apply to
        the actual microscope file input.'''
        self.create_graph(print_info=False)
        self.path = "C:\\Users\\User\\OneDrive - Cardiff University\\Data\\simulations\\RL\\"

        self.watchdog = None
        self.latest_image = None
        self.image_created_flag = False

    def get_state(self):
        return np.array([self.output_tf(), self.I_deflector_y, self.I_deflector_z, self.I_coil])

    def random_start(self):
        self.I_deflector_y = np.random.normal(loc=self.I_deflector_y_init, scale=abs(0.1 * self.I_deflector_y_init))
        self.I_deflector_z = np.random.normal(loc=self.I_deflector_z_init, scale=abs(0.1 * self.I_deflector_z_init))
        self.I_coil = np.random.normal(loc=self.I_coil_init, scale=abs(0.1 * self.I_coil_init))

    def create_system(self):
        """Current working procedure is:
                - Define the larger objects (the imaging columns and the microscope,
                    contained in microscope.py).
                - Make their separate components (contained in all_lenses.py).
                - Attach the separate components to the illumination of imaging columns, using 'column.append(lens)', which is
                    a dedicated function in the "LensGroup" object in all_lenses.py. Appending should be done in the correct order.
                - After the columns are complete, append them to the microscope similarly to previous step.
                - Use the function 'Microscope.build_microscope()' to finalize the microscope configuration.
                - Initiate the simulation by defining the simulator with the microscope as an argument - Simulator(Microscope).
                - Create the simulation beam and run the simulation, using 'Simulation.simulate_all()'
                    Then use the visualize function to draw the microscope and, if desired, draw the magnetic fields.
            """
        self.Test_microscope = Microscope()

        Column = SimpleColumn()

        Column.append(lens.Space(d=30.))
        Column.append(lens.Deflector(d=5., I_y=self.I_deflector_y, I_z=self.I_deflector_z, r=100.))
        Column.append(lens.Space(d=10))
        Column.append(lens.Coil(d=1., I=self.I_coil, r=100.))
        Column.append(lens.Space(d=30., detector=True))

        self.Test_microscope.append(Column)
        self.Test_microscope.build_microscope()

        self.simulation = Simulate(self.Test_microscope)
        self.simulation.create_beam(12, 5)
        self.simulation.simulate_all()
        self.Test_microscope.read_detector(self.simulation)

    def reset(self):
        self.random_start()
        self.create_system()
        self.state = self.get_state()
        self.goal = 0.5 * self.output_tf()

        return self.get_state()

    def output_tf(self):
        return np.std(self.Test_microscope.detector_readings)

    def setFluffy(self, path=None):
        if path is not None:
            self.path = path
        self.watchdog = watch.Fluffy(self.ImageFound, self.path)
        self.watchdog.SniffAround()

    def watchForImage(self):
        self.image_created_flag = False

    @staticmethod
    def getImage(path):
        image = io.imread(path, as_gray=True)
        return image

    def ImageFound(self, image_path):
        self.latest_image = image_path
        self.image_created_flag = True

    def step(self, action):
        """action is between 0, 5:
                {0,1}: reduce, increase first lens
                {2,3}: reduce, increase deflector y
                {4,5}: reduce, increase deflector z.
                -if its stupid and it works, it isn't stupid.
            For now, lets say the adjustments are fixed at +-2 in case of coil and +-1 in case of deflectors.
        """
        self.counter += 1

        done = bool(self.counter >= 50)
        # should think of a more elegant (and scalable) way to do actions...
        self.indicators = ['', '', '']
        previous_state = self.get_state()[0]
        defl_adjustment = 10
        coil_adjustment = 10
        ray_spread = self.state[0]
        if action == 0:
            self.I_coil -= coil_adjustment
            self.indicators[0] = '  \u2193'  # down arrow (unicode)
        if action == 1:
            self.I_coil += coil_adjustment
            self.indicators[0] = '  \u2191'  # up arrow
        if action == 2:
            self.I_deflector_y -= defl_adjustment
            self.indicators[1] = '  \u2193'
        if action == 3:
            self.I_deflector_y += defl_adjustment
            self.indicators[1] = '  \u2191'
        if action == 4:
            self.I_deflector_z -= defl_adjustment
            self.indicators[2] = '  \u2193'
        if action == 5:
            self.I_deflector_z += defl_adjustment
            self.indicators[2] = '  \u2191'

        self.create_system()
        done = bool(self.output_tf() <= self.goal) | done
        if done:
            self.counter = 0
            self.reset()

        if self.output_tf() <= self.goal:
            reward = 1
        else:
            reward = (previous_state - self.output_tf() / (self.initial_spot_spread - self.goal))

        self.watchForImage()
        '''Don't need these two with the microscope image outputs'''
        self.update_graph()
        self.save_graph()
        # self.watchForImage()  # program waits for the recorded image
        while not self.image_created_flag:
            time.sleep(1)
        return self.getImage(self.latest_image), reward, done

    def create_graph(self, print_info=True):
        """ Created using examples from:
        http://www.pyqtgraph.org/downloads/0.10.0/pyqtgraph-0.10.0-deb/pyqtgraph-0.10.0/examples/ScatterPlot.py
        https://stackoverflow.com/questions/45046239/python-realtime-plot-using-pyqtgraph
        """
        self.qt_application = QtGui.QApplication([])

        self.window = pg.GraphicsWindow(title="Rays on the detector")  # creates a window
        self.p = self.window.addPlot(title="Rays on the detector")
        self.p.setXRange(-20, 20, padding=0)
        self.p.setYRange(-20, 20, padding=0)
        self.scatter = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None), brush=pg.mkBrush(255, 255, 255, 120))
        self.p.addItem(self.scatter)
        self.p.hideAxis('bottom')
        self.p.hideAxis('left')
        if print_info:
            self.label = pg.TextItem(anchor=(0.5, 0.5), border=0.5)
            self.label.setPos(15, 15)
            self.p.addItem(self.label)

        self.update_graph(print_info)

    def update_graph(self, info=False):
        data = np.vstack(self.Test_microscope.detector_readings)
        self.scatter.setData(data[0, :, 0], data[0, :, 1])  # set the curve with this data
        if info:
            self.label.setHtml(
                '<div style="text-align: center"><span style="color: #FFF;font-size:10pt;">CL_1 = %d  ' % self.I_coil
                + self.indicators[0] + '<br>Deflector_y = %d  ' % self.I_deflector_y + self.indicators[1]
                + '<br>Deflector_z = %d  ' % self.I_deflector_z + self.indicators[2]
                + '</span></div>')
        QtGui.QApplication.processEvents()

    def save_graph(self, width=None, height=None):
        if width is None:
            width = 1024
        if height is None:
            height = 1024
        exporter = pg.exporters.ImageExporter(self.p)
        exporter.params.param('width').setValue(width, blockSignal=exporter.widthChanged)
        exporter.params.param('height').setValue(height, blockSignal=exporter.heightChanged)
        #
        # exporter.parameters()['width'] = int(1024)
        # exporter.parameters()['height'] = int(1024)

        exporter.export(self.path + str(self.file_number) + '.png')
        self.file_number += 1

    def remove_graph(self):
        pg.QtGui.QApplication.exec_()

    def output_visual(self):
        data = np.vstack(self.Test_microscope.detector_readings)
        x = data[0, :, 0]
        y = data[0, :, 1]
        fig, axs = plt.subplots(1, 1)
        axs.scatter(x, y)
        axs.axis('equal')
        plt.show()

        visual = Visualizer(self.Test_microscope, self.simulation)
        visual.create(field=False, exceptions='coil')
