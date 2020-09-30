import numpy as np
import utils
from scipy.special import ellipk, ellipe
from mayavi import mlab
import magnetic_fields
from detector import Detector
"""
How thing work here:
There are two generic objects that are inherited: Lens and LensGroup.
Lens is inherited by: Coil, Deflector, Space, Splitter, so far..
LensGroup is inherited by: Microscope, Illumination Column, Imaging Column, so far..
Lens is a skeleton for all lenses, itdefines the properties and the functionalities each lens will have.
LensGroup brings together all the lenses that interact with each other, lets them communicate.
Example: "Space" needs to know what it is surrounded by in order to produce the edge fields from lenses.
"""
# %% GENERIC LENS
class Lens(object):
    def __init__(self,
                 width: float = 1.,
                 radius=None,
                 radiusvar=None,  # at some point give a beginning and end radius
                 current=None,
                 coil_turns=None,
                 lens_type='',
                 label=''
                 ):
        # Each lens knows its position
        self.width = width
        self.current = current
        self.name = label
        self.lens_type = lens_type
        if radius == None:
            self.R = 10
        else:
            self.R = radius

        if coil_turns == None:
            self.coil_turns = 1e3
        else:
            self.coil_turns = coil_turns

    def is_beam_here(self, pos):
        '''Finds out whether the beam passes through the lens.
        Arguments:
             position: float; the position of the beam in the x-axis (optical axis)
        Returns: boolean, True if beam is inside the given lens.
        '''

        return self.n.dot(self.beginning) <= np.dot(self.n, pos) <= self.n.dot(self.ending)

    def GetParams(self, start, n):
        self.n, self.m, self.l = utils.base_vectors(n)
        self.beginning = start
        self.ending = self.beginning + self.width * self.n

    def GetExtraParams(self, I_start, I_end, n_start, n_end, r_start,
                       r_end, lens_type_begin, lens_type_end):
        return None

    def getB(self, x):
        return np.c_[0, 0, 0]

    def display(self):
        pass

    def draw_field(self, field):
        pass

    def is_field_here(self, field):
        pass

    def read_detector(self):
        pass

# %% GROUPING LENSES

class LensGroup(Lens):
    def __init__(self):
        super(LensGroup, self).__init__()
        self.elements = []
        self.detector_readings = []

    def append(self, lens):
        self.elements.append(lens)

    def AddInitialParams(self, n):
        self.elements[0].GetParams(np.r_[0, 0, 0], n)  # starts from x=0
        for i in range(0, len(self.elements[:-1])):
            self.elements[i + 1].GetParams(self.elements[i].ending, n)

    def AddExtraParams(self):
        for i in range(0, len(self.elements) - 1):
            if self.elements[i].lens_type == 'space':
                self.elements[i].GetExtraParams(self.elements[i - 1].current,
                                                self.elements[i + 1].current,
                                                self.elements[i - 1].coil_turns,
                                                self.elements[i + 1].coil_turns,
                                                self.elements[i - 1].R,
                                                self.elements[i + 1].R,
                                                self.elements[i - 1].lens_type,
                                                self.elements[i + 1].lens_type)
        if self.elements[0].lens_type == 'space':
            self.elements[0].GetExtraParams(0,
                                            self.elements[1].current,
                                            0,
                                            self.elements[1].coil_turns,
                                            self.elements[1].R,
                                            self.elements[1].R,
                                            0,
                                            self.elements[1].lens_type)

        if self.elements[-1].lens_type == 'space':
            self.elements[-1].GetExtraParams(self.elements[-2].current,
                                             0,
                                             self.elements[-2].coil_turns,
                                             self.elements[-2].R,
                                             self.elements[-2].R,
                                             0,
                                             self.elements[-2].lens_type,
                                             0)

    def complete_arangement(self, column):
        if column == 'illumination':
            self.n = np.r_[0.86602541, 0.5, 0.]
        if column == 'imaging':
            self.n = np.r_[0.86602541, -0.5, 0.]
        if column == 'test':
            self.n = np.r_[1, 0, 0]
        self.AddInitialParams(self.n)
        self.AddExtraParams()

    def getB(self, pos):
        for element in self.elements:
            if element.is_beam_here(pos):
                return element.getB(pos)
        return np.c_[0, 0, 0]

    def display(self):
        for element in self.elements:
            element.display()

    def read_detector(self, simulation):
        for element in self.elements:
            if element.lens_type == 'space':
                if element.detector_position is not None:
                    self.detector_readings.append(element.read_detector(simulation))


# %% DIFFERENT LENS TYPES
class Coil(Lens):
    def __init__(self, d, I, r=None, name=None, n_turns=None):
        super(Coil, self).__init__(width=d,
                                   current=I,
                                   coil_turns=n_turns,
                                   radius=r,
                                   lens_type='coil',
                                   label=name)


    def getB(self, x):
        trans = np.vstack((self.n, self.l, self.m))
        B = np.dot(self.n * np.ones((len(x[:]), 3)), trans)
        return -0*self.coil_turns * 2e-7 * (self.current * B.T).T

    def display(self):
        theta = np.linspace(0, 2 * np.pi, 30)
        theta = theta[..., np.newaxis]
        coil = np.atleast_1d(self.R) * (np.sin(theta) * self.l + np.cos(theta) * self.m)

        coil += self.beginning
        coil_x = coil[:, 0]
        coil_y = coil[:, 1]
        coil_z = coil[:, 2]
        mlab.plot3d(coil_x, coil_y, coil_z,
                    tube_radius=0.05,
                    #                name='Coil %i' % int(lens_num/2),
                    color=(0, 0, 0))

        coil += self.ending - self.beginning
        coil_x = coil[:, 0]
        coil_y = coil[:, 1]
        coil_z = coil[:, 2]
        mlab.plot3d(coil_x, coil_y, coil_z,
                    tube_radius=0.05,
                    #                name='Coil %i' % int(lens_num/2),
                    color=(0, 0, 0))


class Deflector(Lens):
    def __init__(self, d, I_y, I_z, n_turns=None, r=None):
        super(Deflector, self).__init__(width=d,
                                        coil_turns=n_turns,
                                        radius=r,
                                        lens_type='deflector')
        self.I_y = I_y
        self.I_z = I_z


    def getB(self, x):
        return self.getB_y(x) + self.getB_z(x)

    def getB_y(self, x):
        trans = np.vstack((self.n, self.l, self.m))

        B = self.l * np.ones((len(x[:]), 3))  # number of x-es = 1, for ray-tracing,
        # number of x-es = a lot, for field vis

        B[:, 1:] = B[:, 1:] * np.array(self.I_y, dtype=float) * 2e-4  # B*I*mu*n
        # Rotate the field back in the lab's frame
        B = np.dot(B, trans)
        return B

    def getB_z(self, x):
        # Translate the coordinates in the deflector's frame
        trans = np.vstack((self.n, self.l, self.m))

        B = self.m * np.ones((len(x[:]), 3))  # number of x-es = 1, for ray-tracing,
        # number of x-es = a lot, for field vis

        B[:, 1:] = B[:, 1:] * np.array(self.I_z, dtype=float) * 2e-4  # B*I*mu*n
        # Rotate the field back in the lab's frame
        B = np.dot(B, trans)
        return B

    def display(self, half=None):
        if half == None:
            half = 0

        x = np.linspace(self.beginning[0], self.ending[0], 30)
        y = np.linspace(self.beginning[1], self.ending[1], 30)

        x, y = np.meshgrid(x, y)
        x, y = x.T, y.T
        z = self.R * np.ones((30, 30))

        mlab.surf(x, y, z,
                  #                name='Deflector %i' % int(lens_num/2),
                  #                warp_scale= 'auto',
                  color=(0, 0, 0))
        mlab.surf(x, y, -z,
                  #                  warp_scale= 'auto',
                  color=(0, 0, 0))


class Space(Lens):
    def __init__(self, d, detector = False,
                                    detector_position = None):
        super(Space, self).__init__(width=d,
                                    lens_type='space')

        self.coil_at_beginning = False
        self.coil_at_end = False
        if detector == True and detector_position is None:
            self.detector_position = 0.5
        elif detector == True and detector_position is not None:
            self.detector_position = detector_position
        elif detector == False:
            self.detector_position = None

    def GetExtraParams(self,
                       I_start, I_end,
                       n_start, n_end,
                       r_start, r_end,
                       lens_type_begin,
                       lens_type_end):

        if lens_type_begin == 'coil':
            self.coil_at_beginning = True
            self.current = I_start
            self.coil_turns = n_start
            self.R = r_start
        if lens_type_end == 'coil':
            self.coil_at_end = True
            self.current = I_end
            self.coil_turns = n_end
            self.R = r_end
        if lens_type_begin != 'coil' and lens_type_end != 'coil':
            self.current = None
            self.coil_turns = None
            self.R = None

    def getB(self, position_vector):
        """
        Arguments:
            position vector: current position of electron.
                array; shape: (1, 3) or (n, 3) - simulator and field visuals use the same function (would be pointless if
                they didn't).
        Returns:
            B: a vector for the B field at given positions.
                array; shape like position_vector.

            """
        if self.R == None:
            return np.c_[0, 0, 0]
        ### Translate the coordinates in the coil's frame
        if self.coil_at_beginning is True:
            position = position_vector - self.beginning
        if self.coil_at_end is True:
            position = position_vector - self.ending
        # transformation matrix coil frame to lab frame
        trans = np.vstack((self.n, self.l, self.m))  # unit vectors
        # transformation matrix to lab frame to coil frame
        inv_trans = np.linalg.inv(trans)
        # transform vector to coil frame
        position = np.dot(position, inv_trans)

        if position.ndim == 1:
            x = self.n.dot(position) # on-axis distance
            y = self.l.dot(position)
            z = self.m.dot(position)
            if y != 0:
                theta = np.arctan((z / y))
            else:
                theta = 0
        else:
            x = np.einsum('j, ij->i', self.n, position)
            y = np.einsum('j, ij->i', self.l, position)
            z = np.einsum('j, ij->i', self.m, position)
            theta = np.arctan((z / y))
            theta[y == 0] = 0

        B = magnetic_fields.coil_field(x, y, z, self.R)
        B = -self.coil_turns  *12.56637e-7* (
                    np.asarray(self.current) * np.dot(B, trans) )  # without wasting memory: n*mu*I*B
        return B

    def read_detector(self, simulation):
        det_position = self.beginning + self.detector_position*self.width
        detector = Detector(simulation, det_position, self.n)

        return detector.detect()

    def display(self):
        if self.detector_position is not None:
            det_position = self.beginning + self.detector_position * self.width
            const = -det_position.dot(self.n)
            y = y = np.linspace(-self.R/2, self.R/2, 10)
            y, z = np.meshgrid(y, y)
            x = (-self.n[1]*y - self.n[2]*z - const) * (1./self.n[0])
            mlab.surf(x, y, z,
                      transparent=True, opacity=.2)


    class Splitter(Lens):
        def __init__(self, beginning, I_inner, I_outer):
            super(Lens, self).__init__()
            self.elements = []
            self.I_inner = I_inner
            self.I_outer = I_outer
            self.n_inner = np.r_[0.6427876, 0.766, 0] # cos50, sin50, 0
            self.n_outer = np.r_[0.86602541, 0.5, 0.] # cos30, sin30, 0
            self.beginning = beginning

        def inner_select(self):
            B = self.l*self.current_inner
            return B

        def outer_select(self):
            B = self.l*self.current_outer
            return B

        def inside_inner(self):
            return self.n_inner.dot(self.inner_plane_1) <= np.dot(self.n, pos) <= self.n.dot(self.inner_plane_2)
