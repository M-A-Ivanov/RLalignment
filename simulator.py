import numpy as np
from scipy.integrate import ode
from utils import beam_creator


class Simulate:
    def __init__(self, microscope):
        self.velocities_all = []
        self.positions_all = []
        self.microscope = microscope
        self.dt = 0.2
        self.results = []
        self.ic_list = []

    def create_beam(self, n_rays, cone_angle):
        self.ic_list = beam_creator(n_rays, cone_angle, self.microscope.elements[0].n)  # initial conditions

    @staticmethod
    def lorenz(t, y, field):
        deflection = np.cross(y[3:], field)
        return np.array([y[3:], deflection[0, :]]).ravel()

    def one_step_forward(self, r, y0, t0, B):
        """Does the loop for the simulation calculation separately here.
        Might be redundant now that the -simulate- f-n is better written"""

        r.set_initial_value(y0, t0).set_f_params(B)
        if r.successful():
            r.integrate(r.t + self.dt)
            # self.positions.append(r.y[:3])
            # self.velocities.append(r.y[3:])
            return r.y
        
    def simulate(self, ic):
        """Parameters:
                ic: (initial conditions) list of arrays (2, 3) = [x0(1, 3), v0(1, 3)]
           Returns:
                list (2, n): 1D list of positions (n,), 1D list of velocities (n,)
                """
        # self.positions = [initial_conditions[0]]
        # self.velocities = [initial_conditions[1]] 
        positions = [ic[0]]
        velocities = [ic[1]]
        r = ode(self.lorenz).set_integrator('dopri5')
        initial_conditions = np.concatenate(ic)  # make it into a single array, temporary?
        # add some free travel in beginning as an initial condition (should be removed later!!!)
        y = self.one_step_forward(r, initial_conditions, 0, np.zeros((1, 3)))
        positions.append(y[:3])
        velocities.append(y[3:])
        # ray-tracing:
        t_max = 150
        flag = 0
        while r.t < t_max and flag == 0:  # go through all lenses
            B = self.microscope.getB(positions[-1])
            if B is np.c_[0, 0, 0]:
                flag = 1
            y = self.one_step_forward(r, r.y, r.t, self.microscope.getB(positions[-1]))
            positions.append(y[:3])
            velocities.append(y[3:])
        return positions, velocities
    
    def simulate_all(self):
        for i in range(0, np.shape(self.ic_list)[0]):
            positions, velocities = self.simulate(self.ic_list[i])
            self.positions_all.append(positions)
            self.velocities_all.append(velocities)
        self.positions_all = np.array(self.positions_all)
        self.velocities_all = np.array(self.velocities_all)

