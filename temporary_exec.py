# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 15:18:39 2019

@author: c1545871
"""

import all_lenses as lens
from microscope import Microscope, ImagingColumn, IlluminationColumn
from simulator import Simulate
from visualizer import Visualizer

'''Current working procedure is:
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
'''
LEEM = Microscope()
Img_Column = ImagingColumn()
Ill_Column = IlluminationColumn()

Ill_Column.append(lens.Space(d=10.))


CL1 = lens.Coil(d=2., I=250., r=100.)

Ill_Column.append(CL1)

Ill_Column.append(lens.Space(d=20.))

CL1_defl = lens.Deflector(d=5, I=100.)
Ill_Column.append(CL1_defl)

Space_1 = lens.Space(d=20.)
Ill_Column.append(Space_1)


CL2 = lens.Coil(d=2., I=250., r=100.)

Ill_Column.append(CL2)

Ill_Column.append(lens.Space(d=20.))

CL2_defl = lens.Deflector(d=5., I=-100.)
Ill_Column.append(CL2_defl)

Space_2 = lens.Space(d=10.)
Ill_Column.append(Space_2)

LEEM.append(Ill_Column)
LEEM.build_microscope()

simulation = Simulate(LEEM)
simulation.create_beam(24, 5)
simulation.simulate_all()
visual = Visualizer(LEEM, simulation)
visual.create()
