# LEEM-simulation
Ray-tracing simulation of e-beam path through an electron microscope. 
Modelling a Low-Energy-Electron Microscope, consisting of 21 optical elements:
  -8 coil lenses
  -8 deflectors
  -3 stigmators
  -1 objective lens
  -1 beam splitter.
  
The simulation works by solving the ODE equation dv/dt = F_Lorenz/m and integrating over the result. It should show how the beam propagates through and how the magnetic fields look like. It should gather information for the beam at any given plane. 
  
Current known significant problems:
  - Field visualization is off;
  - Need to add multiprocessing (problem is embarrassingly parallel);
  - Need to complete models for beam splitter and objective lens to gain a functional model.
  - Need to make units realistic?
  - Add methods that vary parameters instead of redoing the whole LEEM.
    
Minor issues:
  - The last one of the simulated beams "doesn't see" the magnetic fields;
  - Need to add a functionality that cuts the simulation once a beam has been through all elements, it now stops at chosen amount of steps;
  - detector doesnt draw properly on the x-axis - will put it as a disk to make visualization easier;
  - Need to add a small initial beam displacement. Now it is perfectly aimed to the optical axis.
