# Airfoil-Optimization
An aerodynamic shape optimization algorithm for 2D airfoils.

SHAPE PARAMETRIZATION:
  2D Airfoil geometries are currently parametrized as NACA 4 digit airfoils with three parameters;
  - m: Maximum camber of the airfoil divided by its chord
  - p: Maximum camber location of the airfoil along its chord
  - t: Maximum thickness of the airfoil divided by its chord

AERODYNAMIC ANALYSIS:
  Aerodynamic characteristics of the airfoil is currently solved by XFOIL tool, the outputs are;
  - Aerodynamic coefficients, CL, CDP, CD, CM
  - Pressure distribution of the upper and lower surfaces
  - Airfoil curve in x-y coordinates

OPTIMIZATION ALGORITHM:
  Optimization is based on maximizing CL/CD value. The genetic algorithm used contains;
  - Crossover and mutation rates that can be rearranged
  - Tournament selection method to define new generations
  - Fitness history and best individual log

DISPLAY FORMAT:
  Analysis and optimization processes are visualized for better ease of operation including;
  - Displaying airfoil curve upper and lower surfaces seperated
  - Pressure distribution over its surfaces with transiiton points
  - Real time fitness history and best individuals of each generation

    ![naca4optf1](https://github.com/user-attachments/assets/cc3aafba-fa43-4f81-8ca9-19b4e794e1e0)

TO  BE ADDED:
  - Using CST parametrization instead of NACA 4 digit airfoils 
  - Better handling of XFOIL and its outputs to increase performance
  - Improvised genetic algorithm with adaptive mutation and elitism factor
  - Graphic used interface to maintain the optimization proccess better
