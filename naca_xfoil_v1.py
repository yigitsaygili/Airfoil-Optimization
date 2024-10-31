import os
import subprocess
import numpy as np

def naca_solver(m, p, t):
    naca_code = f"{m}{p}{t:02d}"    # NACA 4 digit code to be inputted
    alpha = 0                       # Angle of attack
    Re = 100000                     # Reynolds number
    n_iter = 1000                   # Iteration count
    is_viscous = False              # Viscosity parameter

    # Remove old files if they exist
    for file_name in ["input_file.in", "polar_file.txt", "geometry_file.txt", "pressure_file.txt"]:
        if os.path.exists(file_name):
            os.remove(file_name)
    
    try:
        if is_viscous == True:
            # Write the XFOIL input file
            with open("input_file.in", 'w') as input_file:
                input_file.write(f"NACA {naca_code}\n")
                input_file.write("PANE\n")
                input_file.write("PSAV geometry_file.txt\n")
                input_file.write("OPER\n")
                input_file.write(f"Visc {Re}\n")
                input_file.write("PACC\n")
                input_file.write("polar_file.txt\n\n")
                input_file.write(f"ITER {n_iter}\n")
                input_file.write(f"ALFA {alpha}\n")
                input_file.write("CPWR pressure_file.txt\n")
                input_file.write("\n\n")
                input_file.write("quit\n")
        else:
            # Write the XFOIL input file
            with open("input_file.in", 'w') as input_file:
                input_file.write(f"NACA {naca_code}\n")
                input_file.write("PANE\n")
                input_file.write("PSAV geometry_file.txt\n")
                input_file.write("OPER\n")
                input_file.write("PACC\n")
                input_file.write("polar_file.txt\n\n")
                input_file.write(f"ITER {n_iter}\n")
                input_file.write(f"ALFA {alpha}\n")
                input_file.write("CPWR pressure_file.txt\n")
                input_file.write("\n\n")
                input_file.write("quit\n")

        # Run XFOIL with the generated input file
        with open(os.devnull, 'wb') as devnull:
            subprocess.run("xfoil.exe < input_file.in", shell=True, stdout=devnull, stderr=devnull, timeout=10)

        # Read data
        polar_data = np.loadtxt("polar_file.txt", skiprows=12)
        geometry_data = np.loadtxt("geometry_file.txt", skiprows=1)
        pressure_data = np.loadtxt("pressure_file.txt", skiprows=3)

        cl = polar_data[1]
        cdp = polar_data[3]
        cl_over_cd = cl / abs(cdp)

    except Exception as e:
        print(f"Error in XFOIL: {e}")
        # Assign a very bad fitness value when XFOIL fails
        cl_over_cd = -1e6  # Penalty for divergence or error

    # Clean up
    for file_name in ["input_file.in", "polar_file.txt", "geometry_file.txt", "pressure_file.txt"]:
        os.remove(file_name)
    
    return cl_over_cd
