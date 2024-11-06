import os
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
from scipy.optimize import minimize

# %% [1] FUNCTION FOR AIRFOIL GENERATION WITH CST PARAMETRIZATION
def cst_airfoil(wl, wu):
    # VARIABLES
    dz = 0.001
    N = 100

    # N1 and N2 parameters (N1 = 0.5 and N2 = 1 for airfoil shape)
    N1 = 0.5
    N2 = 1

    # COORDINATES
    x = np.ones(N + 1)
    y = np.zeros(N + 1)
    zeta = np.zeros(N + 1)

    for i in range(N + 1):
        zeta[i] = 2 * np.pi / N * i
        x[i] = 0.5 * (np.cos(zeta[i]) + 1)

    zerind = np.where(x == 0)[0][0]  # Used to separate upper and lower surfaces

    xl = x[:zerind]  # Lower surface x-coordinates
    xu = x[zerind:]  # Upper surface x-coordinates

    yl = class_shape(wl, xl, N1, N2, -dz)  # Determine lower surface y-coordinates
    yu = class_shape(wu, xu, N1, N2, dz)   # Determine upper surface y-coordinates

    y = np.concatenate((yl, yu))  # Combine upper and lower y coordinates

    x_coord = x
    y_coord = y

    return x_coord, y_coord


# %% [2] FUNCTION FOR CLASS SHAPE TRANSFORMATION
def class_shape(w, x, N1, N2, dz):
    # Class function; taking input of N1 and N2
    C = np.array([xi**N1 * (1 - xi)**N2 for xi in x])

    # Shape function; using Bernstein Polynomials
    n = len(w) - 1  # Order of Bernstein polynomials

    K = np.array([comb(n, i) for i in range(n + 1)])

    S = np.zeros(len(x))
    for i in range(len(x)):
        for j in range(n + 1):
            S[i] += w[j] * K[j] * x[i]**j * (1 - x[i])**(n - j)

    # Calculate y output
    y = C * S + x * dz

    return y


# %% [3] FUNCTION FOR FINDING REVERSE SURFACE COEFFICIENTS
def cst_reverse(x_coord, y_coord, n_weights=3):
    # Convert inputs to numpy arrays
    x_coord = np.asarray(x_coord)
    y_coord = np.asarray(y_coord)

    # Variables
    N1 = 0.5
    N2 = 1

    # Split x-coordinates to lower and upper surfaces
    zerind = np.where(np.isclose(x_coord, 0))[0][0]
    xl = x_coord[:zerind]  # Lower surface x-coordinates
    xu = x_coord[zerind:]  # Upper surface x-coordinates
    yl = y_coord[:zerind]  # Lower surface y-coordinates
    yu = y_coord[zerind:]  # Upper surface y-coordinates

    # Objective function to minimize
    def objective_function(weights):
        wl = weights[:n_weights]
        wu = weights[n_weights:]
        yl_pred = class_shape(wl, xl, N1, N2, -0.001)  # Predicted lower surface
        yu_pred = class_shape(wu, xu, N1, N2, 0.001)    # Predicted upper surface
        return np.sum((yl - yl_pred) ** 2) + np.sum((yu - yu_pred) ** 2)

    # Initial guess for weights (zeros)
    initial_weights = np.zeros(2 * n_weights)

    # Minimize the objective function
    result = minimize(objective_function, initial_weights)

    # Extract weights for lower and upper surfaces
    weights_lower = result.x[:n_weights].tolist()
    weights_upper = result.x[n_weights:].tolist()

    return weights_upper, weights_lower


# %% [4] FUNCTION FOR AERODYNAMIC ANALYSIS WITH XFOIL
def cst_solver(x_coords, y_coords, alpha=0, is_viscous=False):
    Re = 1000000    # Reynolds number
    n_iter = 100    # Iteration count
    
    # Generate airfoil coordinates file
    with open("airfoil_coords.txt", 'w') as coord_file:
        for x, y in zip(x_coords, y_coords):
            coord_file.write(f"{x:.6f} {y:.6f}\n")

    # Remove old files if they exist
    for file_name in ["polar_file.txt", "geometry_file.txt", "pressure_file.txt"]:
        if os.path.exists(file_name):
            os.remove(file_name)

    try:
        if is_viscous == True:
            # Write the XFOIL input file
            with open("input_file.in", 'w') as input_file:
                input_file.write("LOAD\n")
                input_file.write("airfoil_coords.txt\n\n")
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
                input_file.write("LOAD\n")
                input_file.write("airfoil_coords.txt\n\n")
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

    except Exception as e:
        print(f"Error in XFOIL: {e}")
        polar_data = []
        geometry_data = []
        pressure_data = []

    # Clean up
    for file_name in ["airfoil_coords.txt", "input_file.in", "polar_file.txt", "geometry_file.txt", "pressure_file.txt"]:
        os.remove(file_name)
    
    return polar_data, geometry_data, pressure_data


# %% [5] FUNCTION FOR VISUALIZING ANALYSIS RESULTS
def cst_plotter(polar_data, geometry_data, pressure_data):
    
    # Display aerodynamic data
    cl, cd, cdp, cm, top_xtr, bot_xtr = polar_data[1:7] 

    # Plot Airfoil Geometry
    x = geometry_data[:, 0]
    y = geometry_data[:, 1]

    # Find the index of the leading edge (minimum x value)
    leading_edge_index = np.argmin(x)
    x_upper = x[leading_edge_index:]
    y_upper = y[leading_edge_index:]
    x_lower = x[:leading_edge_index + 1]
    y_lower = y[:leading_edge_index + 1]

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(x_upper, y_upper, label='Upper Surface', color='b')
    plt.plot(x_lower, y_lower, label='Lower Surface', color='r')
    plt.title("Airfoil Geometry")
    plt.xlabel('x/c')
    plt.ylabel('y/c')
    plt.grid(True)
    plt.axis('equal')
    plt.legend()

    # Separate Pressure Distribution Data
    x_cp = pressure_data[:, 0]
    y_cp = pressure_data[:, 1]
    cp = -pressure_data[:, 2]

    # Find the index of the leading edge (minimum x value)
    leading_edge_index_cp = np.argmin(x_cp)
    x_cp_upper = x_cp[leading_edge_index_cp:]
    cp_upper = cp[leading_edge_index_cp:]
    x_cp_lower = x_cp[:leading_edge_index_cp + 1]
    cp_lower = cp[:leading_edge_index_cp + 1]

    plt.subplot(1, 2, 2)
    plt.plot(x_cp_upper, cp_upper, label='Upper Surface Cp', color='b')
    plt.plot(x_cp_lower, cp_lower, label='Lower Surface Cp', color='r')
    plt.title('Pressure Distribution')
    plt.xlabel('x/c')
    plt.ylabel('Cp')
    plt.grid(True)
    plt.legend()

    plt.figtext(0.5, 0.01, "CL: %.2f        CD: %.2f        CDP: %.2f       CM: %.2f        Top_Xtr: %.2f       Bot_Xtr: %.2f"\
                 %(cl, cd, cdp, cm, top_xtr, bot_xtr), ha='center', fontsize=12)

    plt.show()


# %% [6] FUNCTION FOR CALCULATING EQUIVALENT NACA DIGITS
def naca_params(x_coords, y_coords):
    # Find the index of the leading edge (minimum x value)
    leading_edge_index = np.argmin(x_coords)
    x_upper = x_coords[leading_edge_index:]
    y_upper = y_coords[leading_edge_index:]
    x_lower = x_coords[:leading_edge_index + 1]
    y_lower = y_coords[:leading_edge_index + 1]

    x_upper = x_upper[::-1]
    y_upper = y_upper[::-1]

    # Calculating the camber line
    x_camber = (x_upper+x_lower)/2
    y_camber = (y_upper+y_lower)/2

    # Calculating the thickness line
    x_thickness = (x_upper+x_lower)/2
    y_thickness = y_upper-y_lower

    # Calculate maximum camber and maximum thickness
    max_y_camber_index = np.argmax(y_camber)
    max_y_camber = y_camber[max_y_camber_index]
    max_x_camber = x_camber[max_y_camber_index]
    
    max_y_thickness_index = np.argmax(y_thickness)
    max_y_thickness = y_thickness[max_y_thickness_index]
    max_x_thickness = x_thickness[max_y_thickness_index]
    
    # Extract NACA code parameters
    chord = np.max(x_coords)
    m = round(max_y_camber / chord * 100)
    p = round(max_x_camber / chord * 10)
    t = round(max_y_thickness / chord * 100)
    naca_code = f"NACA {m}{p}{t:02d}"

    return m, p, t, naca_code

# %% [7] FUNCTION FOR CHECKING AIRFOIL GEOMETRIC CONSISTENCY
def cst_check(wu, wl):
    for wu_i, wl_i in zip(wu, wl):
        if wu_i <= wl_i:
            return False
            
    return True


# %% [8] FUNCTION CALCULATING PREDEFINED AIRFOIL SCORE
def cst_score(x_coords, y_coords, alpha=0, is_viscous=False):
    Re = 1000000    # Reynolds number
    n_iter = 100    # Iteration count
    
    # Generate airfoil coordinates file
    with open("airfoil_coords.txt", 'w') as coord_file:
        for x, y in zip(x_coords, y_coords):
            coord_file.write(f"{x:.6f} {y:.6f}\n")

    # Remove old files if they exist
    for file_name in ["polar_file.txt"]:
        if os.path.exists(file_name):
            os.remove(file_name)

    try:
        if is_viscous == True:
            # Write the XFOIL input file
            with open("input_file.in", 'w') as input_file:
                input_file.write("LOAD\n")
                input_file.write("airfoil_coords.txt\n\n")
                input_file.write("PANE\n")
                input_file.write("OPER\n")
                input_file.write(f"Visc {Re}\n")
                input_file.write("PACC\n")
                input_file.write("polar_file.txt\n\n")
                input_file.write(f"ITER {n_iter}\n")
                input_file.write(f"ALFA {alpha}\n")
                input_file.write("quit\n")
        else:
            # Write the XFOIL input file
            with open("input_file.in", 'w') as input_file:
                input_file.write("LOAD\n")
                input_file.write("airfoil_coords.txt\n\n")
                input_file.write("PANE\n")
                input_file.write("OPER\n")
                input_file.write("PACC\n")
                input_file.write("polar_file.txt\n\n")
                input_file.write(f"ITER {n_iter}\n")
                input_file.write(f"ALFA {alpha}\n")
                input_file.write("quit\n")

        # Run XFOIL with the generated input file
        with open(os.devnull, 'wb') as devnull:
            subprocess.run("xfoil.exe < input_file.in", shell=True, stdout=devnull, stderr=devnull, timeout=10)

        # Read data
        polar_data = np.loadtxt("polar_file.txt", skiprows=12)
        score = polar_data[1]

    except Exception as e:
        print(f"Error in XFOIL: {e}")
        score = -1e6

    # Clean up
    for file_name in ["airfoil_coords.txt", "input_file.in", "polar_file.txt"]:
        os.remove(file_name)
    
    return score


# %% [9] FUNCTION TO VISUALIZING AIRFOIL GEOMETRY
def cst_geometry(x_coord, y_coord):

    # Find the index of the leading edge (minimum x value)
    leading_edge_index = np.argmin(x_coord)
    x_upper = x_coord[leading_edge_index:]
    y_upper = y_coord[leading_edge_index:]
    x_lower = x_coord[:leading_edge_index + 1]
    y_lower = y_coord[:leading_edge_index + 1]

    x_upper = x_upper[::-1]
    y_upper = y_upper[::-1]

    # Calculating the camber line
    x_camber = (x_upper+x_lower)/2
    y_camber = (y_upper+y_lower)/2

    # Calculating the thickness line
    x_thickness = (x_upper+x_lower)/2
    y_thickness = y_upper-y_lower

    plt.plot(x_upper, y_upper, label='Upper Surface', color='b')
    plt.plot(x_lower, y_lower, label='Lower Surface', color='r')
    plt.plot(x_camber, y_camber, label='Camber Line', color='k')
    plt.plot(x_thickness, y_thickness, label='Thickness Line', color='k', linestyle='--')
    plt.title("Airfoil Geometry")
    plt.xlabel('x/c')
    plt.ylabel('y/c')
    plt.grid(True)
    plt.axis('equal')
    plt.legend()


# %% [0] MAIN OPERATION FUNCTION
def cst_solve(wu, wl):
    x_coord, y_coord = cst_airfoil(wl, wu)
    polar_data, geometry_data, pressure_data = cst_solver(x_coord, y_coord)
    m, p, t, naca_code = naca_params(x_coord, y_coord)
    wu_r, wl_r = cst_reverse(x_coord, y_coord)
    consistency = cst_check(wu, wl)
    score = cst_score(x_coord, y_coord)
    cst_geometry(x_coord, y_coord)
    cst_plotter(polar_data, geometry_data, pressure_data)

    return wu, wl, x_coord, y_coord, polar_data, geometry_data, pressure_data, m, p, t, naca_code, wu_r, wl_r, consistency, score

