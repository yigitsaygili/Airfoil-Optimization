from cst_solver import *
import random

wu = [0.3, 0.3, 0.2]
wl = [-0.2, 0.1, 0.0]

wu = [0.48, 0.32, 0.4]
wl = [-0.16, -0.28, -0.08]
cst_solve(wu, wl)
 
print(round(random.uniform(-0.5, 0.5),2))

import matplotlib.pyplot as plt
import numpy as np

# Create sample data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Create a figure and axes
fig, ax = plt.subplots()

# Create a gradient background
# Create a 2D array for the gradient (e.g., from 0 to 1)
gradient = np.linspace(0, 1, 256)
gradient = np.vstack((gradient, gradient))

# Display the gradient as an image
ax.imshow(gradient, aspect='auto', cmap='Blues', extent=[0, 10, -1, 1])

# Plot the data on top of the gradient
ax.plot(x, y, color='orange', label='Sine Wave')

# Add a legend
ax.legend()

# Set the limits and labels
ax.set_xlim(0, 10)
ax.set_ylim(-1, 1)
ax.set_title('Plot with Gradient Background')

# Show the plot
plt.show()
