import numpy as np

# Create a simple 1D array
arr = np.array([1, 2, 3, 4, 5])
print("NumPy Array:", arr)

import scipy
print("SciPy version:", scipy.__path__)

import matplotlib.pyplot as plt
import numpy as np

# Generate sample data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Create the plot
plt.plot(x, y)
plt.title("Matplotlib is Working!")
plt.xlabel("X Axis")
plt.ylabel("Y Axis")

# Display the graph
plt.show()
