import numpy as np
import matplotlib.pyplot as plt
from IPython import display
import time

np.random.seed(seed=1001)
x = np.linspace(-2, 2, 4)

m_true = 1.4
c_true = -3.1

y = m_true * x + c_true

plt.plot(x, y, 'r.', markersize=10)
plt.xlim([-3, 3])
plt.show()

#
# noise corrupt plot
np.random.seed(seed=22050)
noise = np.random.normal(scale=0.5, size=4)  # standard deviation of the noise is 0.5
y = m_true * x + c_true + noise
plt.plot(x, y, 'r.', markersize=10)
plt.xlim([-3, 3])
plt.show()

#
# Visualise the error function surface, create vectors of values.
# create an array of linearly seperated values around m_true
m_vals = np.linspace(m_true - 3, m_true + 3, 100)
# create an array of linearly seperated values around c_true
c_vals = np.linspace(c_true - 3, c_true + 3, 100)
# create a grid of values to evaluate the error function in 2D
m_grid, c_grid = np.meshgrid(m_vals, c_vals)
# compute the error function at each combination point of c and m
E_grid = np.zeros((100, 100))
for i in range(100):
    for j in range(100):
        E_grid[i, j] = ((y - m_grid[i, j] * x - c_grid[i, j]) ** 2).sum()


# draw contour plot of error
def regression_contour(f, ax, m_vals, c_vals, E_grid):
    hcont = ax.coutour(m_vals, c_vals, E_grid, levels=[0, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64])
    plt.clabel(hcont, inline=1, fontsize=10)  # this labels the contours

    ax.set_xlabel('$m$', fontsize=20)
    ax.set_ylabel('$c$', fontsize=20)


f, ax = plt.subplot(figsize=(5, 5))
regression_contour(f, ax, m_vals, c_vals, E_grid)
