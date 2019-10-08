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

# Steepest Descent
# Minimize the sum of squares error function.
# One way of doing that is gradient descent.
# Initialize with a guess for  𝑚  and  𝑐
# Update that guess by subtracting a portion of the gradient from the guess.
# Like walking down a hill in the steepest direction of the hill to get to the bottom.

# we start from a guess for m and c
m_star = 0.0
c_star = -5.0

c_grad = -2 * (y - m_star * x - c_star).sum()
print("Gradient with respect to c is ", c_grad)

# Update Equations
# Now we have gradients with respect to  𝑚  and  𝑐 .
# We can update our inital guesses for  𝑚  and  𝑐  using the gradient.
# We don't want to just subtract the gradient from  𝑚  and  𝑐 ,
# We need to take a small step in the gradient direction.
# Otherwise we might overshoot the minimum.
# We want to follow the gradient to get to the minimum, the gradient changes all the time.


f, ax = plt.subplot(figsize=(7, 7))
regression_contour(f, ax, m_vals, c_vals, E_grid)
ax.plot(m_star, c_star, 'g*', markersize=15)
ax.arrow(m_star, c_star, -m_grid * .05, -c_star * 0.05,head_width = 0.15)
