import numpy as np
import matplotlib.pyplot as plt
from IPython import display
import time

n_data_per_class = 30
np.random.seed(seed=1000001)  # np.random.seed()make the random predictable
# With the seed reset (every time), the same set of numbers will appear every time.
x_plus = np.random.normal(loc=1.3, size=(n_data_per_class, 2))  # the output shape is n_data_per_class * 2
x_minus = np.random.normal(loc=-1.3, size=(n_data_per_class, 2))

# plot data
# figsize : figure dimension (width, height) in inches.
# dpi : dots per inch
plt.figure(figsize=(5, 5), dpi=80)
xlim = np.array([-3.5, 3.5])
ylim = xlim
# x_plus[:,0] means all rows's 0th element
plt.plot(x_plus[:, 0], x_plus[:, 1], 'rx')  # red, x dot
plt.plot(x_minus[:, 0], x_minus[:, 1], 'go')  # green o dot
plt.xlim(xlim[0], xlim[1])
plt.ylim(ylim[0], ylim[1])
plt.xlabel(r'$x_1$', fontsize=14)
plt.ylabel(r'$x_2$', fontsize=14)


# plt.show()


# Routine to keep the margins of the drawing box fixed and get the correct margin points for computing the mid points
# later
def margins_plot(x2, xlim, ylim, w, b):
    if (- w[0] / w[1]) > 0:  # cases for a positive slope
        # xlim = np.flip(xlim, 0)
        # ylim = np.flip(ylim, 0)
        if np.max(x2) > ylim[1] and np.min(x2) < ylim[0]:
            x_margin_neg = (ylim[1] + (b / w[1])) / (- w[0] / w[1])
            x_margin_pos = (ylim[0] + (b / w[1])) / (- w[0] / w[1])
            y_margin_neg = ylim[1]
            y_margin_pos = ylim[0]
        if np.max(x2) < ylim[1] and np.min(x2) > ylim[0]:
            x_margin_neg = xlim[1]
            x_margin_pos = xlim[0]
            y_margin_neg = (- w[0] / w[1]) * xlim[1] - (b / w[1])
            y_margin_pos = (- w[0] / w[1]) * xlim[0] - (b / w[1])
        if np.max(x2) > ylim[1] and np.min(x2) > ylim[0]:
            x_margin_neg = (ylim[1] + (b / w[1])) / (- w[0] / w[1])
            x_margin_pos = xlim[0]
            y_margin_neg = ylim[1]
            y_margin_pos = (- w[0] / w[1]) * xlim[0] - (b / w[1])
        if np.max(x2) < ylim[1] and np.min(x2) < ylim[0]:
            x_margin_neg = xlim[1]
            x_margin_pos = (ylim[0] + (b / w[1])) / (- w[0] / w[1])
            y_margin_neg = (- w[0] / w[1]) * xlim[1] - (b / w[1])
            y_margin_pos = ylim[0]
    else:
        if np.max(x2) > ylim[1] and np.min(x2) < ylim[0]:
            x_margin_neg = (ylim[0] + (b / w[1])) / (- w[0] / w[1])
            x_margin_pos = (ylim[1] + (b / w[1])) / (- w[0] / w[1])
            y_margin_neg = ylim[0]
            y_margin_pos = ylim[1]
        if np.max(x2) < ylim[1] and np.min(x2) > ylim[0]:
            x_margin_neg = xlim[1]
            x_margin_pos = xlim[0]
            y_margin_neg = (- w[0] / w[1]) * xlim[1] - (b / w[1])
            y_margin_pos = (- w[0] / w[1]) * xlim[0] - (b / w[1])
        if np.max(x2) > ylim[1] and np.min(x2) > ylim[0]:
            x_margin_neg = xlim[1]
            x_margin_pos = (ylim[1] + (b / w[1])) / (- w[0] / w[1])
            y_margin_neg = (- w[0] / w[1]) * xlim[1] - (b / w[1])
            y_margin_pos = ylim[1]
        if np.max(x2) < ylim[1] and np.min(x2) < ylim[0]:
            x_margin_neg = (ylim[0] + (b / w[1])) / (- w[0] / w[1])
            x_margin_pos = xlim[0]
            y_margin_neg = ylim[0]
            y_margin_pos = (- w[0] / w[1]) * xlim[0] - (b / w[1])
    return x_margin_neg, x_margin_pos, y_margin_neg, y_margin_pos


# Routine for plotting
# draw the descision boundary, data points, the perceptron arrow
def plot_perceptron(w, b):
    npoints = 100
    xlim = np.array([-3.5, 3.5])
    ylim = xlim

    x1 = np.linspace(xlim[0], xlim[1], npoints)
    x2 = (- w[0] / w[1]) * x1 - (b / w[1])  # w0 * x1 + w1m * x2 = -b

    # the cooridiantes of the decision boundary 决策线的起止点坐标
    x_margin_neg, x_margin_pos, y_margin_neg, y_margin_pos = margins_plot(x2, xlim, ylim, w, b)
    x1c = (x_margin_neg + x_margin_pos) / 2  # Horizontal coordinates of the intermediate point 中间点的横坐标
    x2c = (y_margin_neg + y_margin_pos) / 2  # vertical coordinates of the intermediate point 中间点的纵坐标
    x2per = (w[1] / w[0]) * x1 - (w[1] / w[0]) * x1c + x2c

    # plt.axes()
    #     display.clear_output(wait=True) # clear outputs inline

    # draw the data dots
    plt.figure(figsize=(5, 5), dpi=80)
    plt.plot(x_plus[:, 0], x_plus[:, 1], 'rx')
    plt.plot(x_minus[:, 0], x_minus[:, 1], 'go')
    plt.xlabel(r'$x_1$', fontsize=14)
    plt.ylabel(r'$x_2$', fontsize=14)

    # draw the decision line
    plt.plot(x1, x2, 'b')

    # draw the vertical line of the decision boundary
    #     plt.plot(x1, x2per, '--k', color='whitesmoke')
    plt.xlim(xlim[0], xlim[1])
    plt.ylim(ylim[0], ylim[1])
    # draw the arrow
    # base x, base y, length along x,length along y,head_length, fc: face color, ec: edge color
    plt.arrow(x1c, x2c, w[0], w[1], width=0.03, head_width=0.3, head_length=0.3, fc='k', ec='k')


#     display.display(plt.gcf())


# np.random.seed(seed=1001)
w = 0.5 * np.random.randn(2)
b = 0.5 * np.random.randn()
plot_perceptron(w, b)
x_selected = x_plus[1]
# mfc:marker face color, mec:marker edge color, ms:marker sizd, lw:linewidth
plt.plot(x_selected[0], x_selected[1], 'o', mfc='none', mec='k', ms=15, lw=5)
# The sign function returns -1 if x < 0, 0 if x==0, 1 if x > 0. nan is returned for nan inputs.
print("The sign for the selected value is ", np.sign(np.dot(w, x_selected) + b))
plt.show()


def update_perceptron(w, b, x_plus, x_minus, learn_rate):
    # update the perceptron
    # select a point at random from the data
    choose_plus = np.random.uniform(size=1) > 0.5
    updated = False
    if choose_plus:
        # choose a point from the positive data
        index = np.random.randint(x_plus.shape[0])
        x_select = x_plus[index, :]
        if np.dot(w, x_select) + b <= 0:
            # point is currently incorrectly classified
            w += learn_rate * x_select
            b += learn_rate
            updated = True
    else:
        # choose a point from the negative data
        index = np.random.randint(x_minus.shape[0])
        x_select = x_minus[index, :]
        if np.dot(w, x_select) + b > 0:
            # point is currently incorrectly classified
            w -= learn_rate * x_select
            b -= learn_rate
            updated = True

    return w, b, x_select, updated


def run_perceptron(w, b, learn_rate, how_many):
    for i in range(how_many):
        plot_perceptron(w, b)
        w, b, x_selected, updated = update_perceptron(w, b, x_plus, x_minus, learn_rate)
        plt.plot(x_selected[0], x_selected[1], 'o', mfc='none', mec='k', ms=15, lw=15)
        if updated:
            plt.title('Incorrect classification: needs updating', {'fontsize': 15})
            plt.pause(5)
        else:
            plt.title('Correct classification', {'fontsize': 15})
            plt.pause(2)

