import matplotlib.pyplot as plt
import numpy as np

# Visualization Utility Functions
def ax_default(fignum, ax):
    """Utility function for either creating a new subplot or returning a particular axis"""
    if ax is None:
        fig = plt.figure(fignum)
        ax = fig.add_subplot(111)
    else:
        fig = ax.figure
    return fig, ax

def meanplot(x, mu, color='#3300FF', ax=None, fignum=None, linewidth=2,**kw):
    """Plot a mean function in a given colour."""
    _, axes = ax_default(fignum, ax)
    return axes.plot(x,mu,color=color,linewidth=linewidth,**kw)

def gpplot(x,
           mu,
           lower,
           upper,
           edgecol='#3300FF',
           fillcol='#CC3300',
           ax=None,
           fignum=None,
           **kwargs):
    """Make a simple GP plot from a given mean, a lower and upper confidence bound"""
    _, axes = ax_default(fignum, ax)

    mu = mu.flatten()
    x = x.flatten()
    lower = lower.flatten()
    upper = upper.flatten()

    plots = []

    #here's the mean
    plots.append(meanplot(x, mu, edgecol, axes))

    #here's the box
    kwargs['linewidth']=0.5
    if not 'alpha' in kwargs.keys():
        kwargs['alpha'] = 0.3
    plots.append(axes.fill(np.hstack((x,x[::-1])),np.hstack((upper,lower[::-1])),color=fillcol,**kwargs))

    #this is the edge:
    plots.append(meanplot(x, upper,color=edgecol,linewidth=0.2,ax=axes))
    plots.append(meanplot(x, lower,color=edgecol,linewidth=0.2,ax=axes))

    return plots
