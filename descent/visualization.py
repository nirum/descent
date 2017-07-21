"""
Visualization tools
"""
import numpy as np
import matplotlib.pyplot as plt


def surface(f, xlim, ylim, npts=50, method=plt.contourf, cmap='magma_r'):
    x = np.linspace(*xlim, npts)
    y = np.linspace(*ylim, npts)
    xm, ym = np.meshgrid(x, y)
    zm = f((xm.ravel(), ym.ravel())).reshape(npts, npts)
    return method(xm, ym, zm, cmap=cmap, aspect='equal')
