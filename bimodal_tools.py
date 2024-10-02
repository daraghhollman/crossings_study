"""
Tools to aid in the investigation of the bimodal nature of the magnetic field at Mercury's magnetospheric boundaries.
"""


import matplotlib.pyplot as plt
from numpy import ma


def Create_Axes():
    """Sets up and sizes a matplotlib figure.

    Sets up a matplotlib figure with axes for trajectory, mag, and a histogram.


    Parameters
    ----------
    None


    Returns
    -------
    fig : matplotlib.pyplot.Figure
        The figure

    trajectory_axes : list[matplotlib.pyplot.Axes]
        List (length 2) of axes to plot the trajectory onto.

    mag_axes : list[matplotlib.pyplot.Axes]
        List (length 2) of axes to plot mag data onto.

    histogram_axis : matplotlib.pyplot.Axes
        Axis to plot histogram data onto.
    """

    fig = plt.figure(figsize=(28, 14))

    ax1 = plt.subplot2grid((4, 4), (0, 0), colspan=1, rowspan=2)
    ax2 = plt.subplot2grid((4, 4), (0, 1), colspan=1, rowspan=2)
    trajectory_axes = [ax1, ax2]

    ax3 = plt.subplot2grid((4, 4), (2, 0), colspan=2)
    ax4 = plt.subplot2grid((4, 4), (3, 0), colspan=2)
    mag_axes = [ax3, ax4]

    ax5 = plt.subplot2grid((4, 4), (0, 2), colspan=2, rowspan=4)
    histogram_axis = ax5

    return fig, trajectory_axes, mag_axes, histogram_axis
