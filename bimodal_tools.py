"""
Tools to aid in the investigation of the bimodal nature of the magnetic field at Mercury's magnetospheric boundaries.
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import scipy.signal
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from numpy.typing import ArrayLike
from scipy.optimize import curve_fit

from plotting_tools import colored_line, truncate_colormap


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


class Population_Fit:
    def __init__(
        self, x_values: ArrayLike, y_values: ArrayLike, guess_params: list[float]
    ):
        self.x_values = x_values
        self.y_values = y_values
        self.guess_params = guess_params
        self.pars, self.cov = self.Get_CurveFit()

        self.population_a_pars = self.pars[:3]
        self.population_a_uncertainties = np.sqrt(np.diag(self.cov))[:3]
        self.population_b_pars = self.pars[3:]
        self.population_b_uncertainties = np.sqrt(np.diag(self.cov))[3:]

        # Useful for plotting
        self.x_range = np.linspace(x_values[0], x_values[-1], 1000)
        self.output_y_values = Double_Gaussian(
            self.x_range,
            *self.pars,
        )
        self.population_a = Single_Gaussian(self.x_range, *self.population_a_pars)
        self.population_b = Single_Gaussian(self.x_range, *self.population_b_pars)

    def Get_CurveFit(self):

        pars, cov = curve_fit(
            Double_Gaussian, self.x_values, self.y_values, self.guess_params
        )

        return pars, cov


def Split_Distribution(
    mag_data: pd.DataFrame,
    distribution_values: list[float],
    fit: Population_Fit,
    fig: plt.Figure,
    mag_axes: list[plt.Axes],
    histogram_axis: plt.Axes,
    method="minimum",
):
    """
    INSERT DOCSTRING HERE
    """

    match method:

        case "likelihood":

            # For each point in the data, we determine the likelihood that it is
            # in distribution A and in distribution B.
            # We take the most likely option as the direction (towards -1 or +1),
            # and the value between 0 and +/-1.
            probabilities_a = []
            probabilities_b = []
            for field_value in distribution_values:

                # Find the likelihood we're in distribution A and B
                a_likelihood = Single_Gaussian(field_value, fit.population_a_pars[0] + 3 * fit.population_a_uncertainties[0], fit.population_a_pars[1] + 3 * fit.population_a_uncertainties[1], fit.population_a_pars[2])
                b_likelihood = Single_Gaussian(field_value, fit.population_b_pars[0] - 3 * fit.population_b_uncertainties[0], fit.population_b_pars[1] + 3 * fit.population_b_uncertainties[1], fit.population_b_pars[2])

                probability_point_in_a = a_likelihood / (a_likelihood + b_likelihood)
                probability_point_in_b = b_likelihood / (a_likelihood + b_likelihood)

                if probability_point_in_a > probability_point_in_b:
                    probabilities_a.append(probability_point_in_a)
                    probabilities_b.append(np.nan)

                elif probability_point_in_a < probability_point_in_b:
                    probabilities_b.append(probability_point_in_b)
                    probabilities_a.append(np.nan)

                else:
                    probabilities_a.append(0)
                    probabilities_b.append(0)

            cmap = plt.get_cmap("bwr")
            cmap_a = truncate_colormap(cmap, 0.5, 0)
            cmap_b = truncate_colormap(cmap, 0.5, 1)

            line_x_a = colored_line(
                mag_data["date"],
                mag_data["mag_x"],
                probabilities_a,
                ax=mag_axes[0],
                cmap=cmap_a,
            )
            line_x_b = colored_line(
                mag_data["date"],
                mag_data["mag_x"],
                probabilities_b,
                ax=mag_axes[0],
                cmap=cmap_b,
            )
            line_total_a = colored_line(
                mag_data["date"],
                mag_data["mag_total"],
                probabilities_a,
                ax=mag_axes[1],
                cmap=cmap_a,
            )
            line_total_b = colored_line(
                mag_data["date"],
                mag_data["mag_total"],
                probabilities_b,
                ax=mag_axes[1],
                cmap=cmap_b,
            )

            # Add an Axes to the right of the main Axes.
            ax1_divider = make_axes_locatable(mag_axes[0])
            cax1a = ax1_divider.append_axes("right", size="2%", pad="1%")
            cax1b = ax1_divider.append_axes("right", size="2%", pad="1%")

            a1_colorbar = fig.colorbar(line_x_a, cax=cax1a)
            b1_colorbar = fig.colorbar(line_x_b, cax=cax1b)
            a1_colorbar.set_ticks([])
            b1_colorbar.set_label("Probability")

            ax2_divider = make_axes_locatable(mag_axes[1])
            cax2a = ax2_divider.append_axes("right", size="2%", pad="1%")
            cax2b = ax2_divider.append_axes("right", size="2%", pad="1%")

            a2_colorbar = fig.colorbar(line_total_a, cax=cax2a)
            b2_colorbar = fig.colorbar(line_total_b, cax=cax2b)
            a2_colorbar.set_ticks([])
            b2_colorbar.set_label("Probability")

        case "midpoint":
            # Find the midpoint between the two peaks
            midpoint = (fit.pars[1] + fit.pars[4]) / 2

            region_index = []

            for field_value in distribution_values:

                if field_value > midpoint:
                    current_region = 1
                else:
                    current_region = 0

                region_index.append(current_region)

            colored_line(
                mag_data["date"],
                mag_data["mag_x"],
                region_index,
                ax=mag_axes[0],
                cmap="bwr",
            )
            colored_line(
                mag_data["date"],
                mag_data["mag_total"],
                region_index,
                ax=mag_axes[1],
                cmap="bwr",
            )

            histogram_axis.axvline(midpoint, color="orange", label="Midpoint")

        case "minimum_point":
            # Split based off of the minimum point of the gaussian distribution
            distribution_minimum_index, _ = scipy.signal.find_peaks(
                -fit.output_y_values
            )

            distribution_minimum = fit.x_range[distribution_minimum_index]

            region_index = []

            for field_value in distribution_values:

                if field_value > distribution_minimum:
                    current_region = 1
                else:
                    current_region = 0

                region_index.append(current_region)

            colored_line(
                mag_data["date"],
                mag_data["mag_x"],
                region_index,
                ax=mag_axes[0],
                cmap="bwr",
            )
            colored_line(
                mag_data["date"],
                mag_data["mag_total"],
                region_index,
                ax=mag_axes[1],
                cmap="bwr",
            )

            histogram_axis.axvline(
                distribution_minimum, color="orange", label="Distribution Minimum"
            )

        case "threshold":
            threshold_sigmas = 1
            # Upper threshold:
            #       If the data is low, and we go higher than this then we switch.
            #       Upper threshold is the mean - 1 sigma of the upper gaussian
            upper_threshold = fit.pars[4] - threshold_sigmas * fit.pars[5]

            # Lower threshold:
            #       If the data is high, and we go lower than this then we switch.
            #       Lower threshold is the mean + 1 sigma of the lower gaussian
            lower_threshold = fit.pars[1] + threshold_sigmas * fit.pars[2]

            region_index = []
            # Low region = 0, high region = 1
            # We determine the initial region from a comparison of the first data point
            if distribution_values[0] > upper_threshold:
                current_region = 1
            elif distribution_values[0] < lower_threshold:
                current_region = 0
            # We need to handle cases between
            else:
                upper_difference = upper_threshold - distribution_values[0]
                lower_difference = distribution_values[0] - lower_threshold

                if upper_difference >= lower_difference:
                    current_region = 1
                else:
                    current_region = 0

            for field_value in distribution_values:

                if field_value > upper_threshold:
                    current_region = 1
                elif field_value < lower_threshold:
                    current_region = 0

                region_index.append(current_region)

            colored_line(
                mag_data["date"],
                mag_data["mag_x"],
                region_index,
                ax=mag_axes[0],
                cmap="bwr",
            )
            colored_line(
                mag_data["date"],
                mag_data["mag_total"],
                region_index,
                ax=mag_axes[1],
                cmap="bwr",
            )

            histogram_axis.axvline(
                upper_threshold, color="red", label="Upper Threshold"
            )
            histogram_axis.axvline(
                lower_threshold, color="blue", label="Lower Threshold"
            )


def Double_Gaussian(x, c1, mu1, sigma1, c2, mu2, sigma2):
    res = c1 * np.exp(-((x - mu1) ** 2.0) / (2.0 * sigma1**2.0)) + c2 * np.exp(
        -((x - mu2) ** 2.0) / (2.0 * sigma2**2.0)
    )
    return res


# We define a single gaussian too to overplot
def Single_Gaussian(x, c, mu, sigma):
    res = c * np.exp(-((x - mu) ** 2) / (2 * sigma**2))
    return res
