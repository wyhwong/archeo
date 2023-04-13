import numpy as np

from .logger import get_logger

LOGGER = get_logger(logger_name="Utils | Statistical Operation")


def get_bins_midpts(boundaries: list):
    return [(boundaries[bin_index] + boundaries[bin_index + 1]) / 2 for bin_index in range(len(boundaries) - 1)]


# This function output the statistical properties of a posterior
def compute_posterior_statistics(posterior: list, weights=None, nbins=200) -> tuple:
    # Convert samples to discrete density function
    density, bins = np.histogram(a=posterior, bins=nbins, weights=weights, density=True)
    binwidth = bins[1] - bins[0]
    cumulative_density = 0.0
    error_lower_bound, error_upper_bound, median = None, None, None
    for bin_index, bin_density in enumerate(density):
        cumulative_density += bin_density * binwidth
        if cumulative_density >= 0.1 and error_lower_bound is None:
            error_lower_bound = bins[0] + (bin_index + 0.5) * binwidth
        if cumulative_density >= 0.5 and median is None:
            median = bins[0] + (bin_index + 0.5) * binwidth
            error_lower_bound = median - error_lower_bound
        if cumulative_density >= 0.9 and error_upper_bound is None:
            error_upper_bound = bins[0] + (bin_index + 0.5) * binwidth - median
    return (density, bins, error_lower_bound, error_upper_bound, median)
