"""
These functions support residual analysis by providing a
Hurst-exponent estimate and a human-readable noise classification.
"""

from __future__ import annotations

import math

import numpy as np

# delete? already in learning_models.py as the residuals function
def compute_residuals(y_true, y_pred):
	"""Return residuals as ``y_true - y_pred``.

	This mirrors the residual convention used in :mod:`learning_models`.
	"""

	return np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)


def _window_fluctuation(profile, scale):
	"""Compute the root-mean-square fluctuation of a profile for a given window size (scale).
	It chops the profile into 
    non-overlapping segments, fits a linear trend to each, and measures 
    the variance of the residuals around that trend.
	"""

	window_count = len(profile) // scale
	if window_count == 0:
		return np.nan

	fluctuations = []
	for index in range(window_count):
		# Get the current window
		segment = profile[index * scale : (index + 1) * scale]
		if len(segment) < 2:
			continue
		# Fit a 1st-degree polynomial to the window
		x_values = np.arange(len(segment), dtype=float)
		coefficients = np.polyfit(x_values, segment, deg=1)
		trend = np.polyval(coefficients, x_values)

		# Detrend the segment by subtracting the linear fit
		residuals = segment - trend
		fluctuations.append(float(np.mean(residuals**2)))

	if not fluctuations:
		return np.nan

	# Return the root mean square of all window fluctuations
	return float(math.sqrt(np.mean(fluctuations)))


def calculate_hurst_exponent(signal, min_scale=4, n_scales=20):
	"""Estimate DFA scaling properties and the Hurst exponent.
	
	Parameters
	----------
	signal : array-like
		Residual series or any one-dimensional signal.
	min_scale : int, default 4
		Smallest DFA window size to consider.
	n_scales : int, default 20
		Number of candidate scales sampled on a log scale.
	"""

	signal = np.asarray(signal, dtype=float).ravel()
	signal = signal[np.isfinite(signal)]
	if signal.size < min_scale * 2:
		raise ValueError("signal must contain enough finite values for DFA")

	# Mean center the signal
	signal = signal - np.mean(signal)

	# Build the cumulative profile (converts noise into a random walk)
	profile = np.cumsum(signal)

	# Max scale is set to N/4 to ensure at least 4 windows to average over
	max_scale = signal.size // 4
	if max_scale < min_scale:
		raise ValueError("signal is too short to estimate a Hurst exponent")

	# Choose window sizes(scales) evenly spaced on a logarithmic grid
	scales = np.unique(
		np.logspace(np.log10(min_scale), np.log10(max_scale), n_scales).astype(int)
	)
	scales = scales[scales >= min_scale]

	fluctuations = []
	valid_scales = []
	# Calculate the fluctuation for each window size
	for scale in scales:
		fluctuation = _window_fluctuation(profile, int(scale))
		if np.isfinite(fluctuation) and fluctuation > 0:
			valid_scales.append(int(scale))
			fluctuations.append(float(fluctuation))

	if len(valid_scales) < 2:
		raise ValueError("not enough valid DFA scales to estimate a Hurst exponent")

	valid_scales = np.asarray(valid_scales, dtype=float)
	fluctuations = np.asarray(fluctuations, dtype=float)

	# Fit a line in log-log space to find the scaling behavior
	# The slope of this line is the empirical Hurst exponent
	log_scales = np.log(valid_scales)
	log_fluctuations = np.log(fluctuations)
	slope, intercept = np.polyfit(log_scales, log_fluctuations, deg=1)

	# Calculate R-squared to verify if the log-log relationship is indeed linear
	predicted = intercept + slope * log_scales
	ss_res = np.sum((log_fluctuations - predicted) ** 2)
	ss_tot = np.sum((log_fluctuations - np.mean(log_fluctuations)) ** 2)
	if ss_tot == 0:
		dfa_r2 = float("nan")
	else:
		dfa_r2 = float(1 - ss_res / ss_tot)

	return {
		"hurst": float(slope),
		"intercept": float(intercept),
		"dfa_r2": dfa_r2,
		"scales": valid_scales,
		"fluctuations": fluctuations,
	}


def classify_noise(h):
	"""Classify a Hurst exponent into a simple noise regime.
	< 0.45 : Anti-persistent (alternating high/low values)
    0.45 to 0.55 : Uncorrelated (White noise, random)
    > 0.55 : Persistent (Positive correlations, 'memory' or 'flow')
	"""

	if h is None or not np.isfinite(h):
		return "unknown"
	if h < 0.45:
		return "anti-persistent"
	if h <= 0.55:
		return "uncorrelated"
	return "persistent"


def dfa_summary(signal):
	"""Return a compact DFA summary for report writing."""

	try:
		result = calculate_hurst_exponent(signal)
	except ValueError:
		return {
			"hurst": float("nan"),
			"dfa_r2": float("nan"),
			"noise_type": "unknown",
		}
	return {
		"hurst": result["hurst"],
		"dfa_r2": result["dfa_r2"],
		"noise_type": classify_noise(result["hurst"]),
	}


__all__ = [
	"compute_residuals",
	"calculate_hurst_exponent",
	"classify_noise",
	"dfa_summary",
]
