"""Reusable learning-curve model functions.

This module exposes the core parametric functions and 
a few small helpers used during model fitting.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import curve_fit


def power_law(x, a, b, c):
	"""
    Return the power-law curve: a * x**(-b) + c.

    This function models skill acquisition where improvement is rapid initially 
    but slows down over time, approaching a physical limit (c).

    Parameters
    ----------
    x : array-like or float
        Time or trial index (e.g., sequential attempt number).
    a : float
        Scale parameter representing the initial performance level.
    b : float
        Decay parameter representing the learning rate.
    c : float
        Asymptote / floor representing the theoretical best possible performance.
    """

	x = np.asarray(x, dtype=float)
	# Model includes asymptote c to represent a lower performance limit
	return a * np.power(x, -b) + c


def exponential(x, a, b, c):
	"""
    Return the exponential learning curve: a * exp(-b * x) + c.
    
    This function models skill acquisition as a constant percentage 
    reduction in error/time per trial.
    """

	x = np.asarray(x, dtype=float)
	# Parameters are constrained to positive values during fitting.
	return a * np.exp(-b * x) + c


def residuals(y_true, y_pred):
	"""
    Calculate the difference between observed and predicted values.
    Returns: y_true - y_pred
    """

	return np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)


def rss(y_true, y_pred):
	"""
    Compute the Residual Sum of Squares (RSS).
    """

	res = residuals(y_true, y_pred)
	return float(np.sum(res**2))


def r_squared(y_true, y_pred):
	"""
    Compute the coefficient of determination (R²).
    """

	y_true = np.asarray(y_true, dtype=float)
	y_pred = np.asarray(y_pred, dtype=float)
	# Residual sum of squares
	ss_res = np.sum((y_true - y_pred) ** 2)
	# Total sum of squares
	ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
	if ss_tot == 0:
		return float("nan")
	return float(1 - ss_res / ss_tot)


def calculate_aic(n_obs, rss_value, n_params):
	"""
    Compute Akaike Information Criterion (AIC) assuming Gaussian residuals.
    """

	n_obs = int(n_obs)
	rss_value = float(rss_value)
	if n_obs <= 0:
		raise ValueError("n_obs must be positive")
	# Prevent math.log(0) by assigning the smallest possible float
	if rss_value <= 0:
		rss_value = np.finfo(float).tiny
	return float(n_obs * math.log(rss_value / n_obs) + 2 * n_params)


def calculate_bic(n_obs, rss_value, n_params):
	"""
    Compute Bayesian Information Criterion (BIC) assuming Gaussian residuals.
    """

	n_obs = int(n_obs)
	rss_value = float(rss_value)
	if n_obs <= 0:
		raise ValueError("n_obs must be positive")
	if rss_value <= 0:
		rss_value = np.finfo(float).tiny
	return float(n_obs * math.log(rss_value / n_obs) + n_params * math.log(n_obs))


def fit_power_law(x, y):
	"""
    Fit the asymptote-aware power-law model to the provided data.
    
    Uses non-linear least squares to fit the function to the data. 
    Bounds constrain the parameters to positive values, which is theoretically 
    sound since time and decay rate cannot be negative.
    """

	x = np.asarray(x, dtype=float)
	y = np.asarray(y, dtype=float)
	# Initial guesses: max time (a), slow decay (b), min time (c)
	initial_guess = [float(np.nanmax(y)), 0.1, float(np.nanmin(y))]
	try:
		params, covariance = curve_fit(
			power_law,
			x,
			y,
			p0=initial_guess,
			bounds=(0, np.inf),
			maxfev=10000,
		)
	except (RuntimeError, ValueError):
		# Handle instances where the model fails to converge
		return None, None
	return params, covariance


def fit_exponential(x, y):
	"""
    Fit the exponential model to the provided data using non-linear least squares.
    """

	x = np.asarray(x, dtype=float)
	y = np.asarray(y, dtype=float)
	initial_guess = [float(np.nanmax(y)), 0.01, float(np.nanmin(y))]
	try:
		params, covariance = curve_fit(
			exponential,
			x,
			y,
			p0=initial_guess,
			bounds=(0, np.inf),
			maxfev=10000,
		)
	except (RuntimeError, ValueError):
		return None, None
	return params, covariance


def evaluate_model(y_true, y_pred, n_params):
    """
    Aggregates statistical fit metrics into a single dictionary for easy reporting.
    """

    if y_pred is None:
        return {
            "rss": np.nan,
            "r2": np.nan,
            "aic": np.nan,
            "bic": np.nan,
        }

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    n_obs = len(y_true)
    rss_value = rss(y_true, y_pred)

    return {
        "rss": rss_value,
        "r2": r_squared(y_true, y_pred),
        "aic": calculate_aic(n_obs, rss_value, n_params),
        "bic": calculate_bic(n_obs, rss_value, n_params),
    }


def aic(n_obs, rss_value, n_params):
	"""Backward-compatible alias for calculate_aic."""

	return calculate_aic(n_obs, rss_value, n_params)


def bic(n_obs, rss_value, n_params):
	"""Backward-compatible alias for calculate_bic."""

	return calculate_bic(n_obs, rss_value, n_params)


__all__ = [
	"power_law",
	"exponential",
	"residuals",
	"rss",
	"r_squared",
	"calculate_aic",
	"calculate_bic",
	"fit_power_law",
	"fit_exponential",
	"evaluate_model",
	"aic",
	"bic",
]
