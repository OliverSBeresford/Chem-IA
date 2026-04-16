#!/usr/bin/env python3

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_data(csv_path: Path):
	rows = []
	with csv_path.open(newline="", encoding="utf-8") as f:
		reader = csv.DictReader(f)
		for row in reader:
			rows.append(row)

	temperatures_c = np.array([float(r["Average Temp"]) + 273.15 for r in rows], dtype=float)
	temperature_uncertainties = np.array([float(r["Temp Uncertainty"]) for r in rows], dtype=float)

	rate_values = np.array([float(r["Average Rate (temp)"]) for r in rows], dtype=float)
	rate_uncertainties = np.array([float(r["Rate Uncertainty (Temp)"]) for r in rows], dtype=float)

	return temperatures_c, temperature_uncertainties, rate_values, rate_uncertainties


def fit_trendline(x_values: np.ndarray, y_values: np.ndarray):
	# Unweighted linear regression.
	coeffs = np.polyfit(x_values, y_values, deg=1)
	return coeffs


def r2_score(
	x_values: np.ndarray,
	y_values: np.ndarray,
	slope: float,
	intercept: float,
) -> float:
	y_predicted = slope * x_values + intercept
	y_mean = np.mean(y_values)
	ss_res = np.sum((y_values - y_predicted) ** 2)
	ss_tot = np.sum((y_values - y_mean) ** 2)
	if ss_tot <= 0:
		return np.nan
	return 1.0 - (ss_res / ss_tot)


def line_from_two_points(x1: float, y1: float, x2: float, y2: float):
	if x2 == x1:
		raise ValueError("Cannot compute slope: x-values for the two points are identical.")
	slope = (y2 - y1) / (x2 - x1)
	intercept = y1 - slope * x1
	return slope, intercept


def fit_best_min_max(
	x_values: np.ndarray,
	x_uncertainties: np.ndarray,
	y_values: np.ndarray,
	y_uncertainties: np.ndarray,
):
	best_slope, best_intercept = fit_trendline(x_values, y_values)

	x_first = x_values[0]
	y_first = y_values[0]
	dx_first = x_uncertainties[0]
	dy_first = y_uncertainties[0]

	x_last = x_values[-1]
	y_last = y_values[-1]
	dx_last = x_uncertainties[-1]
	dy_last = y_uncertainties[-1]

	# Corner-to-corner fits from endpoint uncertainty boxes.
	slope_a, intercept_a = line_from_two_points(
		x_first - dx_first,
		y_first - dy_first,
		x_last + dx_last,
		y_last + dy_last,
	)
	slope_b, intercept_b = line_from_two_points(
		x_first + dx_first,
		y_first + dy_first,
		x_last - dx_last,
		y_last - dy_last,
	)

	if slope_a <= slope_b:
		lower_slope, lower_intercept = slope_a, intercept_a
		upper_slope, upper_intercept = slope_b, intercept_b
	else:
		lower_slope, lower_intercept = slope_b, intercept_b
		upper_slope, upper_intercept = slope_a, intercept_a

	return (
		(best_slope, best_intercept),
		(lower_slope, lower_intercept),
		(upper_slope, upper_intercept),
	)


def main():
	base_dir = Path(__file__).resolve().parent
	csv_path = base_dir / "data.csv"
    
	temperatures_c, temperature_uncertainties, rate_values, rate_uncertainties = load_data(csv_path)

	(rate_best_fit, rate_min_fit, rate_max_fit) = fit_best_min_max(
		temperatures_c,
		temperature_uncertainties,
		rate_values,
		rate_uncertainties,
	)
	rate_best_slope, rate_best_intercept = rate_best_fit
	rate_min_slope, rate_min_intercept = rate_min_fit
	rate_max_slope, rate_max_intercept = rate_max_fit

	rate_r2 = r2_score(
		temperatures_c,
		rate_values,
		rate_best_slope,
		rate_best_intercept,
	)

	# Uncertainty propagation for ln(y): sigma_ln(y) = sigma_y / y
	ln_rate_values = np.log(rate_values)
	ln_rate_uncertainties = rate_uncertainties / rate_values

	(ln_rate_best_fit, ln_rate_min_fit, ln_rate_max_fit) = fit_best_min_max(
		temperatures_c,
		temperature_uncertainties,
		ln_rate_values,
		ln_rate_uncertainties,
	)
	ln_rate_best_slope, ln_rate_best_intercept = ln_rate_best_fit
	ln_rate_min_slope, ln_rate_min_intercept = ln_rate_min_fit
	ln_rate_max_slope, ln_rate_max_intercept = ln_rate_max_fit

	ln_rate_r2 = r2_score(
		temperatures_c,
		ln_rate_values,
		ln_rate_best_slope,
		ln_rate_best_intercept,
	)

	temperature_line = np.linspace(temperatures_c.min() - 1, temperatures_c.max() + 1, 200)
	rate_best_line = rate_best_slope * temperature_line + rate_best_intercept
	rate_min_line = rate_min_slope * temperature_line + rate_min_intercept
	rate_max_line = rate_max_slope * temperature_line + rate_max_intercept
	ln_rate_best_line = ln_rate_best_slope * temperature_line + ln_rate_best_intercept
	ln_rate_min_line = ln_rate_min_slope * temperature_line + ln_rate_min_intercept
	ln_rate_max_line = ln_rate_max_slope * temperature_line + ln_rate_max_intercept

	plt.figure(figsize=(9, 6))
	plt.errorbar(
		temperatures_c,
		rate_values,
		xerr=temperature_uncertainties,
		yerr=rate_uncertainties,
		fmt="s",
		capsize=3,
		color="#d62728",
		label="Hydrogen production rate",
	)
	plt.plot(
		temperature_line,
		rate_best_line,
		"--",
		color="#1f77b4",
		alpha=0.9,
		label=f"Best fit: gradient = {rate_best_slope:.6e}, y-intercept = {rate_best_intercept:.6e}, R^2 = {rate_r2:.4f}",
	)
	plt.plot(
		temperature_line,
		rate_min_line,
		":",
		color="#2ca02c",
		alpha=0.9,
		label=f"Minimum fit: gradient = {rate_min_slope:.6e}, y-intercept = {rate_min_intercept:.6e}",
	)
	plt.plot(
		temperature_line,
		rate_max_line,
		":",
		color="#ff7f0e",
		alpha=0.9,
		label=f"Maximum fit: gradient = {rate_max_slope:.6e}, y-intercept = {rate_max_intercept:.6e}",
	)

	plt.title("Hydrogen Production Rate vs Temperature")
	plt.xlabel("Temperature (°C)")
	plt.ylabel("Rate (mol/s)")
	plt.grid(alpha=0.25)
	plt.legend()
	plt.tight_layout()

	output_path = base_dir / "graph.png"
	plt.savefig(output_path, dpi=200)
	plt.show()

	plt.figure(figsize=(9, 6))
	plt.errorbar(
		temperatures_c,
		ln_rate_values,
		xerr=temperature_uncertainties,
		yerr=ln_rate_uncertainties,
		fmt="s",
		capsize=3,
		color="#d62728",
		label="ln(Hydrogen production rate)",
	)

	plt.plot(
		temperature_line,
		ln_rate_best_line,
		"--",
		color="#1f77b4",
		alpha=0.9,
		label=f"Best fit in ln-space: gradient = {ln_rate_best_slope:.6f}, y-intercept = {ln_rate_best_intercept:.6f}, R^2 = {ln_rate_r2:.4f}",
	)
	plt.plot(
		temperature_line,
		ln_rate_min_line,
		":",
		color="#2ca02c",
		alpha=0.9,
		label=f"Minimum fit in ln-space: gradient = {ln_rate_min_slope:.6f}, y-intercept = {ln_rate_min_intercept:.6f}",
	)
	plt.plot(
		temperature_line,
		ln_rate_max_line,
		":",
		color="#ff7f0e",
		alpha=0.9,
		label=f"Maximum fit in ln-space: gradient = {ln_rate_max_slope:.6f}, y-intercept = {ln_rate_max_intercept:.6f}",
	)

	plt.title("ln(Hydrogen Production Rate) vs Temperature with Uncertainty")
	plt.xlabel("Temperature (°C)")
	plt.ylabel("ln(Hydrogen Production Rate)")
	plt.grid(alpha=0.25)
	plt.legend()
	plt.tight_layout()

	output_path_ln = base_dir / "graph_ln.png"
	plt.savefig(output_path_ln, dpi=200)
	plt.show()

	print(f"Average Rate (temp) fit R^2: {rate_r2:.6f}")
	print(f"ln(Average Rate (temp)) fit R^2: {ln_rate_r2:.6f}")


if __name__ == "__main__":
	main()
