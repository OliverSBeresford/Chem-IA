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

	x = np.array([float(r["Average Temp"]) for r in rows], dtype=float)
	xerr = np.array([float(r["Temp Uncertainty"]) for r in rows], dtype=float)

	y1 = np.array([float(r["Average Rate"]) for r in rows], dtype=float)
	y1err = np.array([float(r["Rate Uncertainty"]) for r in rows], dtype=float)

	y2 = np.array([float(r["Average Rate (temp)"]) for r in rows], dtype=float)
	y2err = np.array([float(r["Rate Uncertainty (Temp)"]) for r in rows], dtype=float)

	return x, xerr, y1, y1err, y2, y2err


def fit_trendline(x: np.ndarray, y: np.ndarray, yerr: np.ndarray):
	# Weighted linear fit so points with smaller y-uncertainty have more influence.
	safe_yerr = np.where(yerr <= 0, np.min(yerr[yerr > 0]) if np.any(yerr > 0) else 1.0, yerr)
	coeffs = np.polyfit(x, y, deg=1, w=1.0 / safe_yerr)
	return coeffs


def weighted_r2(x: np.ndarray, y: np.ndarray, yerr: np.ndarray, m: float, b: float) -> float:
	safe_yerr = np.where(yerr <= 0, np.min(yerr[yerr > 0]) if np.any(yerr > 0) else 1.0, yerr)
	w = 1.0 / (safe_yerr ** 2)
	y_pred = m * x + b
	y_bar = np.average(y, weights=w)
	ss_res = np.sum(w * (y - y_pred) ** 2)
	ss_tot = np.sum(w * (y - y_bar) ** 2)
	if ss_tot <= 0:
		return np.nan
	return 1.0 - (ss_res / ss_tot)


def main():
	base_dir = Path(__file__).resolve().parent
	csv_path = base_dir / "data.csv"

	x, xerr, y1, y1err, y2, y2err = load_data(csv_path)

	m1, b1 = fit_trendline(x, y1, y1err)
	m2, b2 = fit_trendline(x, y2, y2err)
	r2_1 = weighted_r2(x, y1, y1err, m1, b1)
	r2_2 = weighted_r2(x, y2, y2err, m2, b2)

	# Uncertainty propagation for ln(y): sigma_ln(y) = sigma_y / y
	y1_ln = np.log(y1)
	y2_ln = np.log(y2)
	y1_ln_err = y1err / y1
	y2_ln_err = y2err / y2

	m1_ln, b1_ln = fit_trendline(x, y1_ln, y1_ln_err)
	m2_ln, b2_ln = fit_trendline(x, y2_ln, y2_ln_err)
	r2_1_ln = weighted_r2(x, y1_ln, y1_ln_err, m1_ln, b1_ln)
	r2_2_ln = weighted_r2(x, y2_ln, y2_ln_err, m2_ln, b2_ln)

	x_line = np.linspace(x.min() - 1, x.max() + 1, 200)
	y1_line = m1 * x_line + b1
	y2_line = m2 * x_line + b2
	y1_ln_line = m1_ln * x_line + b1_ln
	y2_ln_line = m2_ln * x_line + b2_ln

	plt.figure(figsize=(9, 6))
	plt.errorbar(
		x,
		y1,
		xerr=xerr,
		yerr=y1err,
		fmt="o",
		capsize=3,
		color="#1f77b4",
		label="Average Rate",
	)
	plt.errorbar(
		x,
		y2,
		xerr=xerr,
		yerr=y2err,
		fmt="s",
		capsize=3,
		color="#d62728",
		label="Average Rate (temp)",
	)

	plt.plot(
		x_line,
		y1_line,
		"--",
		color="#1f77b4",
		alpha=0.9,
		label=f"Trendline: Average Rate (R^2 = {r2_1:.4f})",
	)
	plt.plot(
		x_line,
		y2_line,
		"--",
		color="#d62728",
		alpha=0.9,
		label=f"Trendline: Average Rate (temp) (R^2 = {r2_2:.4f})",
	)

	plt.title("Rate vs Temperature with Uncertainty")
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
		x,
		y1_ln,
		xerr=xerr,
		yerr=y1_ln_err,
		fmt="o",
		capsize=3,
		color="#1f77b4",
		label="ln(Average Rate)",
	)
	plt.errorbar(
		x,
		y2_ln,
		xerr=xerr,
		yerr=y2_ln_err,
		fmt="s",
		capsize=3,
		color="#d62728",
		label="ln(Average Rate (temp))",
	)

	plt.plot(
		x_line,
		y1_ln_line,
		"--",
		color="#1f77b4",
		alpha=0.9,
		label=f"Linear fit in ln-space (Average Rate), R^2 = {r2_1_ln:.4f}",
	)
	plt.plot(
		x_line,
		y2_ln_line,
		"--",
		color="#d62728",
		alpha=0.9,
		label=f"Linear fit in ln-space (Average Rate (temp)), R^2 = {r2_2_ln:.4f}",
	)

	plt.title("ln(Rate) vs Temperature with Uncertainty")
	plt.xlabel("Temperature (°C)")
	plt.ylabel("ln(Rate)")
	plt.grid(alpha=0.25)
	plt.legend()
	plt.tight_layout()

	output_path_ln = base_dir / "graph_ln.png"
	plt.savefig(output_path_ln, dpi=200)
	plt.show()

	print(f"Average Rate fit R^2: {r2_1:.6f}")
	print(f"Average Rate (temp) fit R^2: {r2_2:.6f}")
	print(f"ln(Average Rate) fit R^2: {r2_1_ln:.6f}")
	print(f"ln(Average Rate (temp)) fit R^2: {r2_2_ln:.6f}")


if __name__ == "__main__":
	main()
