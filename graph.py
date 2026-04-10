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


def main():
	base_dir = Path(__file__).resolve().parent
	csv_path = base_dir / "data.csv"

	x, xerr, y1, y1err, y2, y2err = load_data(csv_path)

	m1, b1 = fit_trendline(x, y1, y1err)
	m2, b2 = fit_trendline(x, y2, y2err)

	x_line = np.linspace(x.min() - 1, x.max() + 1, 200)
	y1_line = m1 * x_line + b1
	y2_line = m2 * x_line + b2

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

	plt.plot(x_line, y1_line, "--", color="#1f77b4", alpha=0.9, label="Trendline: Average Rate")
	plt.plot(x_line, y2_line, "--", color="#d62728", alpha=0.9, label="Trendline: Average Rate (temp)")

	plt.title("Rate vs Temperature with Uncertainty")
	plt.xlabel("Temperature (°C)")
	plt.ylabel("Rate (mol/s)")
	plt.grid(alpha=0.25)
	plt.legend()
	plt.tight_layout()

	output_path = base_dir / "graph.png"
	plt.savefig(output_path, dpi=200)
	plt.show()


if __name__ == "__main__":
	main()
