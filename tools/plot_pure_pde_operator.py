#!/usr/bin/env python3
"""Plot pure-PDE-vs-constant metrics from ``evaluate_pure_pde_operator.py``.

Input CSV columns:
    variable, lead, pde_wrmse, constant_wrmse, pde_improvement_pct

The script writes two PNG files:
    pure_pde_vs_constant_wrmse.png
    pure_pde_improvement_pct.png
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path


VARIABLE_ORDER = ("z", "t", "q", "u", "v", "physics_all")


def prepare_matplotlib() -> None:
    if "MPLCONFIGDIR" not in os.environ:
        config_dir = Path(os.environ.get("TMPDIR", "/tmp")) / "weatherpred_matplotlib"
        config_dir.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(config_dir)
    import matplotlib

    matplotlib.use("Agg", force=True)


def read_rows(path: Path) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        required = {"variable", "lead", "pde_wrmse", "constant_wrmse", "pde_improvement_pct"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"{path} is missing columns: {', '.join(sorted(missing))}")
        for row in reader:
            lead_raw = str(row["lead"])
            lead: int | str = "mean" if lead_raw == "mean" else int(lead_raw)
            rows.append(
                {
                    "variable": str(row["variable"]),
                    "lead": lead,
                    "pde_wrmse": float(row["pde_wrmse"]),
                    "constant_wrmse": float(row["constant_wrmse"]),
                    "pde_improvement_pct": float(row["pde_improvement_pct"]),
                }
            )
    if not rows:
        raise SystemExit(f"No rows in {path}")
    return rows


def sorted_variables(rows: list[dict[str, float | int | str]]) -> list[str]:
    present = {str(row["variable"]) for row in rows}
    ordered = [name for name in VARIABLE_ORDER if name in present]
    ordered.extend(sorted(present.difference(ordered)))
    return ordered


def plot_wrmse_grid(rows: list[dict[str, float | int | str]], out_path: Path) -> None:
    prepare_matplotlib()
    import matplotlib.pyplot as plt

    variables = sorted_variables(rows)
    cols = 3
    rows_n = (len(variables) + cols - 1) // cols
    fig, axes = plt.subplots(rows_n, cols, figsize=(5.0 * cols, 3.5 * rows_n), squeeze=False)
    axes_flat = list(axes.ravel())

    for ax, variable in zip(axes_flat, variables, strict=False):
        data = [
            row
            for row in rows
            if row["variable"] == variable and row["lead"] != "mean"
        ]
        data.sort(key=lambda row: int(row["lead"]))
        leads = [int(row["lead"]) for row in data]
        pde = [float(row["pde_wrmse"]) for row in data]
        constant = [float(row["constant_wrmse"]) for row in data]
        ax.plot(leads, pde, marker="o", linewidth=2, label="Pure PDE")
        ax.plot(leads, constant, marker="o", linewidth=2, label="Constant")
        ax.set_title(variable)
        ax.set_xlabel("Lead hour")
        ax.set_ylabel("WRMSE")
        ax.grid(True, alpha=0.25)
        ax.legend()

    for ax in axes_flat[len(variables) :]:
        ax.axis("off")

    fig.suptitle("Pure PDE vs Constant Prediction", fontsize=14)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_improvement(rows: list[dict[str, float | int | str]], out_path: Path) -> None:
    prepare_matplotlib()
    import matplotlib.pyplot as plt

    variables = sorted_variables(rows)
    mean_by_var = {
        str(row["variable"]): float(row["pde_improvement_pct"])
        for row in rows
        if row["lead"] == "mean"
    }
    values = [mean_by_var[name] for name in variables if name in mean_by_var]
    labels = [name for name in variables if name in mean_by_var]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = ["#2e7d32" if value >= 0 else "#b3261e" for value in values]
    ax.bar(labels, values, color=colors)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_ylabel("Improvement over constant, %")
    ax.set_title("Mean Pure PDE Improvement")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, help="CSV from evaluate_pure_pde_operator.py.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to plots/pure_pde_operator/<csv-stem>.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path("plots/pure_pde_operator") / csv_path.stem
    )
    rows = read_rows(csv_path)
    wrmse_path = output_dir / "pure_pde_vs_constant_wrmse.png"
    improvement_path = output_dir / "pure_pde_improvement_pct.png"
    plot_wrmse_grid(rows, wrmse_path)
    plot_improvement(rows, improvement_path)
    print(wrmse_path)
    print(improvement_path)


if __name__ == "__main__":
    main()
