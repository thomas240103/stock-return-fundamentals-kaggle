from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stock_returns.config import ID_COL, SECTOR_MAPPING, TARGET_COL


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create EDA tables and plots for the Kaggle fundamentals data.")
    parser.add_argument("--train", default="data/raw/train.csv", help="Path to Kaggle train.csv.")
    parser.add_argument("--test", default="data/raw/test.csv", help="Optional path to Kaggle test.csv.")
    parser.add_argument("--output-dir", default="outputs/eda", help="Directory for EDA CSVs, PNGs, and report.")
    parser.add_argument("--top-n", type=int, default=30, help="Number of top rows/features to show in selected plots.")
    parser.add_argument("--sample-rows", type=int, default=5000, help="Max rows for scatter plots.")
    return parser.parse_args()


def require_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except ImportError as exc:
        raise ImportError("matplotlib is required for EDA plots. Install with: pip install matplotlib") from exc


def ensure_output_dir(path: str | Path) -> Path:
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def read_csv(path: str | Path, required: bool = True) -> pd.DataFrame | None:
    csv_path = Path(path)
    if not csv_path.exists():
        if required:
            raise FileNotFoundError(f"File not found: {csv_path}")
        return None
    return pd.read_csv(csv_path)


def save_table(df: pd.DataFrame, output_dir: Path, filename: str) -> Path:
    path = output_dir / filename
    df.to_csv(path, index=False)
    return path


def save_plot(fig, output_dir: Path, filename: str) -> Path:
    path = output_dir / filename
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    return path


def numeric_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=[np.number]).columns.tolist()


def dataset_overview(train: pd.DataFrame, test: pd.DataFrame | None) -> pd.DataFrame:
    rows = []
    for name, df in [("train", train), ("test", test)]:
        if df is None:
            continue
        target_present = TARGET_COL in df.columns
        rows.append(
            {
                "dataset": name,
                "rows": len(df),
                "columns": df.shape[1],
                "numeric_columns": len(numeric_columns(df)),
                "duplicate_ids": int(df[ID_COL].duplicated().sum()) if ID_COL in df.columns else np.nan,
                "target_present": target_present,
                "target_mean": float(df[TARGET_COL].mean()) if target_present else np.nan,
                "target_median": float(df[TARGET_COL].median()) if target_present else np.nan,
                "target_std": float(df[TARGET_COL].std()) if target_present else np.nan,
                "target_min": float(df[TARGET_COL].min()) if target_present else np.nan,
                "target_max": float(df[TARGET_COL].max()) if target_present else np.nan,
            }
        )
    return pd.DataFrame(rows)


def missingness_table(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    missing_count = df.isna().sum()
    table = pd.DataFrame(
        {
            "dataset": dataset_name,
            "column": df.columns,
            "dtype": [str(df[col].dtype) for col in df.columns],
            "missing_count": missing_count.to_numpy(),
            "missing_fraction": (missing_count / len(df)).to_numpy(),
            "n_unique": [df[col].nunique(dropna=True) for col in df.columns],
        }
    )
    return table.sort_values(["missing_fraction", "column"], ascending=[False, True]).reset_index(drop=True)


def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    cols = numeric_columns(df)
    if not cols:
        return pd.DataFrame()
    summary = df[cols].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).T
    summary.insert(0, "column", summary.index)
    return summary.reset_index(drop=True)


def target_by_year(train: pd.DataFrame) -> pd.DataFrame:
    if TARGET_COL not in train.columns or "start_year" not in train.columns:
        return pd.DataFrame()
    grouped = train.groupby("start_year")[TARGET_COL]
    return grouped.agg(
        count="count",
        mean="mean",
        median="median",
        std="std",
        min="min",
        q01=lambda x: x.quantile(0.01),
        q05=lambda x: x.quantile(0.05),
        q95=lambda x: x.quantile(0.95),
        q99=lambda x: x.quantile(0.99),
        max="max",
    ).reset_index()


def sector_summary(train: pd.DataFrame) -> pd.DataFrame:
    if TARGET_COL not in train.columns or "sector_code" not in train.columns:
        return pd.DataFrame()
    grouped = train.groupby("sector_code")[TARGET_COL]
    summary = grouped.agg(count="count", mean="mean", median="median", std="std").reset_index()
    summary["sector_name"] = summary["sector_code"].map(SECTOR_MAPPING)
    return summary.sort_values("median", ascending=False)


def target_correlations(train: pd.DataFrame) -> pd.DataFrame:
    if TARGET_COL not in train.columns:
        return pd.DataFrame()
    cols = [col for col in numeric_columns(train) if col not in {TARGET_COL, ID_COL}]
    rows = []
    target = pd.to_numeric(train[TARGET_COL], errors="coerce")
    for col in cols:
        values = pd.to_numeric(train[col], errors="coerce")
        if values.notna().sum() < 5:
            continue
        corr = values.corr(target)
        if pd.isna(corr):
            continue
        rows.append(
            {
                "column": col,
                "pearson_corr_with_target": float(corr),
                "abs_corr_with_target": float(abs(corr)),
                "missing_fraction": float(values.isna().mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("abs_corr_with_target", ascending=False).reset_index(drop=True)


def train_test_shift(train: pd.DataFrame, test: pd.DataFrame | None) -> pd.DataFrame:
    if test is None:
        return pd.DataFrame()
    common_cols = [col for col in numeric_columns(train) if col in test.columns and col not in {TARGET_COL, ID_COL}]
    rows = []
    for col in common_cols:
        train_values = pd.to_numeric(train[col], errors="coerce")
        test_values = pd.to_numeric(test[col], errors="coerce")
        pooled_std = np.nanmean([train_values.std(), test_values.std()])
        standardized_mean_diff = (
            (test_values.mean() - train_values.mean()) / pooled_std if pooled_std and np.isfinite(pooled_std) else np.nan
        )
        rows.append(
            {
                "column": col,
                "train_mean": float(train_values.mean()),
                "test_mean": float(test_values.mean()),
                "standardized_mean_diff": float(standardized_mean_diff),
                "train_missing_fraction": float(train_values.isna().mean()),
                "test_missing_fraction": float(test_values.isna().mean()),
                "missing_fraction_diff": float(test_values.isna().mean() - train_values.isna().mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("standardized_mean_diff", key=lambda s: s.abs(), ascending=False)


def clipped_target(train: pd.DataFrame) -> pd.Series:
    target = pd.to_numeric(train[TARGET_COL], errors="coerce")
    low, high = target.quantile([0.01, 0.99])
    return target.clip(low, high)


def plot_target_hist(train: pd.DataFrame, output_dir: Path) -> Path | None:
    if TARGET_COL not in train.columns:
        return None
    plt = require_matplotlib()
    values = clipped_target(train).dropna()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(values, bins=60, color="#2f6f9f", alpha=0.85)
    ax.axvline(values.mean(), color="#c0392b", linewidth=2, label="mean")
    ax.axvline(values.median(), color="#2c3e50", linewidth=2, label="median")
    ax.set_title("Target distribution clipped to 1st-99th percentile for readability")
    ax.set_xlabel("return_pct")
    ax.set_ylabel("count")
    ax.legend()
    path = save_plot(fig, output_dir, "target_distribution_clipped.png")
    plt.close(fig)
    return path


def plot_target_by_year(train: pd.DataFrame, output_dir: Path) -> Path | None:
    if TARGET_COL not in train.columns or "start_year" not in train.columns:
        return None
    plt = require_matplotlib()
    years = sorted(train["start_year"].dropna().unique())
    data = [clipped_target(train.loc[train["start_year"] == year]).dropna().to_numpy() for year in years]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.boxplot(data, labels=[str(year) for year in years], showfliers=False)
    ax.set_title("Target by start_year, clipped for readability")
    ax.set_xlabel("start_year")
    ax.set_ylabel("return_pct")
    path = save_plot(fig, output_dir, "target_by_year_boxplot.png")
    plt.close(fig)
    return path


def plot_missingness(missing: pd.DataFrame, output_dir: Path, top_n: int) -> Path | None:
    top = missing.head(top_n).sort_values("missing_fraction")
    if top.empty:
        return None
    plt = require_matplotlib()
    fig, ax = plt.subplots(figsize=(9, max(5, top_n * 0.25)))
    ax.barh(top["column"], top["missing_fraction"] * 100.0, color="#7f8c8d")
    ax.set_title(f"Top {len(top)} missing columns in train")
    ax.set_xlabel("missing %")
    path = save_plot(fig, output_dir, "missingness_top_columns.png")
    plt.close(fig)
    return path


def plot_top_correlations(correlations: pd.DataFrame, output_dir: Path, top_n: int) -> Path | None:
    top = correlations.head(top_n).sort_values("abs_corr_with_target")
    if top.empty:
        return None
    plt = require_matplotlib()
    colors = np.where(top["pearson_corr_with_target"] >= 0, "#2874a6", "#b03a2e")
    fig, ax = plt.subplots(figsize=(9, max(5, top_n * 0.25)))
    ax.barh(top["column"], top["pearson_corr_with_target"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(f"Top {len(top)} numeric correlations with return_pct")
    ax.set_xlabel("Pearson correlation")
    path = save_plot(fig, output_dir, "target_correlations_top_columns.png")
    plt.close(fig)
    return path


def plot_sector_summary(summary: pd.DataFrame, output_dir: Path) -> Path | None:
    if summary.empty:
        return None
    plt = require_matplotlib()
    plot_df = summary.sort_values("median")
    labels = plot_df["sector_name"].fillna(plot_df["sector_code"].astype(str))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(labels, plot_df["median"], color="#2471a3")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("Median return_pct by sector")
    ax.set_xlabel("median return_pct")
    path = save_plot(fig, output_dir, "sector_median_target.png")
    plt.close(fig)
    return path


def plot_train_test_shift(shift: pd.DataFrame, output_dir: Path, top_n: int) -> Path | None:
    if shift.empty:
        return None
    top = shift.head(top_n).sort_values("standardized_mean_diff")
    plt = require_matplotlib()
    colors = np.where(top["standardized_mean_diff"] >= 0, "#2874a6", "#b03a2e")
    fig, ax = plt.subplots(figsize=(9, max(5, top_n * 0.25)))
    ax.barh(top["column"], top["standardized_mean_diff"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(f"Top {len(top)} train-test numeric mean shifts")
    ax.set_xlabel("standardized mean difference")
    path = save_plot(fig, output_dir, "train_test_shift_top_columns.png")
    plt.close(fig)
    return path


def plot_feature_scatter(train: pd.DataFrame, correlations: pd.DataFrame, output_dir: Path, sample_rows: int) -> Path | None:
    if correlations.empty or TARGET_COL not in train.columns:
        return None
    plt = require_matplotlib()
    features = correlations["column"].head(4).tolist()
    if not features:
        return None
    sample = train.sample(min(len(train), sample_rows), random_state=42) if len(train) > sample_rows else train
    y = clipped_target(sample)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, feature in zip(axes.ravel(), features):
        x = pd.to_numeric(sample[feature], errors="coerce")
        x_low, x_high = x.quantile([0.01, 0.99])
        ax.scatter(x.clip(x_low, x_high), y, s=8, alpha=0.35, color="#34495e")
        ax.set_title(feature)
        ax.set_xlabel(f"{feature} clipped")
        ax.set_ylabel("return_pct clipped")
    for ax in axes.ravel()[len(features) :]:
        ax.axis("off")
    path = save_plot(fig, output_dir, "top_feature_scatter_vs_target.png")
    plt.close(fig)
    return path


def format_markdown_value(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        value = float(value)
        if abs(value) >= 1000 or 0 < abs(value) < 0.001:
            return f"{value:.4g}"
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value).replace("|", "\\|").replace("\n", " ")


def dataframe_to_markdown(df: pd.DataFrame, max_rows: int = 12) -> list[str]:
    if df.empty:
        return ["_No rows available._"]

    display_df = df.head(max_rows).copy()
    columns = [str(col) for col in display_df.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in display_df.iterrows():
        lines.append("| " + " | ".join(format_markdown_value(row[col]) for col in display_df.columns) + " |")
    if len(df) > max_rows:
        lines.append("")
        lines.append(f"_Showing first {max_rows} of {len(df)} rows. See the CSV files for the full tables._")
    return lines


def add_markdown_table(lines: list[str], title: str, df: pd.DataFrame, max_rows: int = 12) -> None:
    lines.extend(["", f"## {title}", ""])
    lines.extend(dataframe_to_markdown(df, max_rows=max_rows))


def write_markdown_report(
    output_dir: Path,
    paths: dict[str, Path | None],
    overview: pd.DataFrame,
    train_missing: pd.DataFrame,
    by_year: pd.DataFrame,
    sectors: pd.DataFrame,
    correlations: pd.DataFrame,
    shift: pd.DataFrame,
) -> Path:
    report = output_dir / "eda_report.md"
    lines = [
        "# EDA Report",
        "",
        "Generated by `scripts/eda_report.py`.",
        "",
        "This report embeds the main diagnostic tables so the Markdown file is useful even when downloaded without the CSV files.",
        "",
        "## Quick Read",
        "",
    ]

    if not overview.empty:
        train_rows = overview.loc[overview["dataset"] == "train", "rows"]
        test_rows = overview.loc[overview["dataset"] == "test", "rows"]
        if not train_rows.empty:
            lines.append(f"- Train rows: {format_markdown_value(train_rows.iloc[0])}")
        if not test_rows.empty:
            lines.append(f"- Test rows: {format_markdown_value(test_rows.iloc[0])}")
        train_target = overview.loc[overview["dataset"] == "train"]
        if not train_target.empty and "target_std" in train_target.columns:
            lines.append(f"- Train target mean: {format_markdown_value(train_target['target_mean'].iloc[0])}")
            lines.append(f"- Train target std: {format_markdown_value(train_target['target_std'].iloc[0])}")

    if not correlations.empty:
        top_corr = correlations.iloc[0]
        lines.append(
            "- Strongest numeric linear target correlation: "
            f"`{top_corr['column']}` = {format_markdown_value(top_corr['pearson_corr_with_target'])}"
        )
    if not train_missing.empty:
        top_missing = train_missing.iloc[0]
        lines.append(
            "- Most missing train column: "
            f"`{top_missing['column']}` = {format_markdown_value(top_missing['missing_fraction'] * 100.0)}%"
        )
    if not shift.empty:
        top_shift = shift.iloc[0]
        lines.append(
            "- Largest train-test mean shift: "
            f"`{top_shift['column']}` = {format_markdown_value(top_shift['standardized_mean_diff'])} std units"
        )

    add_markdown_table(lines, "Dataset Overview", overview)
    add_markdown_table(lines, "Target By Year", by_year)
    add_markdown_table(lines, "Top Missing Train Columns", train_missing.head(20))
    add_markdown_table(lines, "Sector Target Summary", sectors)
    add_markdown_table(lines, "Top Target Correlations", correlations.head(20))
    if not shift.empty:
        add_markdown_table(lines, "Top Train-Test Feature Shifts", shift.head(20))

    lines.extend(
        [
            "",
            "## Output Files",
            "",
            "### Tables",
            "",
            "- `dataset_overview.csv`",
            "- `train_missingness.csv`",
            "- `numeric_summary_train.csv`",
            "- `target_by_year.csv`",
            "- `sector_summary.csv`",
            "- `target_correlations.csv`",
            "- `train_test_shift.csv` when test data is provided",
            "",
            "### Plots",
            "",
        ]
    )
    for label, path in paths.items():
        if path is not None:
            lines.append(f"- `{path.name}`: {label}")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    args = parse_args()
    output_dir = ensure_output_dir(args.output_dir)
    train = read_csv(args.train, required=True)
    test = read_csv(args.test, required=False)

    overview = dataset_overview(train, test)
    train_missing = missingness_table(train, "train")
    summary = numeric_summary(train)
    by_year = target_by_year(train)
    sectors = sector_summary(train)
    correlations = target_correlations(train)
    shift = train_test_shift(train, test)

    save_table(overview, output_dir, "dataset_overview.csv")
    save_table(train_missing, output_dir, "train_missingness.csv")
    save_table(summary, output_dir, "numeric_summary_train.csv")
    save_table(by_year, output_dir, "target_by_year.csv")
    save_table(sectors, output_dir, "sector_summary.csv")
    save_table(correlations, output_dir, "target_correlations.csv")
    if test is not None:
        save_table(missingness_table(test, "test"), output_dir, "test_missingness.csv")
        save_table(shift, output_dir, "train_test_shift.csv")

    plot_paths = {
        "target histogram, clipped to 1st-99th percentile": plot_target_hist(train, output_dir),
        "target boxplot by year": plot_target_by_year(train, output_dir),
        "top missing columns": plot_missingness(train_missing, output_dir, args.top_n),
        "top target correlations": plot_top_correlations(correlations, output_dir, args.top_n),
        "sector median target": plot_sector_summary(sectors, output_dir),
        "train-test feature shift": plot_train_test_shift(shift, output_dir, args.top_n),
        "top feature scatter plots": plot_feature_scatter(train, correlations, output_dir, args.sample_rows),
    }
    report_path = write_markdown_report(
        output_dir=output_dir,
        paths=plot_paths,
        overview=overview,
        train_missing=train_missing,
        by_year=by_year,
        sectors=sectors,
        correlations=correlations,
        shift=shift,
    )

    print(f"EDA report saved to {output_dir}")
    print(f"Open {report_path}")


if __name__ == "__main__":
    main()
