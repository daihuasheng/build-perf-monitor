import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

MIN_TOTAL_PRIMARY_METRIC_KB_FOR_PLOT = 10240  # 10MB, will apply to the primary metric


def _get_primary_metric_from_summary_log(csv_filepath: Path) -> Optional[str]:
    """
    Parses the corresponding _summary.log file to find the primary metric used.
    Example line: # Peak Overall Memory (PSS_KB): 12345 KB (at ...)
    Or from header: Memory Metric: PSS_PSUTIL (implies PSS_KB) or RSS_PIDSTAT (implies RSS_KB)
    """
    summary_log_filename = csv_filepath.stem + "_summary.log"
    # If csv is project_jX_mem_pss_psutil_ts.csv, summary is project_jX_mem_pss_psutil_ts_summary.log
    # Need to adjust if stem includes _summary itself if called with summary log
    if csv_filepath.name.endswith(
        "_summary.log"
    ):  # Should not happen if called with CSV
        return None

    # Construct summary log path based on CSV path
    # Assuming CSV is like: project_jX_mem_pss_psutil_timestamp.csv
    # Summary log is:      project_jX_mem_pss_psutil_timestamp_summary.log
    # So, csv_filepath.stem might be "project_jX_mem_pss_psutil_timestamp"
    # And summary log is f"{csv_filepath.stem}_summary.log" - this seems correct.

    summary_log_path = csv_filepath.parent / f"{csv_filepath.stem}_summary.log"

    if not summary_log_path.exists():
        logger.warning(
            f"Could not find summary log: {summary_log_path} to determine primary metric. Falling back to RSS_KB."
        )
        return "RSS_KB"  # Fallback, or could be None to raise error

    try:
        with open(summary_log_path, "r") as f:
            for line in f:
                # Try to get from "Peak Overall Memory (METRIC_KB): ..."
                match_peak = re.search(r"Peak Overall Memory \(([^)]+)\):", line)
                if match_peak:
                    metric = match_peak.group(1)
                    logger.info(
                        f"Determined primary metric '{metric}' from summary log (peak line)."
                    )
                    return metric
                # Fallback: Try to get from "Memory Metric: COLLECTOR_TYPE"
                match_collector = re.search(r"Memory Metric: (\w+)", line)
                if match_collector:
                    collector_type_short = match_collector.group(1).lower()
                    if "pss" in collector_type_short:  # PSS_PSUTIL
                        logger.info(
                            f"Determined primary metric 'PSS_KB' from summary log (collector type line: {collector_type_short})."
                        )
                        return "PSS_KB"
                    elif "rss" in collector_type_short:  # RSS_PIDSTAT
                        logger.info(
                            f"Determined primary metric 'RSS_KB' from summary log (collector type line: {collector_type_short})."
                        )
                        return "RSS_KB"
    except Exception as e:
        logger.error(
            f"Error reading or parsing summary log {summary_log_path}: {e}. Falling back to RSS_KB."
        )

    return "RSS_KB"  # Default fallback if not found


def _generate_line_plot(
    df: pd.DataFrame,
    primary_metric_col: str,
    resample_interval_str: str,
    csv_filepath: Path,
    output_dir: Path,
):
    """Generates and saves a line plot for memory usage using the primary_metric_col."""
    fig, ax = plt.subplots(figsize=(18, 10))
    all_categories_empty_after_resample = True
    category_resampled_data = {}

    for category, group_data in df.groupby("Category"):
        if group_data.empty or primary_metric_col not in group_data.columns:
            continue

        group_data_indexed = group_data.set_index("Timestamp")
        # Resample the primary metric
        resampled_metric = (
            group_data_indexed[primary_metric_col]
            .resample(resample_interval_str)
            .mean()
        )
        resampled_metric = resampled_metric.ffill(limit=2)

        if not resampled_metric.empty and not resampled_metric.isnull().all():
            all_categories_empty_after_resample = False
            ax.plot(
                resampled_metric.index,
                resampled_metric.values,
                label=category,
                marker=".",
                linestyle="-",
                markersize=5,
                linewidth=1.5,
            )
            category_resampled_data[category] = resampled_metric

    if (
        all_categories_empty_after_resample and not category_resampled_data
    ):  # Check if any data was plotted
        logger.warning(
            f"Line Plot: All categories in {csv_filepath.name} resulted in empty or all-NaN data for metric '{primary_metric_col}' after resampling to {resample_interval_str}. Skipping line plot."
        )
        plt.close(fig)
        return

    # Calculate and plot Total line if there's data
    if category_resampled_data:
        # Combine all resampled category data into a DataFrame, sum across categories for each timestamp
        combined_df = pd.DataFrame(category_resampled_data)
        total_resampled = combined_df.sum(axis=1)  # Sum across columns (categories)
        if not total_resampled.empty and not total_resampled.isnull().all():
            ax.plot(
                total_resampled.index,
                total_resampled.values,
                label="Total (Sum of Categories)",
                color="black",
                linestyle="--",
                linewidth=2,
            )

    ax.set_xlabel(f"Time (Resampled to {resample_interval_str} intervals)")
    ax.set_ylabel(f"Average {primary_metric_col} Memory Usage (KB)")
    ax.set_title(
        f"Memory Usage Over Time ({primary_metric_col} - Lines) - {csv_filepath.stem}\nResample: {resample_interval_str}",
        fontsize=14,
    )

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    fig.autofmt_xdate(rotation=30, ha="right")

    ax.legend(
        loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0.0, fontsize="small"
    )
    ax.grid(True, linestyle="--", alpha=0.7)

    plt.subplots_adjust(
        left=0.08, right=0.75, top=0.92, bottom=0.15
    )  # Adjusted right for potentially longer legend
    plot_filename = (
        output_dir / f"{csv_filepath.stem}_{primary_metric_col}_lines_plot.png"
    )
    plt.savefig(plot_filename, bbox_inches="tight")
    logger.info(f"Memory usage line plot saved to: {plot_filename}")
    plt.close(fig)


def _generate_stacked_area_plot(
    df: pd.DataFrame,
    primary_metric_col: str,
    resample_interval_str: str,
    csv_filepath: Path,
    output_dir: Path,
):
    """Generates and saves a stacked area plot for memory usage using the primary_metric_col."""
    if df.empty or primary_metric_col not in df.columns:
        logger.warning(
            f"Stacked Plot: No data or primary metric '{primary_metric_col}' to plot for {csv_filepath.name}. Skipping stacked plot."
        )
        return

    # Pivot table: Timestamp as index, Category as columns, mean of primary_metric_col as values
    pivot_df = (
        df.groupby(
            [pd.Grouper(key="Timestamp", freq=resample_interval_str), "Category"]
        )[primary_metric_col]
        .mean()
        .unstack(fill_value=0)
    )

    if pivot_df.empty or pivot_df.shape[1] == 0:
        logger.warning(
            f"Stacked Plot: Pivoted data is empty for {csv_filepath.name} (metric: {primary_metric_col}) after resampling to {resample_interval_str}. Skipping stacked plot."
        )
        return

    pivot_df[pivot_df < 0] = 0  # Ensure all values are non-negative

    fig, ax = plt.subplots(figsize=(18, 10))

    try:
        ax.stackplot(
            pivot_df.index,
            pivot_df.T.values,
            labels=pivot_df.columns.tolist(),
            alpha=0.8,
        )
    except Exception as e:
        logger.error(
            f"Error during stackplot generation for {csv_filepath.name}: {e}. Data shape: {pivot_df.shape}",
            exc_info=True,
        )
        plt.close(fig)
        return

    ax.set_xlabel(f"Time (Resampled to {resample_interval_str} intervals)")
    ax.set_ylabel(f"Total {primary_metric_col} Memory Usage (KB) - Stacked")
    ax.set_title(
        f"Memory Usage Over Time ({primary_metric_col} - Stacked Area) - {csv_filepath.stem}\nResample: {resample_interval_str}",
        fontsize=14,
    )

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    fig.autofmt_xdate(rotation=30, ha="right")

    ax.legend(
        loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0.0, fontsize="small"
    )
    ax.grid(True, linestyle=":", alpha=0.5)

    plt.subplots_adjust(left=0.08, right=0.75, top=0.92, bottom=0.15)  # Adjusted right
    plot_filename = (
        output_dir / f"{csv_filepath.stem}_{primary_metric_col}_stacked_plot.png"
    )
    plt.savefig(plot_filename, bbox_inches="tight")
    logger.info(f"Memory usage stacked area plot saved to: {plot_filename}")
    plt.close(fig)


def plot_memory_usage_from_csv(csv_filepath: Path, output_dir: Path):
    """
    Reads memory usage data from a CSV file and generates time-series plots.
    It determines the primary metric from the associated _summary.log file.
    """
    if not csv_filepath.exists():
        logger.error(f"CSV file not found: {csv_filepath}")
        return

    primary_metric_col = _get_primary_metric_from_summary_log(csv_filepath)
    if not primary_metric_col:
        logger.error(
            f"Could not determine primary metric for {csv_filepath.name}. Skipping plot generation."
        )
        return

    try:
        # Read the CSV. The CSV contains per-process data AND summary lines (CATEGORY_SUM, ALL_SUM).
        # We need to filter for per-process data.
        # Per-process lines have more columns than summary lines.
        # A robust way is to check the number of columns or the type of the second column.
        # For simplicity, we'll assume summary lines can be identified if the 'PID' column (expected in per-process) is NaN or missing.
        # Or, more directly, filter out rows where the second column is 'CATEGORY_SUM' or 'ALL_SUM'.

        # Read all lines first to determine column names for per-process data
        # The header of the CSV corresponds to per-process data.
        with open(csv_filepath, "r") as f:
            header_line = f.readline().strip()
        if not header_line:
            logger.warning(
                f"CSV file {csv_filepath} is empty or has no header. Skipping plot."
            )
            return

        all_csv_columns = header_line.split(",")

        # Read the CSV, expecting the header to define columns for per-process data
        df = pd.read_csv(
            csv_filepath, comment="#"
        )  # comment='#' is fine, summary log has no comments
        logger.info(f"Successfully read CSV: {csv_filepath} with {len(df)} rows.")

        if df.empty:
            logger.warning(
                f"No data found in {csv_filepath} after initial read. Skipping plot."
            )
            return

        # Filter out summary rows ('CATEGORY_SUM', 'ALL_SUM') based on the second column's value.
        # The second column in per-process data is 'Category'. For summary rows, it's 'CATEGORY_SUM' or 'ALL_SUM'.
        # Let's assume the second column from the header is 'Category' for per-process data.
        # If the CSV structure is Timestamp_epoch, Type_Marker/Category, ...
        # The `monitor_utils.py` writes the Type_Marker ('CATEGORY_SUM', 'ALL_SUM') into the second column for these rows.
        # For actual data rows, the second column is the actual category name.

        # Identify per-process data rows. These rows should NOT have 'CATEGORY_SUM' or 'ALL_SUM' in their second column.
        # The second column name in the header is 'Category'.
        if "Category" not in df.columns:  # Should be the second column from header
            logger.error(
                f"CSV file {csv_filepath.name} does not have the expected 'Category' (second) column in its header. Columns: {df.columns.tolist()}"
            )
            return

        # Filter out rows that are summary rows
        df_per_process = df[
            ~df.iloc[:, 1].isin(["CATEGORY_SUM", "ALL_SUM"])
        ].copy()  # Use .iloc[:,1] for second column by position

        if df_per_process.empty:
            logger.warning(
                f"No per-process data rows found in {csv_filepath.name} after filtering summary rows. Skipping plot."
            )
            return

        logger.info(
            f"Found {len(df_per_process)} per-process data rows in {csv_filepath.name}."
        )

        required_cols = ["Timestamp_epoch", "Category", primary_metric_col]
        if not all(col in df_per_process.columns for col in required_cols):
            logger.error(
                f"Per-process data in {csv_filepath} is missing one or more required columns for plotting: {required_cols}. Found: {df_per_process.columns.tolist()}"
            )
            return

        # Ensure primary_metric_col is numeric, coercing errors to NaN
        df_per_process[primary_metric_col] = pd.to_numeric(
            df_per_process[primary_metric_col], errors="coerce"
        )
        df_per_process.dropna(
            subset=[primary_metric_col], inplace=True
        )  # Drop rows where primary metric is NaN after coercion

        if df_per_process.empty:
            logger.warning(
                f"No valid numeric data for primary metric '{primary_metric_col}' in {csv_filepath.name}. Skipping plot."
            )
            return

        # 1. Reclassify old categories (for compatibility with older CSVs) - applied to df_per_process
        reclassification_map = {
            "py_decodetree": "script_python",
            "py_qapi_gen": "script_python",
        }
        df_per_process["Category"] = df_per_process["Category"].replace(
            reclassification_map
        )

        df_per_process = df_per_process[
            ~df_per_process["Category"].str.startswith("ignore_")
        ]
        if df_per_process.empty:
            logger.warning(
                f"No data left after filtering ignored categories in {csv_filepath}. Skipping plot."
            )
            return

        # 2. Filter categories with total primary_metric < MIN_TOTAL_PRIMARY_METRIC_KB_FOR_PLOT
        category_total_metric = df_per_process.groupby("Category")[
            primary_metric_col
        ].sum()
        significant_categories = category_total_metric[
            category_total_metric >= MIN_TOTAL_PRIMARY_METRIC_KB_FOR_PLOT
        ].index.tolist()

        if not significant_categories:
            logger.warning(
                f"No categories with total {primary_metric_col} >= {MIN_TOTAL_PRIMARY_METRIC_KB_FOR_PLOT}KB found in {csv_filepath.name}. Skipping plots."
            )
            return

        df_plot_data = df_per_process[
            df_per_process["Category"].isin(significant_categories)
        ].copy()  # Use .copy()
        logger.info(
            f"Plotting for significant categories in {csv_filepath.name} (metric: {primary_metric_col}): {significant_categories}"
        )

        df_plot_data["Timestamp"] = pd.to_datetime(
            df_plot_data["Timestamp_epoch"], unit="s"
        )

        if df_plot_data.empty or len(df_plot_data["Timestamp"]) < 1:
            logger.warning(
                f"Not enough data points in {csv_filepath.name} after filtering to determine duration or plot. Skipping."
            )
            return

        min_time = df_plot_data["Timestamp"].min()
        max_time = df_plot_data["Timestamp"].max()
        duration_seconds = (
            (max_time - min_time).total_seconds()
            if pd.notna(min_time)
            and pd.notna(max_time)
            and len(df_plot_data["Timestamp"]) > 1
            else 0
        )

        if duration_seconds <= 10:
            resample_interval_str = "1S"
        elif duration_seconds < 60:
            resample_interval_str = "5S"
        elif duration_seconds < 300:
            resample_interval_str = "10S"
        elif duration_seconds < 900:
            resample_interval_str = "30S"
        elif duration_seconds < 3600:
            resample_interval_str = "1T"
        elif duration_seconds < 3 * 3600:
            resample_interval_str = "2T"
        else:
            resample_interval_str = "5T"
        logger.info(
            f"Data duration for {csv_filepath.name}: {duration_seconds:.0f}s. Dynamic resample interval: {resample_interval_str}"
        )

        _generate_line_plot(
            df_plot_data.copy(),
            primary_metric_col,
            resample_interval_str,
            csv_filepath,
            output_dir,
        )
        _generate_stacked_area_plot(
            df_plot_data.copy(),
            primary_metric_col,
            resample_interval_str,
            csv_filepath,
            output_dir,
        )

    except pd.errors.EmptyDataError:
        logger.warning(
            f"No data or columns to parse in {csv_filepath} (Pandas EmptyDataError). Skipping plot."
        )
    except Exception as e:
        logger.error(f"Error generating plots for {csv_filepath}: {e}", exc_info=True)


def generate_plots_for_logs(log_dir: Path):
    """
    Generates plots for all .csv files in the specified log directory.
    Plots will be saved in the same directory as the CSV files.
    """
    logger.info(f"Searching for CSV log files in: {log_dir} to generate plots.")
    csv_files = list(log_dir.glob("*.csv"))

    if not csv_files:
        logger.info(f"No CSV log files found in {log_dir} to generate plots for.")
        return

    for csv_file in csv_files:
        # Ensure we are not trying to plot a summary log file itself if it ends with .csv by mistake
        if "_summary.log" in csv_file.name:  # A bit of a heuristic
            logger.info(
                f"Skipping potential summary log file from plotting: {csv_file.name}"
            )
            continue
        logger.info(f"--- Generating plot for {csv_file.name} ---")
        plot_memory_usage_from_csv(csv_file, log_dir)

    # Removed the extraneous plotting code that was here previously
