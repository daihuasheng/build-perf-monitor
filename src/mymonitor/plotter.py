"""
Generates plots from monitoring data files.

This module is responsible for the final visualization step of the monitoring
process. It reads the detailed data from Parquet files generated during a run,
processes this data using the Polars library, and creates interactive time-series
plots using Plotly.

The main functionalities include:
- Automatically discovering all `.parquet` data files in a given log directory.
- Parsing the associated `_summary.log` file to determine the primary memory
  metric (e.g., PSS_KB or RSS_KB) that was used for the run.
- Filtering data to focus on significant processes and categories, reducing noise.
- Dynamically adjusting the time-series resampling interval based on the total
  duration of the build to ensure plots are readable.
- Generating two types of plots for each data file:
  1. A line plot showing the average memory usage per category.
  2. A stacked area plot showing the total memory usage composition over time.
- Saving plots as interactive HTML files and, if Kaleido is installed, as
  static PNG images.
"""

import logging
import re
from pathlib import Path
from typing import Optional

# Third-party library imports
import polars as pl
import plotly.express as px
import plotly.graph_objects as go

logger = logging.getLogger(__name__)

# --- Module Constants ---

# A threshold (in KB) to filter out categories with insignificant memory usage
# from the plots, keeping them clean and focused. 10MB = 10 * 1024 KB.
MIN_TOTAL_PRIMARY_METRIC_KB_FOR_PLOT = 10240


def _get_primary_metric_from_summary_log(data_filepath: Path) -> Optional[str]:
    """
    Parses the corresponding _summary.log file to find the primary metric used.

    This helper function is crucial for making the plotting logic independent of
    the collector used. It finds the summary log associated with a given data file
    and reads it to determine whether PSS or RSS was the main metric.

    Args:
        data_filepath: Path to the main data file (e.g., `.parquet`). The function
                       will look for a summary log file with a similar name.

    Returns:
        The primary metric string (e.g., "PSS_KB", "RSS_KB") if found,
        otherwise "RSS_KB" as a fallback. Returns None if the input path is
        itself a summary log.
    """
    # Do not process summary logs themselves.
    if data_filepath.name.endswith("_summary.log"):
        return None

    summary_log_path = data_filepath.with_name(f"{data_filepath.stem}_summary.log")
    if not summary_log_path.exists():
        logger.warning(
            f"Could not find summary log: {summary_log_path} to determine primary metric. Falling back to RSS_KB."
        )
        return "RSS_KB"

    try:
        with open(summary_log_path, "r") as f:
            for line in f:
                # First, try to find the most explicit line.
                match_peak = re.search(r"Peak Overall Memory \(([^)]+)\):", line)
                if match_peak:
                    metric = match_peak.group(1)
                    logger.info(
                        f"Determined primary metric '{metric}' from summary log (peak line)."
                    )
                    return metric
                # As a fallback, check the collector type line.
                match_collector = re.search(r"Memory Metric Collector: (\w+)", line)
                if match_collector:
                    collector_type_short = match_collector.group(1).lower()
                    if "pss" in collector_type_short:
                        return "PSS_KB"
                    elif "rss" in collector_type_short:
                        return "RSS_KB"
    except Exception as e:
        logger.error(
            f"Error reading or parsing summary log {summary_log_path}: {e}. Falling back to RSS_KB."
        )
    return "RSS_KB"  # Default fallback if parsing fails.


def _save_plotly_figure(fig: go.Figure, base_filename: str, output_dir: Path):
    """
    Saves a Plotly figure to both HTML and, if possible, PNG formats.

    Args:
        fig: The Plotly figure object to save.
        base_filename: The base name for the output files (without extension).
        output_dir: The directory to save the files in.
    """
    plot_filename_html = output_dir / f"{base_filename}.html"
    try:
        fig.write_html(plot_filename_html)
        logger.info(f"Interactive plot saved to: {plot_filename_html}")
        # Attempt to save a static PNG image if Kaleido is installed.
        try:
            plot_filename_png = output_dir / f"{base_filename}.png"
            fig.write_image(plot_filename_png, width=1200, height=600)
            logger.info(f"Static plot saved to: {plot_filename_png}")
        except Exception as e_kaleido:
            # This is a non-critical failure, so log it as a warning.
            logger.warning(
                f"Failed to save static plot to PNG (Kaleido might be missing or misconfigured): {e_kaleido}. "
                f"To enable PNG export, install Kaleido: `uv pip install mymonitor[export]`"
            )
    except Exception as e:
        logger.error(
            f"Failed to save plot {plot_filename_html} using Plotly: {e}",
            exc_info=True,
        )


def _generate_line_plot_plotly(
    df_plot_data: pl.DataFrame,
    primary_metric_col: str,
    resample_interval_str: str,
    data_filepath: Path,
    output_dir: Path,
):
    """
    Generates and saves an interactive line plot using Plotly.

    The plot shows the average of the primary metric over time for each category,
    resampled to the specified interval. A 'Total' line representing the sum
    of these averages across categories is also added for context.

    Args:
        df_plot_data: Polars DataFrame containing the data to plot.
        primary_metric_col: Name of the column containing the primary metric data.
        resample_interval_str: Polars interval string (e.g., "1s", "5m") for resampling.
        data_filepath: Path to the original Parquet file (used for naming the output).
        output_dir: Directory where the generated HTML plot file will be saved.
    """
    if df_plot_data.is_empty():
        logger.warning(
            f"Line Plot: Input data is empty for {data_filepath.name}. Skipping."
        )
        return

    # Resample data for each category individually to correctly calculate the mean
    # for each one before combining them.
    resampled_dfs_list = []
    for category_name_tuple, group_df in df_plot_data.group_by(
        "Category", maintain_order=True
    ):
        if group_df.is_empty():
            continue

        resampled_cat_df = (
            group_df.sort("Timestamp")
            .group_by_dynamic(
                index_column="Timestamp",
                every=resample_interval_str,
                group_by="Category",
            )
            .agg(pl.col(primary_metric_col).mean().alias(primary_metric_col))
            .fill_null(0)  # Fill gaps that result from resampling sparse data.
        )
        if not resampled_cat_df.is_empty():
            resampled_dfs_list.append(resampled_cat_df)

    if not resampled_dfs_list:
        logger.warning(
            f"Line Plot: No data after resampling for {data_filepath.name}. Skipping."
        )
        return

    combined_resampled_df = pl.concat(resampled_dfs_list)

    if combined_resampled_df.is_empty():
        logger.warning(
            f"Line Plot: Combined resampled data is empty for {data_filepath.name}. Skipping."
        )
        return

    # Generate the line plot using Plotly Express.
    fig = px.line(
        combined_resampled_df.to_pandas(),  # Plotly Express often prefers Pandas.
        x="Timestamp",
        y=primary_metric_col,
        color="Category",  # Creates a different line for each category.
        title=f"Memory Usage Over Time ({primary_metric_col} - Lines) - {data_filepath.stem}<br>Resample: {resample_interval_str}",
        labels={
            "Timestamp": f"Time (Resampled to {resample_interval_str})",
            primary_metric_col: f"Average {primary_metric_col} (KB)",
        },
        markers=True,
    )

    # Calculate and add a 'Total' line by summing the means of all categories at each point.
    total_df = (
        combined_resampled_df.group_by("Timestamp")
        .agg(pl.col(primary_metric_col).sum().alias("Total_Memory"))
        .sort("Timestamp")
    )

    if not total_df.is_empty():
        fig.add_trace(
            go.Scatter(
                x=total_df["Timestamp"].to_list(),
                y=total_df["Total_Memory"].to_list(),
                mode="lines",
                name="Total (Sum of Categories)",
                line={"color": "black", "dash": "dash"},
            )
        )

    fig.update_layout(
        legend_title_text="Category",
        xaxis_title=f"Time (Resampled to {resample_interval_str} intervals)",
        yaxis_title=f"Average {primary_metric_col} Memory Usage (KB)",
    )

    base_plot_filename = f"{data_filepath.stem}_{primary_metric_col}_lines_plot"
    _save_plotly_figure(fig, base_plot_filename, output_dir)


def _generate_stacked_area_plot_plotly(
    df_plot_data: pl.DataFrame,
    primary_metric_col: str,
    resample_interval_str: str,
    data_filepath: Path,
    output_dir: Path,
):
    """
    Generates and saves an interactive stacked area plot using Plotly.

    The plot shows the mean of the primary metric for each category, stacked
    over time, after resampling to the specified interval. This is useful for
    visualizing the overall memory composition.

    Args:
        df_plot_data: Polars DataFrame containing the data to plot.
        primary_metric_col: Name of the column containing the primary metric data.
        resample_interval_str: Polars interval string (e.g., "1s", "5m") for resampling.
        data_filepath: Path to the original Parquet file (used for naming the output).
        output_dir: Directory where the generated HTML plot file will be saved.
    """
    if df_plot_data.is_empty():
        logger.warning(
            f"Stacked Plot: Input data is empty for {data_filepath.name}. Skipping."
        )
        return

    # Resample data, taking the mean for each category within each interval.
    resampled_df = (
        df_plot_data.sort("Timestamp")
        .group_by_dynamic(
            index_column="Timestamp",
            every=resample_interval_str,
            group_by="Category",
        )
        .agg(pl.col(primary_metric_col).mean().fill_null(0).alias(primary_metric_col))
    )

    if resampled_df.is_empty():
        logger.warning(
            f"Stacked Plot: Resampled data is empty for {data_filepath.name}. Skipping."
        )
        return

    # Generate the stacked area plot.
    fig = px.area(
        resampled_df.to_pandas(),
        x="Timestamp",
        y=primary_metric_col,
        color="Category",  # This column's values will be stacked.
        title=f"Memory Usage Over Time ({primary_metric_col} - Stacked Area) - {data_filepath.stem}<br>Resample: {resample_interval_str}",
        labels={
            "Timestamp": f"Time (Resampled to {resample_interval_str})",
            primary_metric_col: f"{primary_metric_col} (KB)",
        },
    )

    fig.update_layout(
        legend_title_text="Category",
        xaxis_title=f"Time (Resampled to {resample_interval_str} intervals)",
        yaxis_title=f"Total {primary_metric_col} Memory Usage (KB) - Stacked",
    )

    base_plot_filename = f"{data_filepath.stem}_{primary_metric_col}_stacked_plot"
    _save_plotly_figure(fig, base_plot_filename, output_dir)


def plot_memory_usage_from_data_file(data_filepath: Path, output_dir: Path):
    """
    Reads memory usage data from a Parquet file and generates time-series plots.

    This is the main worker function for plotting. It processes a single Parquet
    file, determines the primary metric, filters and preprocesses the data, and
    then generates both line and stacked area plots.

    Args:
        data_filepath: Path to the input Parquet file.
        output_dir: Directory where the generated HTML plot files will be saved.
    """
    if not data_filepath.exists():
        logger.error(f"Data file not found: {data_filepath}")
        return

    primary_metric_col = _get_primary_metric_from_summary_log(data_filepath)
    if not primary_metric_col:
        logger.error(
            f"Could not determine primary metric for {data_filepath.name}. Skipping plot generation."
        )
        return

    try:
        # --- Data Loading and Initial Filtering ---
        df_pl = pl.read_parquet(data_filepath)
        if df_pl.is_empty():
            logger.warning(f"No data found in {data_filepath}. Skipping plot.")
            return

        # Filter for per-process data rows, ignoring summary rows.
        df_per_process_pl = df_pl.filter(pl.col("Record_Type") == "PROCESS")
        if df_per_process_pl.is_empty():
            logger.warning(
                f"No per-process data rows in {data_filepath.name}. Skipping."
            )
            return

        # --- Data Preprocessing and Cleaning ---
        # Create a unified 'Category' column for plotting.
        df_per_process_pl = df_per_process_pl.with_columns(
            (pl.col("Major_Category") + "_" + pl.col("Minor_Category")).alias(
                "Category"
            )
        )

        # Ensure required columns exist.
        required_cols = ["Timestamp_epoch", "Category", primary_metric_col]
        if not all(col in df_per_process_pl.columns for col in required_cols):
            missing = [c for c in required_cols if c not in df_per_process_pl.columns]
            logger.error(
                f"Data in {data_filepath} missing required columns: {missing}."
            )
            return

        # Ensure primary metric column is numeric and filter out non-numeric data.
        df_per_process_pl = df_per_process_pl.with_columns(
            pl.col(primary_metric_col).cast(pl.Float64, strict=False)
        ).filter(pl.col(primary_metric_col).is_not_null())

        if df_per_process_pl.is_empty():
            logger.warning(
                f"No valid numeric data for '{primary_metric_col}' in {data_filepath.name}. Skipping."
            )
            return

        # Filter out categories marked to be ignored.
        df_per_process_pl = df_per_process_pl.filter(
            ~pl.col("Category").str.starts_with("Ignored_")
        )
        if df_per_process_pl.is_empty():
            logger.warning("No data after filtering ignored categories. Skipping.")
            return

        # Filter for categories with significant memory usage to keep plots clean.
        category_total_metric = df_per_process_pl.group_by("Category").agg(
            pl.col(primary_metric_col).sum().alias("total_metric")
        )
        significant_categories = category_total_metric.filter(
            pl.col("total_metric") >= MIN_TOTAL_PRIMARY_METRIC_KB_FOR_PLOT
        )["Category"].to_list()

        if not significant_categories:
            logger.warning(
                f"No significant categories found in {data_filepath.name}. Skipping plots."
            )
            return

        df_plot_data_pl = df_per_process_pl.filter(
            pl.col("Category").is_in(significant_categories)
        )
        logger.info(f"Plotting for significant categories: {significant_categories}")

        # Convert epoch seconds to Polars Datetime for time-series analysis.
        df_plot_data_pl = df_plot_data_pl.with_columns(
            pl.from_epoch("Timestamp_epoch", time_unit="s").alias("Timestamp")
        )

        # --- Dynamic Resampling Interval ---
        # Determine a suitable resampling interval based on the total run duration.
        min_time_dt = df_plot_data_pl["Timestamp"].min()
        max_time_dt = df_plot_data_pl["Timestamp"].max()
        duration_seconds = (
            (max_time_dt - min_time_dt).total_seconds()
            if min_time_dt and max_time_dt
            else 0
        )

        if duration_seconds <= 60:  # Up to 1 min
            resample_interval_str_polars = "5s"
        elif duration_seconds < 300:  # Up to 5 mins
            resample_interval_str_polars = "10s"
        elif duration_seconds < 900:  # Up to 15 mins
            resample_interval_str_polars = "30s"
        else:  # More than 15 mins
            resample_interval_str_polars = "1m"
        logger.info(
            f"Data duration: {duration_seconds:.0f}s. Resample interval: {resample_interval_str_polars}"
        )

        # --- Plot Generation ---
        _generate_line_plot_plotly(
            df_plot_data_pl,
            primary_metric_col,
            resample_interval_str_polars,
            data_filepath,
            output_dir,
        )
        _generate_stacked_area_plot_plotly(
            df_plot_data_pl,
            primary_metric_col,
            resample_interval_str_polars,
            data_filepath,
            output_dir,
        )

    except pl.exceptions.NoDataError:
        logger.warning(
            f"No data in {data_filepath} (Polars NoDataError). Skipping plot."
        )
    except Exception as e:
        logger.error(f"Error generating plots for {data_filepath}: {e}", exc_info=True)


def generate_plots_for_logs(log_dir: Path):
    """
    Generates plots for all relevant .parquet files in the specified log directory.

    This is the main public entry point for the plotter module. It finds all
    `.parquet` files in the given directory and calls the main worker function
    to generate plots for each one.

    Args:
        log_dir: The directory containing the log files from a monitoring run.
    """
    logger.info(f"Searching for Parquet data files in: {log_dir} to generate plots.")
    parquet_files = list(log_dir.glob("*.parquet"))

    if not parquet_files:
        logger.info(f"No Parquet data files found in {log_dir} to generate plots for.")
        return

    for data_file in parquet_files:
        logger.info(f"--- Generating plot for {data_file.name} ---")
        plot_memory_usage_from_data_file(data_file, log_dir)
