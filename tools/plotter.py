"""Standalone Command-Line Tool for Generating Plots from MyMonitor Data.

This script serves as a powerful, independent utility for visualizing performance
data collected by the MyMonitor application. It reads `.parquet` data files
from a specified run directory and generates interactive, insightful plots
using the Plotly library.

The tool can be executed in two main modes:
1.  **Detailed Plot Mode (Default)**: For each data file found, it generates
    time-series plots (line and/or stacked area) showing memory usage for
    different process categories over time.
2.  **Summary Plot Mode (`--summary-plot`)**: It scans all `_summary.log` files
    in the directory to create a single, comparative chart, plotting key
    metrics like total build time and peak memory usage against the different
    parallelism levels (`-j N`) used in the run.

It can be invoked automatically by the main `mymonitor` application after a
monitoring session or run manually by a user to re-analyze data or customize
plots with various filtering options.

Usage examples:
  # Generate all default plots for a specific run
  python tools/plotter.py --log-dir /path/to/logs/run_20250624_103000

  # Generate only a summary plot for the same run
  python tools/plotter.py --log-dir /path/to/logs/run_20250624_103000 --summary-plot

  # Generate a detailed line plot for a specific project and job level
  python tools/plotter.py --log-dir /path/to/logs/run_20250624_103000 \
    --project-name my_project --jobs 8 --chart-type line
"""

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Third-party library imports
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import polars as pl

# --- Logging Setup ---
# Configure a global logger for this script to provide informative output.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("PlotterTool")

# --- Module Constants ---

# A threshold (in KB) to filter out categories with insignificant memory usage
# from the detailed plots. This helps keep the final charts clean and focused
# on the most impactful processes. 10MB = 10 * 1024 KB.
MIN_TOTAL_PRIMARY_METRIC_KB_FOR_PLOT = 10240


# --- Helper Functions ---


def _parse_summary_log(filepath: Path) -> Optional[Dict[str, Any]]:
    """Parses a _summary.log file to extract key metrics for the summary plot.

    This function reads a given summary log file and uses regular expressions
    to robustly extract essential run metrics.

    Args:
        filepath: The path to the `_summary.log` file.

    Returns:
        A dictionary containing the parsed metrics (project, Parallelism,
        Duration_sec, Peak_Memory_KB, and metric type) if all fields are
        found, otherwise None.
    """
    try:
        content = filepath.read_text()

        # Use regex to find the required values robustly, ignoring surrounding text.
        project_match = re.search(r"Project:\s*(.*)", content)
        jobs_match = re.search(r"Parallelism:\s*-j(\d+)", content)
        duration_match = re.search(
            r"Total Build & Monitoring Duration:\s*.*?\((\d+\.\d+) seconds\)", content
        )
        peak_mem_match = re.search(
            r"Peak Overall Memory \(([^)]+)\):\s*(\d+)\s*KB", content
        )

        # Ensure all required pieces of information were successfully extracted.
        if all([project_match, jobs_match, duration_match, peak_mem_match]):
            # Add type assertions to satisfy the linter
            assert project_match is not None
            assert jobs_match is not None
            assert duration_match is not None
            assert peak_mem_match is not None
            
            # Return a dictionary with standardized keys for DataFrame creation.
            return {
                "project": project_match.group(1).strip(),
                "Parallelism": int(jobs_match.group(1)),
                "Duration_sec": float(duration_match.group(1)),
                "Peak_Memory_KB": int(peak_mem_match.group(2)),
                "metric": peak_mem_match.group(1),
            }
    except Exception as e:
        logger.error(f"Could not parse summary log {filepath.name}: {e}")
    return None


def _create_summary_figure(
    group_df: pd.DataFrame, project_name: str, metric: str
) -> go.Figure:
    """Creates the Plotly figure object for the summary plot.

    This function constructs a dual-axis chart to compare build duration
    (line plot on secondary y-axis) against peak memory usage (bar plot on
    primary y-axis) across different parallelism levels.

    Args:
        group_df: A pandas DataFrame containing the aggregated data for a
                  single project, with columns 'Parallelism', 'Peak_Memory_KB',
                  and 'Duration_sec'.
        project_name: The name of the project being plotted, used for the title.
        metric: The primary memory metric used (e.g., 'PSS_KB'), for axis labels.

    Returns:
        A configured Plotly Figure object ready for saving.
    """
    # Sort data by parallelism level for a coherent plot.
    group_df = group_df.sort_values("Parallelism")

    fig = go.Figure()

    # --- Bar chart for Peak Memory (Primary Y-axis) ---
    fig.add_trace(
        go.Bar(
            x=group_df["Parallelism"],
            y=group_df["Peak_Memory_KB"],
            name="Peak Memory (KB)",
            marker_color="indianred",
            text=group_df["Peak_Memory_KB"].round(0),
            textposition="auto",
        )
    )

    # --- Line chart for Build Duration (Secondary Y-axis) ---
    fig.add_trace(
        go.Scatter(
            x=group_df["Parallelism"],
            y=group_df["Duration_sec"],
            name="Build Duration (sec)",
            marker_color="cornflowerblue",
            yaxis="y2",  # Assign this trace to the secondary y-axis.
            mode="lines+markers",
        )
    )

    # --- Layout Configuration ---
    # Configure titles, axes, and legend for clarity.
    fig.update_layout(
        title_text=f"Build Summary: Peak Memory & Duration vs. Parallelism for '{project_name}'",
        xaxis=dict(
            title_text="Parallelism Level (-j)",
            type="category",  # Treat job levels as distinct categories.
            title_font=dict(size=14),
            tickfont=dict(size=12),
        ),
        yaxis=dict(
            title_text=f"Peak Memory ({metric})",
            title_font=dict(size=14, color="indianred"),
            tickfont=dict(size=12, color="indianred"),
        ),
        yaxis2=dict(
            title_text="Build Duration (seconds)",
            title_font=dict(size=14, color="cornflowerblue"),
            tickfont=dict(size=12, color="cornflowerblue"),
            overlaying="y",  # This y-axis is an overlay on top of the primary one.
            side="right",
        ),
        legend=dict(x=0.01, y=0.98, bordercolor="Black", borderwidth=1),
        barmode="group",
    )

    return fig


def generate_run_summary_plot(args: argparse.Namespace):
    """Orchestrates the generation of the run summary plot.

    This function scans the log directory for all `_summary.log` files,
    parses them, groups the data by project, and then calls the plotting
    and saving functions for each project with sufficient data.

    Args:
        args: The parsed command-line arguments from argparse.
    """
    logger.info("--- Generating Run Summary Plot ---")
    summary_logs = list(args.log_dir.glob("*_summary.log"))

    if not summary_logs:
        logger.warning("No summary logs found. Cannot generate summary plot.")
        return

    # Parse all found log files and filter out any that failed parsing.
    summary_data = [_parse_summary_log(f) for f in summary_logs]
    summary_data = [d for d in summary_data if d]

    if not summary_data:
        logger.warning("Could not parse any summary logs successfully.")
        return

    # Convert the list of dicts into a pandas DataFrame for easy manipulation.
    df_summary = pd.DataFrame(summary_data)

    # Group data by project name to generate a separate plot for each.
    for project_name, group_df in df_summary.groupby("project"):
        # Ensure project_name is a string for type safety
        project_name_str = str(project_name)
        
        # A comparison plot requires at least two data points.
        if len(group_df) < 2:
            logger.info(
                f"Skipping summary plot for '{project_name_str}': only one data point found."
            )
            continue

        # Assume the metric is the same for all runs of a single project.
        metric = group_df["metric"].iloc[0]
        fig = _create_summary_figure(group_df, project_name_str, metric)

        base_filename = f"{project_name_str}_build_summary_plot"
        _save_plotly_figure(fig, base_filename, args.output_dir or args.log_dir)


def _get_primary_metric_from_summary_log(data_filepath: Path) -> Optional[str]:
    """Parses the corresponding _summary.log file to find the primary metric.

    For a given data file (e.g., `proj_j4.parquet`), this function looks for
    `proj_j4_summary.log` to determine which memory metric (e.g., PSS_KB,
    RSS_KB) was used for the run. This allows the plotting functions to use
    the correct data column.

    Args:
        data_filepath: The path to the `.parquet` data file.

    Returns:
        The name of the primary metric column (e.g., "PSS_KB") as a string,
        or a fallback value if the log is not found or cannot be parsed.
        Returns None if the input file itself is a summary log.
    """
    if data_filepath.name.endswith("_summary.log"):
        return None

    summary_log_path = data_filepath.with_name(f"{data_filepath.stem}_summary.log")
    if not summary_log_path.exists():
        logger.warning(
            f"Could not find summary log: {summary_log_path} to determine primary metric. Falling back to RSS_KB."
        )
        return "RSS_KB"

    try:
        content = summary_log_path.read_text()
        # First, try to find the explicit peak memory line, which is most reliable.
        match_peak = re.search(r"Peak Overall Memory \(([^)]+)\):", content)
        if match_peak:
            metric = match_peak.group(1)
            logger.info(
                f"Determined primary metric '{metric}' from summary log (peak line)."
            )
            return metric
        # As a fallback, check the collector type line.
        match_collector = re.search(r"Memory Metric Collector: (\w+)", content)
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
    # Default fallback if all parsing attempts fail.
    return "RSS_KB"


def _save_plotly_figure(fig: go.Figure, base_filename: str, output_dir: Path):
    """Saves a Plotly figure to both HTML and, if possible, PNG formats.

        This function handles the file I/O for saving plots. It always saves an
        interactive HTML file. It also attempts to save a static PNG image,
    t    gracefully handling the case where the optional `kaleido` dependency
        is not installed.

        Args:
            fig: The Plotly Figure object to save.
            base_filename: The base name for the output files (without extension).
            output_dir: The directory where the plot files will be saved.
    """
    plot_filename_html = output_dir / f"{base_filename}.html"
    try:
        fig.write_html(plot_filename_html)
        logger.info(f"Interactive plot saved to: {plot_filename_html}")

        # Attempt to save to PNG, which requires the 'kaleido' package.
        try:
            plot_filename_png = output_dir / f"{base_filename}.png"
            fig.write_image(plot_filename_png, width=1200, height=600)
            logger.info(f"Static plot saved to: {plot_filename_png}")
        except Exception:
            # This is not a critical error. Inform the user how to enable it.
            logger.warning(
                "Failed to save static plot to PNG. To enable this feature, "
                "install the optional 'export' dependencies: "
                "`uv pip install mymonitor[export]`"
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
    """Generates and saves an interactive line plot using Plotly.

    Args:
        df_plot_data: A Polars DataFrame containing the filtered data to plot.
        primary_metric_col: The name of the column containing memory data.
        resample_interval_str: The Polars interval string for resampling (e.g., '5s').
        data_filepath: The path of the original data file, for titling.
        output_dir: The directory to save the plot file.
    """
    if df_plot_data.is_empty():
        logger.warning(
            f"Line Plot: Input data is empty for {data_filepath.name}. Skipping."
        )
        return

    # Resample data for each category to create smoother lines.
    resampled_dfs_list = []
    for _, group_df in df_plot_data.group_by("Category", maintain_order=True):
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
            .fill_null(0)
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

    # Create the line plot using Plotly Express.
    fig = px.line(
        combined_resampled_df.to_pandas(),  # Plotly works best with pandas
        x="Timestamp",
        y=primary_metric_col,
        color="Category",
        title=f"Memory Usage Over Time ({primary_metric_col} - Lines) - {data_filepath.stem}<br>Resample: {resample_interval_str}",
        labels={
            "Timestamp": f"Time (Resampled to {resample_interval_str})",
            primary_metric_col: f"Average {primary_metric_col} (KB)",
        },
        markers=True,
    )

    # Add a trace for the total memory usage.
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
    """Generates and saves an interactive stacked area plot using Plotly.

    Args:
        df_plot_data: A Polars DataFrame containing the filtered data to plot.
        primary_metric_col: The name of the column containing memory data.
        resample_interval_str: The Polars interval string for resampling (e.g., '5s').
        data_filepath: The path of the original data file, for titling.
        output_dir: The directory to save the plot file.
    """
    if df_plot_data.is_empty():
        logger.warning(
            f"Stacked Plot: Input data is empty for {data_filepath.name}. Skipping."
        )
        return

    # Resample data to prepare for plotting.
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

    # Create the stacked area plot.
    fig = px.area(
        resampled_df.to_pandas(),
        x="Timestamp",
        y=primary_metric_col,
        color="Category",
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


# --- Main Worker Function ---


def plot_memory_usage_from_data_file(data_filepath: Path, args: argparse.Namespace):
    """Reads a Parquet file and generates detailed time-series plots.

    This is the main worker function for the detailed plot mode. It handles:
    - Reading the Parquet data file.
    - Determining the primary metric from the corresponding summary log.
    - Cleaning and preparing the data (e.g., creating 'Category' column).
    - Applying filters based on command-line arguments (--category, --top-n).
    - Determining the appropriate time resampling interval.
    - Calling the specific plot generation functions.

    Args:
        data_filepath: Path to the input `.parquet` data file.
        args: The parsed command-line arguments.
    """
    output_dir = args.output_dir or args.log_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    primary_metric_col = _get_primary_metric_from_summary_log(data_filepath)
    if not primary_metric_col:
        logger.error(
            f"Could not determine primary metric for {data_filepath.name}. Skipping."
        )
        return

    try:
        df_pl = pl.read_parquet(data_filepath)
        if df_pl.is_empty():
            logger.warning(f"No data found in {data_filepath}. Skipping plot.")
            return

        # Isolate per-process data, which is the basis for the plots.
        df_per_process_pl = df_pl.filter(pl.col("Record_Type") == "PROCESS")
        if df_per_process_pl.is_empty():
            logger.warning(
                f"No per-process data rows in {data_filepath.name}. Skipping."
            )
            return

        # Combine major and minor categories into a single 'Category' column for plotting.
        df_per_process_pl = df_per_process_pl.with_columns(
            (pl.col("Major_Category") + "_" + pl.col("Minor_Category")).alias(
                "Category"
            )
        )

        # Validate that all necessary columns are present.
        required_cols = ["Timestamp_epoch", "Category", primary_metric_col]
        if not all(col in df_per_process_pl.columns for col in required_cols):
            missing = [c for c in required_cols if c not in df_per_process_pl.columns]
            logger.error(
                f"Data in {data_filepath} missing required columns: {missing}."
            )
            return

        # Ensure the metric column is numeric and filter out nulls.
        df_per_process_pl = df_per_process_pl.with_columns(
            pl.col(primary_metric_col).cast(pl.Float64, strict=False)
        ).filter(pl.col(primary_metric_col).is_not_null())

        if df_per_process_pl.is_empty():
            logger.warning(
                f"No valid numeric data for '{primary_metric_col}' in {data_filepath.name}. Skipping."
            )
            return

        # Exclude categories marked as 'Ignored_'.
        df_per_process_pl = df_per_process_pl.filter(
            ~pl.col("Category").str.starts_with("Ignored_")
        )
        if df_per_process_pl.is_empty():
            logger.warning("No data after filtering ignored categories. Skipping.")
            return

        # --- Custom Filtering based on CLI arguments ---
        if args.category:
            logger.info(f"Filtering for user-specified categories: {args.category}")
            df_per_process_pl = df_per_process_pl.filter(
                pl.col("Category").is_in(args.category)
            )
        elif args.top_n:
            logger.info(f"Filtering for top {args.top_n} categories by peak memory.")
            top_categories = (
                df_per_process_pl.group_by("Category")
                .agg(pl.col(primary_metric_col).max().alias("peak_mem"))
                .sort("peak_mem", descending=True)
                .head(args.top_n)["Category"]
                .to_list()
            )
            df_per_process_pl = df_per_process_pl.filter(
                pl.col("Category").is_in(top_categories)
            )
        else:
            # Default behavior: filter for categories that meet a minimum total memory usage.
            category_total_metric = df_per_process_pl.group_by("Category").agg(
                pl.col(primary_metric_col).sum().alias("total_metric")
            )
            significant_categories = category_total_metric.filter(
                pl.col("total_metric") >= MIN_TOTAL_PRIMARY_METRIC_KB_FOR_PLOT
            )["Category"].to_list()
            df_per_process_pl = df_per_process_pl.filter(
                pl.col("Category").is_in(significant_categories)
            )

        if df_per_process_pl.is_empty():
            logger.warning(
                f"No data remains after filtering for {data_filepath.name}. Skipping plots."
            )
            return

        logger.info(
            f"Plotting for categories: {df_per_process_pl['Category'].unique().to_list()}"
        )

        # Convert epoch seconds to a datetime object for time-series plotting.
        df_plot_data_pl = df_per_process_pl.with_columns(
            pl.from_epoch("Timestamp_epoch", time_unit="s").alias("Timestamp")
        )

        # --- Resampling Interval Logic ---
        if args.resample_interval:
            resample_interval_str_polars = args.resample_interval
            logger.info(
                f"Using user-provided resample interval: {resample_interval_str_polars}"
            )
        else:
            # Dynamically determine resampling interval based on data size
            # Use number of data points as a proxy for duration to avoid datetime type issues
            num_samples = len(df_plot_data_pl)
            
            # Heuristic: estimate data collection duration based on sample count
            # Assuming typical monitoring intervals of 1-2 seconds per sample
            if num_samples <= 60:  # ~1 minute of data
                resample_interval_str_polars = "5s"
            elif num_samples <= 300:  # ~5 minutes of data  
                resample_interval_str_polars = "10s"
            elif num_samples <= 900:  # ~15 minutes of data
                resample_interval_str_polars = "30s"
            else:  # Longer builds
                resample_interval_str_polars = "1m"
            
            logger.info(
                f"Data points: {num_samples}. Using resample interval: {resample_interval_str_polars}"
            )

        # --- Plot Generation ---
        if args.chart_type in ["line", "all"]:
            _generate_line_plot_plotly(
                df_plot_data_pl,
                primary_metric_col,
                resample_interval_str_polars,
                data_filepath,
                output_dir,
            )
        if args.chart_type in ["stacked", "all"]:
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


# --- Main CLI Entry Point ---


def main():
    """Main command-line interface function for the plotter tool.

    This function defines and parses all command-line arguments, then
    dispatches to the appropriate handler function based on the arguments
    provided (e.g., summary plot vs. detailed plots).
    """
    parser = argparse.ArgumentParser(
        description="Generate plots from mymonitor data files.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        required=True,
        help="Required. Path to the run-specific log directory containing .parquet and .log files.",
    )
    parser.add_argument(
        "--project-name",
        type=str,
        help="Filter to only generate plots for a specific project name.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        help="Filter to only generate plots for a specific parallelism level (-j N).",
    )
    parser.add_argument(
        "--chart-type",
        choices=["line", "stacked", "all"],
        default="all",
        help="Specify which chart types to generate. Default: all.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory to save plots. Defaults to the specified --log-dir.",
    )
    parser.add_argument(
        "--resample-interval",
        type=str,
        help="Override automatic resampling. Use Polars interval string (e.g., '1s', '10s', '1m').",
    )

    # A mutually exclusive group ensures that the user can only specify one
    # of these filtering methods at a time.
    filter_group = parser.add_mutually_exclusive_group()
    filter_group.add_argument(
        "--category",
        type=str,
        action="append",
        help="Generate plots for only a specific category (e.g., 'Compiler_clang'). Can be specified multiple times.",
    )
    filter_group.add_argument(
        "--top-n",
        type=int,
        metavar="N",
        help="Only plot the top N categories by peak memory usage.",
    )
    parser.add_argument(
        "--summary-plot",
        action="store_true",
        help="Generate a single summary plot comparing build times and peak memory across all parallelism levels in the log directory.",
    )

    args = parser.parse_args()

    # Validate command line arguments
    if not args.log_dir.is_dir():
        logger.error(f"Log directory not found: {args.log_dir}")
        sys.exit(1)
    
    # Additional validation for numeric arguments
    if args.jobs is not None:
        if args.jobs < 1 or args.jobs > 1024:
            logger.error(f"Invalid --jobs value: {args.jobs}. Must be between 1 and 1024.")
            sys.exit(1)
    
    if args.top_n is not None:
        if args.top_n < 1 or args.top_n > 100:
            logger.error(f"Invalid --top-n value: {args.top_n}. Must be between 1 and 100.")
            sys.exit(1)
    
    # Validate project name format if provided
    if args.project_name is not None:
        if not isinstance(args.project_name, str) or not args.project_name.strip():
            logger.error("Project name cannot be empty.")
            sys.exit(1)
        
        # Check for valid characters (basic validation)
        import re
        if not re.match(r'^[a-zA-Z0-9_-]+$', args.project_name):
            logger.error(f"Invalid project name '{args.project_name}'. Only letters, numbers, hyphens, and underscores are allowed.")
            sys.exit(1)
    
    # Validate output directory if provided
    if args.output_dir is not None:
        if not args.output_dir.parent.exists():
            logger.error(f"Parent directory of output directory does not exist: {args.output_dir.parent}")
            sys.exit(1)
        
        # Try to create the output directory if it doesn't exist
        try:
            args.output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.error(f"Cannot create output directory '{args.output_dir}': {e}")
            sys.exit(1)
    
    # Validate resample interval format if provided
    if args.resample_interval is not None:
        # Basic validation for Polars interval format
        import re
        if not re.match(r'^\d+[smhdw]$', args.resample_interval):
            logger.error(f"Invalid resample interval format '{args.resample_interval}'. Expected format: number followed by s/m/h/d/w (e.g., '10s', '1m').")
        sys.exit(1)

    # --- Dispatch to the correct mode ---
    if args.summary_plot:
        generate_run_summary_plot(args)
        return  # Exit after generating the summary plot

    # --- Default Mode: Generate detailed plots ---
    logger.info(f"Searching for Parquet data files in: {args.log_dir}")
    all_parquet_files = list(args.log_dir.glob("*.parquet"))

    if not all_parquet_files:
        logger.info(f"No Parquet data files found in {args.log_dir}.")
        return

    # Filter files based on project name and jobs if specified by the user.
    files_to_plot = []
    for f in all_parquet_files:
        filename = f.name
        if args.project_name and args.project_name not in filename:
            continue
        if args.jobs and f"_j{args.jobs}_" not in filename:
            continue
        files_to_plot.append(f)

    if not files_to_plot:
        logger.warning("No data files matched the specified filters.")
        return

    logger.info(f"Found {len(files_to_plot)} matching data file(s) to process.")
    for data_file in files_to_plot:
        logger.info(f"--- Generating plot for {data_file.name} ---")
        plot_memory_usage_from_data_file(data_file, args)


# Standard Python entry point guard. This allows the script to be imported
# as a module without executing the main function.
if __name__ == "__main__":
    main()
