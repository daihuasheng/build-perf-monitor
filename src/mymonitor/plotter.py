import logging
import polars as pl
import plotly.express as px
import plotly.graph_objects as go
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

MIN_TOTAL_PRIMARY_METRIC_KB_FOR_PLOT = 10240  # 10MB, will apply to the primary metric


def _get_primary_metric_from_summary_log(
    data_filepath: Path,
) -> Optional[str]:  # Renamed arg
    """
    Parses the corresponding _summary.log file to find the primary metric used.

    Args:
        data_filepath: Path to the main data file (CSV or Parquet). The function will look
                      for a summary log file with a similar name.

    Returns:
        The primary metric string (e.g., "RSS_KB", "PSS_KB") if found,
        otherwise "RSS_KB" as a fallback or None if the input is a summary log.
    """
    summary_log_filename = data_filepath.stem + "_summary.log"
    if data_filepath.name.endswith("_summary.log"):
        return None  # Do not process summary logs themselves for a primary metric
    summary_log_path = data_filepath.parent / f"{data_filepath.stem}_summary.log"
    if not summary_log_path.exists():
        logger.warning(
            f"Could not find summary log: {summary_log_path} to determine primary metric. Falling back to RSS_KB."
        )
        return "RSS_KB"
    try:
        with open(summary_log_path, "r") as f:
            for line in f:
                match_peak = re.search(r"Peak Overall Memory \(([^)]+)\):", line)
                if match_peak:
                    metric = match_peak.group(1)
                    logger.info(
                        f"Determined primary metric '{metric}' from summary log (peak line)."
                    )
                    return metric
                match_collector = re.search(r"Memory Metric: (\w+)", line)
                if match_collector:
                    collector_type_short = match_collector.group(1).lower()
                    if "pss" in collector_type_short:
                        logger.info(
                            f"Determined primary metric 'PSS_KB' from summary log (collector type line: {collector_type_short})."
                        )
                        return "PSS_KB"
                    elif "rss" in collector_type_short:
                        logger.info(
                            f"Determined primary metric 'RSS_KB' from summary log (collector type line: {collector_type_short})."
                        )
                        return "RSS_KB"
    except Exception as e:
        logger.error(
            f"Error reading or parsing summary log {summary_log_path}: {e}. Falling back to RSS_KB."
        )
    return "RSS_KB"  # Default fallback


def _save_plotly_figure(
    fig: go.Figure,
    base_filename: str,
    output_dir: Path,
):
    """
    Saves a Plotly figure to both HTML and PNG formats.

    Args:
        fig: The Plotly figure object to save.
        base_filename: The base name for the output files (without extension).
        output_dir: The directory to save the files in.
    """
    plot_filename_html = output_dir / f"{base_filename}.html"
    try:
        fig.write_html(plot_filename_html)
        logger.info(
            f"Interactive plot saved to: {plot_filename_html}"
        )
        # If you need PNG and have kaleido installed:
        try:
            plot_filename_png = output_dir / f"{base_filename}.png"
            fig.write_image(plot_filename_png, width=1200, height=600)
            logger.info(f"Static plot saved to: {plot_filename_png}")
        except Exception as e_kaleido:
            logger.warning(
                f"Failed to save static plot to PNG (Kaleido might be missing or misconfigured): {e_kaleido}. "
                f"To enable PNG export, install Kaleido: pip install kaleido or mymonitor[export]"
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
    of these averages across categories is also added.

    Args:
        df_plot_data: Polars DataFrame containing the data to plot.
                      Expected columns: "Timestamp", "Category", and primary_metric_col.
        primary_metric_col: Name of the column containing the primary metric data.
        resample_interval_str: Polars interval string (e.g., "1s", "5m") for resampling.
        data_filepath: Path to the original CSV or Parquet file (used for naming the output).
        output_dir: Directory where the generated HTML plot file will be saved.
    """
    if df_plot_data.is_empty():
        logger.warning(
            f"Line Plot: Input data is empty for {data_filepath.name}. Skipping."
        )
        return

    # Resample data for each category
    resampled_dfs_list = []
    for category_name_tuple, group_df in df_plot_data.group_by(
        "Category", maintain_order=True
    ):
        category_name = category_name_tuple[0]  # group_by returns a tuple for the key
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
            .fill_null(0)  # Fill NaNs that can result from resampling sparse data
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

    fig = px.line(
        combined_resampled_df.to_pandas(),  # Plotly Express often prefers Pandas DataFrame
        x="Timestamp",
        y=primary_metric_col,
        color="Category",  # Creates different lines for each category
        title=f"Memory Usage Over Time ({primary_metric_col} - Lines) - {data_filepath.stem}<br>Resample: {resample_interval_str}",
        labels={
            "Timestamp": f"Time (Resampled to {resample_interval_str})",
            primary_metric_col: f"Average {primary_metric_col} (KB)",
        },
        markers=True,
    )

    # Calculate and add Total line (sum of the *means* of categories at each resampled point)
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
                line=dict(color="black", dash="dash"),
            )
        )

    fig.update_layout(
        legend_title_text="Category",
        xaxis_title=f"Time (Resampled to {resample_interval_str} intervals)",
        yaxis_title=f"Average {primary_metric_col} Memory Usage (KB)",
    )

    # MODIFIED: Use the new helper function to save the plot
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
    over time, after resampling to the specified interval.

    Args:
        df_plot_data: Polars DataFrame containing the data to plot.
                      Expected columns: "Timestamp", "Category", and primary_metric_col.
        primary_metric_col: Name of the column containing the primary metric data.
        resample_interval_str: Polars interval string (e.g., "1s", "5m") for resampling.
        data_filepath: Path to the original CSV or Parquet file (used for naming the output).
        output_dir: Directory where the generated HTML plot file will be saved.
    """
    if df_plot_data.is_empty():
        logger.warning(
            f"Stacked Plot: Input data is empty for {data_filepath.name}. Skipping."
        )
        return

    # Resample data, taking the mean for each category within each interval
    resampled_df = (
        df_plot_data.sort("Timestamp")
        .group_by_dynamic(
            index_column="Timestamp",
            every=resample_interval_str,
            group_by="Category",  # Group by category for separate lines/areas before stacking
        )
        .agg(pl.col(primary_metric_col).mean().fill_null(0).alias(primary_metric_col))
    )

    if resampled_df.is_empty():
        logger.warning(
            f"Stacked Plot: Resampled data is empty for {data_filepath.name}. Skipping."
        )
        return

    fig = px.area(
        resampled_df.to_pandas(),  # Plotly Express often prefers Pandas DataFrame
        x="Timestamp",
        y=primary_metric_col,
        color="Category",  # Stacks by this column
        title=f"Memory Usage Over Time ({primary_metric_col} - Stacked Area) - {data_filepath.stem}<br>Resample: {resample_interval_str}",
        labels={
            "Timestamp": f"Time (Resampled to {resample_interval_str})",
            primary_metric_col: f"{primary_metric_col} (KB)",
        },
        # groupnorm='percent' # Uncomment for 100% stacked area chart
    )

    fig.update_layout(
        legend_title_text="Category",
        xaxis_title=f"Time (Resampled to {resample_interval_str} intervals)",
        yaxis_title=f"Total {primary_metric_col} Memory Usage (KB) - Stacked",
    )

    # MODIFIED: Use the new helper function to save the plot
    base_plot_filename = f"{data_filepath.stem}_{primary_metric_col}_stacked_plot"
    _save_plotly_figure(fig, base_plot_filename, output_dir)


def plot_memory_usage_from_data_file(data_filepath: Path, output_dir: Path):  # RENAMED
    """
    Reads memory usage data from a Parquet file and generates time-series plots.

    This function processes a Parquet file containing memory metrics, determines the
    primary metric to plot (e.g., RSS_KB, PSS_KB) by consulting an associated
    summary log file, filters and preprocesses the data, and then generates
    both line and stacked area plots of memory usage over time.

    Args:
        data_filepath: Path to the input Parquet file.
        output_dir: Directory where the generated HTML plot files will be saved.
    """
    if not data_filepath.exists():
        logger.error(f"Data file not found: {data_filepath}")  # CHANGED
        return

    primary_metric_col = _get_primary_metric_from_summary_log(data_filepath)
    if not primary_metric_col:
        logger.error(
            f"Could not determine primary metric for {data_filepath.name}. Skipping plot generation."
        )
        return

    try:
        # Read Parquet file with Polars
        df_pl = pl.read_parquet(data_filepath)  # CHANGED
        logger.info(
            f"Successfully read Parquet with Polars: {data_filepath} with {df_pl.height} rows."  # CHANGED
        )

        if df_pl.is_empty():
            logger.warning(
                f"No data found in {data_filepath} after Polars read. Skipping plot."
            )
            return

        # Filter for per-process data rows using the 'Record_Type' column
        df_per_process_pl = df_pl.filter(pl.col("Record_Type") == "PROCESS")  # CHANGED

        if df_per_process_pl.is_empty():
            logger.warning(
                f"No per-process data rows (Record_Type='PROCESS') in {data_filepath.name} (Polars). Skipping."
            )
            return

        logger.info(f"Found {df_per_process_pl.height} per-process data rows (Polars).")

        # Create the 'Category' column for plotting from Major_Category and Minor_Category
        # This 'Category' column will be used by subsequent plotting logic
        df_per_process_pl = df_per_process_pl.with_columns(
            (pl.col("Major_Category") + "_" + pl.col("Minor_Category")).alias(
                "Category"
            )
        )

        required_cols = ["Timestamp_epoch", "Category", primary_metric_col]
        if not all(col in df_per_process_pl.columns for col in required_cols):
            missing_cols = [
                col for col in required_cols if col not in df_per_process_pl.columns
            ]
            logger.error(
                f"Per-process data in {data_filepath} missing required columns for plotting: {missing_cols}. Available: {df_per_process_pl.columns}"
            )
            return

        # Ensure primary_metric_col is numeric, coercing errors to null, then filter out nulls.
        # Parquet read might already give correct types if schema was good during write.
        df_per_process_pl = df_per_process_pl.with_columns(
            pl.col(primary_metric_col).cast(pl.Float64, strict=False)
        ).filter(pl.col(primary_metric_col).is_not_null())

        if df_per_process_pl.is_empty():
            logger.warning(
                f"No valid numeric data for '{primary_metric_col}' in {data_filepath.name} (Polars). Skipping."
            )
            return

        # Reclassify categories based on a predefined map (operates on the new 'Category' column)
        reclassification_map = {
            # Example: "Scripting_Python" -> "script_python" if needed,
            # or map specific Major_Minor combinations.
            # For now, assume the map keys are designed for Major_Minor.
        }
        df_per_process_pl = df_per_process_pl.with_columns(
            pl.col("Category")
            .replace(reclassification_map, default=pl.col("Category"))
            .alias("Category")
        )

        # Filter out categories that are marked to be ignored (operates on the new 'Category' column)
        df_per_process_pl = df_per_process_pl.filter(
            ~pl.col("Category").str.starts_with("ignore_")
        )
        if df_per_process_pl.is_empty():
            logger.warning(
                f"No data after filtering ignored categories (Polars). Skipping."
            )
            return

        # Filter categories based on a minimum total sum of the primary metric.
        # This helps in focusing plots on categories with significant memory usage.
        category_total_metric_pl = df_per_process_pl.group_by("Category").agg(
            pl.col(primary_metric_col).sum().alias("total_metric")
        )
        significant_categories_df = category_total_metric_pl.filter(
            pl.col("total_metric") >= MIN_TOTAL_PRIMARY_METRIC_KB_FOR_PLOT
        )
        significant_categories = significant_categories_df["Category"].to_list()

        if not significant_categories:
            logger.warning(
                f"No significant categories found (Polars) for {data_filepath.name} with metric {primary_metric_col} (threshold: {MIN_TOTAL_PRIMARY_METRIC_KB_FOR_PLOT} KB). Skipping plots."
            )
            return

        df_plot_data_pl = df_per_process_pl.filter(
            pl.col("Category").is_in(significant_categories)
        )
        logger.info(
            f"Plotting for significant categories in {data_filepath.name} (Polars, metric: {primary_metric_col}): {significant_categories}"
        )

        # Convert 'Timestamp_epoch' (seconds since epoch) to Polars Datetime type.
        df_plot_data_pl = df_plot_data_pl.with_columns(
            pl.from_epoch("Timestamp_epoch", time_unit="s").alias("Timestamp")
        )

        if (
            df_plot_data_pl.is_empty()
            or df_plot_data_pl.select(pl.col("Timestamp").count().alias("c"))["c"][0]
            < 1
        ):
            logger.warning(
                f"Not enough data points in {data_filepath.name} after filtering (Polars). Skipping."
            )
            return

        min_time_dt: Optional[datetime] = df_plot_data_pl["Timestamp"].min()
        max_time_dt: Optional[datetime] = df_plot_data_pl["Timestamp"].max()

        duration_seconds = 0.0
        if (
            min_time_dt and max_time_dt and df_plot_data_pl.height > 1
        ):  # Need at least two points to have a duration
            duration_seconds = (max_time_dt - min_time_dt).total_seconds()

        # Determine resample interval string for Polars based on total duration.
        # This aims to create a reasonable number of points on the plot.
        if duration_seconds <= 10:
            resample_interval_str_polars = "1s"
        elif duration_seconds < 60:  # up to 1 min
            resample_interval_str_polars = "5s"
        elif duration_seconds < 300:  # up to 5 mins
            resample_interval_str_polars = "10s"
        elif duration_seconds < 900:  # up to 15 mins
            resample_interval_str_polars = "30s"
        elif duration_seconds < 3600:  # up to 1 hour
            resample_interval_str_polars = "1m"
        elif duration_seconds < 3 * 3600:  # up to 3 hours
            resample_interval_str_polars = "2m"
        else:  # more than 3 hours
            resample_interval_str_polars = "5m"
        logger.info(
            f"Data duration for {data_filepath.name}: {duration_seconds:.0f}s. Polars resample interval: {resample_interval_str_polars}"
        )

        _generate_line_plot_plotly(
            df_plot_data_pl,
            primary_metric_col,
            resample_interval_str_polars,
            data_filepath,  # CHANGED from csv_filepath
            output_dir,
        )
        _generate_stacked_area_plot_plotly(
            df_plot_data_pl,
            primary_metric_col,
            resample_interval_str_polars,
            data_filepath,  # CHANGED from csv_filepath
            output_dir,
        )

    except pl.exceptions.NoDataError:
        logger.warning(
            f"No data in {data_filepath} (Polars NoDataError). Skipping plot."
        )
    except Exception as e:
        logger.error(
            f"Error generating plots for {data_filepath} with Polars/Plotly: {e}",  # CHANGED
            exc_info=True,
        )


def generate_plots_for_logs(log_dir: Path):
    """
    Generates plots for all relevant .parquet files in the specified log directory.

    It iterates through CSV files, skipping summary log files, and calls
    `plot_memory_usage_from_csv` for each data CSV.

    Args:
        log_dir: The directory containing the CSV log files.
    """
    logger.info(
        f"Searching for Parquet data files in: {log_dir} to generate plots."
    )  # CHANGED
    parquet_files = list(log_dir.glob("*.parquet"))  # CHANGED

    if not parquet_files:
        logger.info(
            f"No Parquet data files found in {log_dir} to generate plots for."
        )  # CHANGED
        return

    for data_file in parquet_files:  # CHANGED
        # Summary logs are separate and not Parquet, so no need to filter them here.
        logger.info(
            f"--- Generating plot for {data_file.name} (using Polars & Plotly) ---"
        )
        plot_memory_usage_from_data_file(data_file, log_dir)  # RENAMED and CHANGED
