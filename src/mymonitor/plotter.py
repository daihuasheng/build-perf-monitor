import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import logging
from datetime import timedelta

logger = logging.getLogger(__name__)

MIN_TOTAL_RSS_KB_FOR_PLOT = 10240  # 10MB

def _generate_line_plot(df: pd.DataFrame, resample_interval_str: str, csv_filepath: Path, output_dir: Path):
    """Generates and saves a line plot for memory usage."""
    fig, ax = plt.subplots(figsize=(18, 10)) # Increased size
    all_categories_empty_after_resample = True

    for category, group_data in df.groupby('Category'):
        if group_data.empty:
            continue

        group_data_indexed = group_data.set_index('Timestamp')
        resampled_rss = group_data_indexed['RSS_KB'].resample(resample_interval_str).mean()
        resampled_rss = resampled_rss.ffill(limit=2) 

        if not resampled_rss.empty and not resampled_rss.isnull().all():
            all_categories_empty_after_resample = False
            ax.plot(resampled_rss.index, resampled_rss.values, label=category, marker='.', linestyle='-', markersize=5, linewidth=1.5)
    
    if all_categories_empty_after_resample:
        logger.warning(f"Line Plot: All categories in {csv_filepath.name} resulted in empty or all-NaN data after resampling to {resample_interval_str}. Skipping line plot.")
        plt.close(fig)
        return

    ax.set_xlabel(f"Time (Resampled to {resample_interval_str} intervals)")
    ax.set_ylabel("Average RSS Memory Usage (KB)")
    ax.set_title(f"Memory Usage Over Time (Lines) - {csv_filepath.stem}\nResample: {resample_interval_str}", fontsize=14)
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    fig.autofmt_xdate(rotation=30, ha='right') 

    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0., fontsize='small')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.subplots_adjust(left=0.08, right=0.78, top=0.92, bottom=0.15) # Adjust layout
    plot_filename = output_dir / f"{csv_filepath.stem}_memory_lines_plot.png"
    plt.savefig(plot_filename, bbox_inches='tight')
    logger.info(f"Memory usage line plot saved to: {plot_filename}")
    plt.close(fig)

def _generate_stacked_area_plot(df: pd.DataFrame, resample_interval_str: str, csv_filepath: Path, output_dir: Path):
    """Generates and saves a stacked area plot for memory usage."""
    if df.empty:
        logger.warning(f"Stacked Plot: No data to plot for {csv_filepath.name}. Skipping stacked plot.")
        return

    # Pivot table: Timestamp as index, Category as columns, mean RSS_KB as values
    # Group by resampled Timestamp and Category, then take the mean, then unstack.
    pivot_df = df.groupby([
        pd.Grouper(key='Timestamp', freq=resample_interval_str), 
        'Category'
    ])['RSS_KB'].mean().unstack(fill_value=0)

    if pivot_df.empty or pivot_df.shape[1] == 0:
        logger.warning(f"Stacked Plot: Pivoted data is empty for {csv_filepath.name} after resampling to {resample_interval_str}. Skipping stacked plot.")
        return
    
    # Ensure all values are non-negative for stackplot
    pivot_df[pivot_df < 0] = 0

    fig, ax = plt.subplots(figsize=(18, 10)) # Increased size
    
    try:
        ax.stackplot(pivot_df.index, pivot_df.T.values, labels=pivot_df.columns.tolist(), alpha=0.8)
    except Exception as e:
        logger.error(f"Error during stackplot generation for {csv_filepath.name}: {e}. Data shape: {pivot_df.shape}", exc_info=True)
        plt.close(fig)
        return

    ax.set_xlabel(f"Time (Resampled to {resample_interval_str} intervals)")
    ax.set_ylabel("Total RSS Memory Usage (KB) - Stacked")
    ax.set_title(f"Memory Usage Over Time (Stacked Area) - {csv_filepath.stem}\nResample: {resample_interval_str}", fontsize=14)
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    fig.autofmt_xdate(rotation=30, ha='right')

    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0., fontsize='small')
    ax.grid(True, linestyle=':', alpha=0.5)
    
    plt.subplots_adjust(left=0.08, right=0.78, top=0.92, bottom=0.15) # Adjust layout
    plot_filename = output_dir / f"{csv_filepath.stem}_memory_stacked_plot.png"
    plt.savefig(plot_filename, bbox_inches='tight')
    logger.info(f"Memory usage stacked area plot saved to: {plot_filename}")
    plt.close(fig)


def plot_memory_usage_from_csv(csv_filepath: Path, output_dir: Path):
    """
    Reads memory usage data from a CSV file and generates time-series plots.
    """
    if not csv_filepath.exists():
        logger.error(f"CSV file not found: {csv_filepath}")
        return

    try:
        df = pd.read_csv(csv_filepath, comment='#')
        logger.info(f"Successfully read CSV: {csv_filepath}")

        if df.empty:
            logger.warning(f"No data found in {csv_filepath} (it might be empty or only comments). Skipping plot.")
            return

        required_cols = ['Timestamp_epoch', 'Category', 'RSS_KB']
        if not all(col in df.columns for col in required_cols):
            logger.error(f"CSV file {csv_filepath} is missing one or more required columns: {required_cols}. Found: {df.columns.tolist()}")
            return

        # 1. Reclassify old categories (for compatibility with older CSVs)
        reclassification_map = {
            "py_decodetree": "script_python",
            "py_qapi_gen": "script_python"
            # Add more if other old categories need remapping in the future
        }
        df['Category'] = df['Category'].replace(reclassification_map)
        
        # Filter out 'ignore_*' categories (e.g., vscode_server)
        df = df[~df['Category'].str.startswith('ignore_')]
        if df.empty:
            logger.warning(f"No data left after filtering ignored categories in {csv_filepath}. Skipping plot.")
            return

        # 2. Filter categories with total RSS < MIN_TOTAL_RSS_KB_FOR_PLOT
        category_total_rss = df.groupby('Category')['RSS_KB'].sum()
        significant_categories = category_total_rss[category_total_rss >= MIN_TOTAL_RSS_KB_FOR_PLOT].index.tolist()
        
        if not significant_categories:
            logger.warning(f"No categories with total RSS >= {MIN_TOTAL_RSS_KB_FOR_PLOT}KB found in {csv_filepath.name}. Skipping plots.")
            return
            
        df = df[df['Category'].isin(significant_categories)]
        logger.info(f"Plotting for significant categories in {csv_filepath.name}: {significant_categories}")


        df['Timestamp'] = pd.to_datetime(df['Timestamp_epoch'], unit='s')

        if df.empty or len(df['Timestamp']) < 1: # Need at least one point
            logger.warning(f"Not enough data points in {csv_filepath.name} after filtering to determine duration or plot. Skipping.")
            return
        
        # 3. Dynamic Resampling Interval
        min_time = df['Timestamp'].min()
        max_time = df['Timestamp'].max()
        duration_seconds = (max_time - min_time).total_seconds() if pd.notna(min_time) and pd.notna(max_time) and len(df['Timestamp']) > 1 else 0
        
        if duration_seconds <= 10: # Very short or single point
            resample_interval_str = '1S' # Smallest practical interval
        elif duration_seconds < 60: # Less than 1 minute
            resample_interval_str = '5S'
        elif duration_seconds < 300:  # Less than 5 minutes
            resample_interval_str = '10S'
        elif duration_seconds < 900: # Less than 15 minutes
            resample_interval_str = '30S'
        elif duration_seconds < 3600: # Less than 1 hour
            resample_interval_str = '1T' 
        elif duration_seconds < 3 * 3600: # Less than 3 hours
            resample_interval_str = '2T' 
        else: # Longer than 3 hours
            resample_interval_str = '5T'
        logger.info(f"Data duration for {csv_filepath.name}: {duration_seconds:.0f}s. Dynamic resample interval: {resample_interval_str}")

        # Generate Line Plot
        _generate_line_plot(df.copy(), resample_interval_str, csv_filepath, output_dir)

        # Generate Stacked Area Plot
        _generate_stacked_area_plot(df.copy(), resample_interval_str, csv_filepath, output_dir)

    except pd.errors.EmptyDataError:
        logger.warning(f"No data or columns to parse in {csv_filepath} (Pandas EmptyDataError). Skipping plot.")
    except Exception as e:
        logger.error(f"Error generating plots for {csv_filepath}: {e}", exc_info=True)

def generate_plots_for_logs(log_dir: Path):
    """
    Generates plots for all .csv files in the specified log directory.
    """
    output_plot_dir = log_dir / "plots"
    output_plot_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Searching for CSV log files in: {log_dir}")
    csv_files = list(log_dir.glob("*.csv"))

    if not csv_files:
        logger.info("No CSV log files found to generate plots for.")
        return

    for csv_file in csv_files:
        logger.info(f"--- Generating plot for {csv_file.name} ---")
        plot_memory_usage_from_csv(csv_file, output_plot_dir)