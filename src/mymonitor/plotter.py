import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def plot_memory_usage_from_csv(csv_filepath: Path, output_dir: Path):
    """
    Reads memory usage data from a CSV file and generates a time-series plot.
    """
    if not csv_filepath.exists():
        logger.error(f"CSV file not found: {csv_filepath}")
        return

    try:
        # Read the CSV, treating lines starting with '#' as comments
        df = pd.read_csv(csv_filepath, comment='#')
        logger.info(f"Successfully read CSV: {csv_filepath}")
        logger.debug(f"DataFrame head:\n{df.head()}")

        if df.empty:
            logger.warning(f"No data found in {csv_filepath} after skipping comments. Skipping plot.")
            return

        # Ensure required columns exist
        required_cols = ['Timestamp_epoch', 'Category', 'RSS_KB']
        if not all(col in df.columns for col in required_cols):
            logger.error(f"CSV file {csv_filepath} is missing one or more required columns: {required_cols}. Found: {df.columns.tolist()}")
            return

        # Convert Timestamp_epoch to datetime objects
        df['Timestamp'] = pd.to_datetime(df['Timestamp_epoch'], unit='s')

        # Filter out ignored categories if any (e.g., 'ignore_vscode_server')
        df = df[~df['Category'].str.startswith('ignore_')]

        if df.empty:
            logger.warning(f"No data left after filtering ignored categories in {csv_filepath}. Skipping plot.")
            return

        # --- Plotting ---
        fig, ax = plt.subplots(figsize=(15, 8))

        # Plot RSS usage for each category over time
        for category, group_data in df.groupby('Category'):
            ax.plot(group_data['Timestamp'], group_data['RSS_KB'], label=category, marker='.', linestyle='-')
        
        ax.set_xlabel("Time")
        ax.set_ylabel("RSS Memory Usage (KB)")
        ax.set_title(f"Memory Usage Over Time - {csv_filepath.stem}")
        
        # Format x-axis to show time nicely
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        fig.autofmt_xdate() # Auto-rotate date labels

        ax.legend(loc='upper left', bbox_to_anchor=(1, 1)) # Legend outside plot area
        ax.grid(True)
        
        plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend

        plot_filename = output_dir / f"{csv_filepath.stem}_memory_plot.png"
        plt.savefig(plot_filename)
        logger.info(f"Memory usage plot saved to: {plot_filename}")
        plt.close(fig) # Close the figure to free memory

    except pd.errors.EmptyDataError:
        logger.warning(f"No data or columns to parse in {csv_filepath}. Skipping plot.")
    except Exception as e:
        logger.error(f"Error generating plot for {csv_filepath}: {e}", exc_info=True)

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