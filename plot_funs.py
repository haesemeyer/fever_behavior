import matplotlib.pyplot as pl
import numpy as np
import seaborn as sns
import pandas as pd
from os import path
import utils
from typing import Optional, Dict


def overview_plot(fish: pd.DataFrame, bouts: pd.DataFrame, ename: str, save_count: int, save_folder: str,
                  treatment: str):
    plot_col = 'C0'
    if fish.shape[0] / 100 > bouts.shape[0] >= fish.shape[0] / 100 / 2:
        plot_col = 'C1'
    elif bouts.shape[0] < fish.shape[0] / 100 / 2:
        plot_col = 'C3'
    j_grid = sns.jointplot(data=fish, x="X Position", y="Y Position", s=2, alpha=0.05, color=plot_col)
    pl.title(f"{path.split(ename)[1]} - {treatment}")
    j_grid.figure.tight_layout()
    j_grid.figure.savefig(path.join(save_folder, f"Overview_{save_count}.png"), dpi=450)
    pl.close(j_grid.figure)


def lineplot(data: pd.DataFrame, y: str, hue: Optional[str], x: np.ndarray, x_name: str, boot_fun: np.mean, nboot=1000,
             line_args: Optional[Dict] = None, shade_args: Optional[Dict] = None, add_marker=False) -> pl.Figure:
    """
    Seaborn style lineplot function that does not require x-values to be within the dataframe
    :param data: Dataframe with the plot-data
    :param y: Name of the column with y-values within the dataframe
    :param hue: Name of the column to split observations
    :param x: vector of x-values with same length as y-values
    :param x_name: The label of the x-axis
    :param boot_fun: The function to use for bootstrapping
    :param nboot: The number of bootstrap samples to generate
    :param line_args: Additional keyword arguments for the boot-average lineplot
    :param shade_args: Additional keyword arguments for the boot-error shading
    :param add_marker: If set each bin center will be marked
    :return: The generated figure object
    """
    if shade_args is None:
        shade_args = {}
    if line_args is None:
        line_args = {}
    if 'alpha' not in shade_args:
        shade_args['alpha'] = 0.4
    if hue is None:
        hues = [None]
    else:
        hues = np.unique(data[hue])
    fig = pl.figure()
    for i, h in enumerate(hues):
        if h is None:
            values = np.vstack(data[y])
        else:
            values = np.vstack(data[y][data[hue] == h])
        boot_variate = utils.bootstrap(values, nboot, boot_fun)
        m = np.mean(boot_variate, axis=0)
        e = np.std(boot_variate, axis=0)
        if 'color' not in shade_args:
            pl.fill_between(x, m - e, m + e, color=f"C{i}", **shade_args)
        else:
            pl.fill_between(x, m - e, m + e, **shade_args)
        if 'color' not in line_args:
            pl.plot(x, m, f"C{i}", label=h, marker='.' if add_marker else "None", **line_args)
        else:
            pl.plot(x, m, label=h, marker='.' if add_marker else "None", **line_args)
    pl.xlabel(x_name)
    pl.ylabel(f"{y} +/- se")
    pl.legend()
    sns.despine()
    return fig


def set_journal_style(plot_width_mm=30, plot_height_mm=30, margin_mm=10):
    """
    Set Matplotlib style for journal publication with:
    - Only x and y axes visible.
    - A legend without a bounding box and with elements having only a fill (no stroke).
    - An actual plot area (excluding labels) of at least `plot_width_mm` × `plot_height_mm`.

    Parameters:
    - plot_width_mm (float): Minimum plot area width in mm (default: 30 mm).
    - plot_height_mm (float): Minimum plot area height in mm (default: 30 mm).
    - margin_mm (float): Extra margin for labels and titles (default: 10 mm).
    """
    # Convert mm to inches (1 inch = 25.4 mm)
    fig_width_in = (plot_width_mm + 2 * margin_mm) / 25.4
    fig_height_in = (plot_height_mm + 2 * margin_mm) / 25.4

    pl.rcParams.update({
        'font.size': 7,
        'font.family': 'Arial',
        'axes.titlesize': 7,
        'axes.labelsize': 7,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'lines.linewidth': 0.5,
        'axes.linewidth': 0.5,
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'xtick.minor.width': 0.5,
        'ytick.minor.width': 0.5,
        'xtick.major.size': 1.984,  # 0.7 mm tick length
        'ytick.major.size': 1.984,
        'xtick.minor.size': 1.984,
        'ytick.minor.size': 1.984,
        'savefig.dpi': 300,  # High resolution
        'figure.figsize': (fig_width_in, fig_height_in),
        'savefig.transparent': True,  # transparent background
        'figure.constrained_layout.use': True  # Ensure proper layout
    })


# Function to remove the top and right spines
def remove_spines(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.spines["left"].set_linewidth(0.5)


# Function to format the legend (removes bounding box and legend stroke)
def format_legend(ax):
    legend = ax.get_legend()
    if legend:
        legend.get_frame().set_linewidth(0)  # Remove bounding box
        legend.get_frame().set_facecolor("none")  # Transparent background
        legend.get_frame().set_edgecolor("none")  # No border

        # Modify legend elements to remove stroke but keep fill
        for handle in legend.legendHandles:
            try:
                handle.set_edgecolor("none")  # Remove stroke
            except AttributeError:
                pass  # Some elements may not support this, ignore errors


if __name__ == '__main__':
    pass
