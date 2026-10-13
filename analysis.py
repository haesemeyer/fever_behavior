"""
Main analysis script for behavioral fever data
"""

import matplotlib as mpl
import matplotlib.pyplot as pl
import argparse
import os
from os import path
from typing import Any, List, Dict, Tuple
import numpy as np
import pandas as pd
import loading
import pre_processing as preproc
import seaborn as sns
import plot_funs as pf
import utils
from scipy.stats import ranksums


class CheckArgs(argparse.Action):
    """
    Check our command line arguments for validity
    """
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        super().__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values: Any, option_string=None):
        if self.dest == 'bf_folder' or self.dest == 'fg_folder':
            if not path.exists(values):
                raise argparse.ArgumentError(self, "Specified directory does not exist")
            if not path.isdir(values):
                raise argparse.ArgumentError(self, "The destination is a file but should be a directory")
            setattr(namespace, self.dest, values)
        else:
            raise Exception("Parser was asked to check unknown argument")


def load_all(exp_list: List[str]) -> Tuple[Dict, Dict, Dict]:
    """
    For list of experiments returns pre-processed fish data and bout data frames
    :param exp_list: List of experimental info file paths
    :return:
        [0]: Dictionary of lists of pre-processed fish dataframes by fish-type
        [1]: Dictionary of lists of extracted bout dataframes by fish-type
        [2]: Dictionary of lists of original filenames by fish-type
    """
    all_fish_data = {}
    all_bout_data = {}
    all_expname_data = {}
    # load
    for e in exp_list:
        fish = loading.load_exp_data_from_nwb(e)
        for k in fish:
            if k in all_fish_data:
                all_fish_data[k] += [fs[0] for fs in fish[k]]
                all_expname_data[k] += [f"{e}&&{fs[1]}" for fs in fish[k]]
            else:
                all_fish_data[k] = [fs[0] for fs in fish[k]]
                all_expname_data[k] = [f"{e}&&{fs[1]}" for fs in fish[k]]
    # pre-process
    for k in all_fish_data:
        for f in all_fish_data[k]:
            preproc.pre_process_fish_data(f)
            if k not in all_bout_data:
                all_bout_data[k] = []
            all_bout_data[k].append(preproc.identify_bouts(f, 100))
    return all_fish_data, all_bout_data, all_expname_data


def get_dcq() -> Dict:
    """
    Parses our lane-augmented experiment filenames and identifies the corresponding dCq in the Q-Dataframe
    :return: n_fish long vector with dCq values in same order as fish and bout data lists
    """
    def lane_in_q(q_fid: float) -> int:
        # obtain lane-data from q-data
        if q_fid - np.round(q_fid) >= 0.1:
            return int(np.round((q_fid - np.round(q_fid)) * 10))  # old style saving
        return int(q_fid)

    delta_cq = {}
    for kk in all_fever_expnames:
        delta_cq[kk] = []
        for ee in all_fever_expnames[kk]:
            fn_augm = path.split(ee)[1]
            sep_ix = fn_augm.find("&&")
            exp_file_name = fn_augm[:sep_ix]
            exp_file_lane = int(fn_augm[sep_ix+2:])
            # get all q-data that could correspond to this experiment
            exp_q_data = all_q_data[all_q_data["BehaviorNWB"] == exp_file_name]
            if exp_q_data.shape[0] == 0:
                delta_cq[kk].append(np.nan)
                continue
            lane_found = False
            for data_line in exp_q_data.iterrows():
                qlane = lane_in_q(data_line[1]["fish"])
                if qlane == exp_file_lane:
                    lane_found = True
                    delta_cq[kk].append(data_line[1]["dCq"])
            if not lane_found:
                delta_cq[kk].append(np.nan)
    return delta_cq


def valid(a: pd.Series) -> np.ndarray:
    a_array = np.array(a)
    val = np.isfinite(a_array)
    return a_array[val]

def co_valid(a: pd.Series, co_a: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    a_array = np.array(a)
    coa_array = np.array(co_a)
    val = np.isfinite(a_array)
    return a_array[val], coa_array[val]


if __name__ == '__main__':
    burn_in = 5*60*100  # 5-minute burn-in period

    mpl.rcParams['pdf.fonttype'] = 42

    a_parser = argparse.ArgumentParser(prog="analysis",
                                       description="Runs analysis for behavioral fever experiments")
    a_parser.add_argument("-bf", "--bf_folder", help="Path to folder with behavioral fever experiments", type=str,
                          default="", action=CheckArgs)
    a_parser.add_argument("-ht", "--hot_temp", help="Warm edge temperature of gradient", type=float, default=30.0)
    a_parser.add_argument("-ct", "--cold_temp", help="Cool edge temperature of gradient", type=float, default=20.0)

    args = a_parser.parse_args()

    fever_folder = args.bf_folder
    hot_temp = args.hot_temp
    cold_temp = args.cold_temp


    hot_zone = 28
    cold_zone = 22

    fever_exps, fever_csv = loading.find_all_exp_paths(fever_folder)

    plot_dir = path.join(fever_folder, "plots")
    if not path.exists(plot_dir):
        os.makedirs(plot_dir)

    # load and process data
    all_fever_fish, all_fever_bouts, all_fever_expnames = load_all(fever_exps)

    # load and concatenate all Q-data
    if len(fever_csv) > 0:
        q_data = [pd.read_csv(csv) for csv in fever_csv]
        if len(q_data) > 1:
            all_q_data = pd.concat(q_data, axis=0)
        else:
            all_q_data = q_data[0]

        # get the delta cqs where available
        all_delta_cq = get_dcq()
    else:
        all_delta_cq = {k: np.full(len(all_fever_fish[k]), np.nan) for k in all_fever_fish}

    # across all experiments, make plot of trajectory as well as histograms along x and y
    # color signifies problematic bout counts:
    # blue: at least one bout every second; orange: at least 0.5 bouts; red: less than 0.5 bouts
    plot_count = 0
    for k in all_fever_fish:
        for f, b, e in zip(all_fever_fish[k], all_fever_bouts[k], all_fever_expnames[k]):
            pf.overview_plot(f, b, e, plot_count, plot_dir, k)
            plot_count += 1

    # create dictionaries of valid experiments - experiments with at least one bout every two seconds
    val_fever_fish, val_fever_bouts, val_fever_expnames, val_delta_cq = {}, {}, {}, {}
    for k in all_fever_fish:
        for f, b, n, q in zip(all_fever_fish[k], all_fever_bouts[k], all_fever_expnames[k], all_delta_cq[k]):
            if b.shape[0] >= f.shape[0]/200:
                if k not in val_fever_fish:
                    val_fever_fish[k] = []
                    val_fever_bouts[k] = []
                    val_fever_expnames[k] = []
                    val_delta_cq[k] = []
                val_fever_fish[k].append(f)
                val_fever_bouts[k].append(b)
                val_fever_expnames[k].append(n)
                val_delta_cq[k].append(q)

    # estimate confines of chamber
    x_min = min([np.nanmin(f["X Position"]) for k in val_fever_fish for f in val_fever_fish[k]])
    x_max = max([np.nanmax(f["X Position"]) for k in val_fever_fish for f in val_fever_fish[k]])
    y_min = min([np.nanmin(f["Y Position"]) for k in val_fever_fish for f in val_fever_fish[k]])
    y_max = max([np.nanmax(f["Y Position"]) for k in val_fever_fish for f in val_fever_fish[k]])
    chamber = utils.ChamberEstimate(x_min, x_max, y_min, y_max)

    border_size = 4  # Border size in 4mm - all data within will be excluded

    # filter bouts
    filtered_bouts = {}
    for k in val_fever_bouts:
        filtered_bouts[k] = []
        for bout, fish in zip(val_fever_bouts[k], val_fever_fish[k]):
            filtered_bouts[k].append(utils.edge_filter_bouts(chamber, border_size, bout, fish))
    val_fever_bouts = filtered_bouts

    try:
        example_fish = val_fever_fish["nac:PBS"][0]
        example_timeframe = slice(25500, 26000)
        time = np.arange(example_fish.shape[0])/100

        # plot example fish positions and gradient direction histogram for example fish across entire experiment
        fig = pl.figure()
        pl.scatter(np.array(example_fish["X Position"])[::10], np.array(example_fish["Y Position"])[::10], s=2, alpha=0.1, color='k')
        pl.plot([0, 50], [y_min, y_min], 'C3')
        pl.plot([0, 50], [y_max, y_max], 'C3')
        pl.ylim(y_min, y_max)
        pl.axis("equal")
        pl.xlabel("X Position [mm]")
        pl.ylabel("Y Position [mm]")
        sns.despine(fig)
        fig.savefig(path.join(plot_dir, "Example_positions.pdf"), dpi=600)

        fig = pl.figure()
        pl.hist(example_fish["Y Position"], np.linspace(y_min, y_max, 21), density=True)
        pl.plot([y_min, y_min], [0, 0.016], 'C3')
        pl.plot([y_max, y_max], [0, 0.016], 'C3')
        pl.xlabel("Y Position [mm]")
        pl.ylabel("Density")
        sns.despine(fig)
        fig.savefig(path.join(plot_dir, "Example_Y_position_histogram.pdf"))

        # plot kinematics and trace for a short timeframe
        fig, axes = pl.subplots(nrows=2, sharex=True)
        axes[0].plot(time[example_timeframe], np.rad2deg(example_fish["Heading"][example_timeframe]))
        axes[0].set_ylabel("Heading [degrees]")
        axes[1].plot(time[example_timeframe], example_fish["Instant speed"][example_timeframe])
        axes[1].set_ylabel("Speed [mm/s]")
        axes[1].set_xlabel("Time [s]")
        sns.despine(fig)
        fig.savefig(path.join(plot_dir, "Example_kinematics.pdf"))

        fig = pl.figure()
        pl.plot(example_fish["X Position"][example_timeframe], example_fish["Y Position"][example_timeframe])
        pl.axis("equal")
        pl.xlabel("X Position [mm]")
        pl.ylabel("Y Position [mm]")
        sns.despine(fig)
        fig.savefig(path.join(plot_dir, "Example_trajectory.pdf"))
    except KeyError:
        pass

    # collect weighted mean gradient temperatures and temperature kde's
    temp_pref = {"Treatment": [], "Avg. temperature [C]": [], "Median temperature [C]": [],
                 "dCq": [], "Variance [C**2]": []}
    temp_distribution = {"Treatment": [], "Density": [], "Cumulative density": []}
    temp_oc_bins = np.linspace(cold_temp, hot_temp, 21)
    temp_oc_bc = temp_oc_bins[:-1] + np.diff(temp_oc_bins)/2

    disp_distribution = {"Treatment": [], "Density": []}
    disp_bins = np.linspace(0, 4, 50)
    disp_bc = disp_bins[:-1] + np.diff(disp_bins)/2

    angle_distribution = {"Treatment": [], "Density": []}
    angle_bins = np.linspace(-90, 90, 50)
    angle_bc = angle_bins[:-1] + np.diff(angle_bins)/2

    ibi_distribution = {"Treatment": [], "Density": []}
    # NOTE: The wonky bincount is because this is a discrete measure (resolution = 10ms). This means that if some
    # bin boundaries exactly line up with discrete steps we can get bins that are enriched/starved of data
    ibi_bins = np.linspace(0, 1500, 51)
    ibi_bc = ibi_bins[:-1] + np.diff(ibi_bins)/2

    # dictionary of lists for bootstrapped ks-test containing raw occupancy temperatures of each fish
    fish_temps_raw = {k: [] for k in val_fever_fish}

    for k in val_fever_fish:
        for df_fish, df_bout, dcq in zip(val_fever_fish[k], val_fever_bouts[k], val_delta_cq[k]):
            temp_pref["Treatment"].append(k)
            temp_distribution["Treatment"].append(k)
            disp_distribution["Treatment"].append(k)
            angle_distribution["Treatment"].append(k)
            ibi_distribution["Treatment"].append(k)
            temps = df_fish['Temperature'][burn_in:]
            fish_temps_raw[k].append(temps)
            temps = temps[np.isfinite(temps)]
            temp_pref["Avg. temperature [C]"].append(np.nanmean(temps))
            temp_pref["Median temperature [C]"].append(np.nanmedian(temps))
            temp_pref["Variance [C**2]"].append(np.nanvar(temps))
            hp = np.sum(temps > hot_zone)
            cp = np.sum(temps < cold_zone)
            temp_pref["dCq"].append(dcq)
            temp_distribution["Density"].append(np.histogram(temps, bins=temp_oc_bins, density=True)[0])
            h = np.histogram(temps, bins=temp_oc_bins, density=False)[0].astype(float)
            h /= h.sum()
            temp_distribution["Cumulative density"].append(np.cumsum(h))
            disp_distribution["Density"].append(np.histogram(df_bout['Displacement'], bins=disp_bins, density=True)[0])
            angle_distribution["Density"].append(np.histogram(np.rad2deg(df_bout['Angle change']), bins=angle_bins,
                                                              density=True)[0])
            ibi_distribution["Density"].append(np.histogram(df_bout["IBI"], bins=ibi_bins, density=True)[0])
    df_temp_pref = pd.DataFrame(temp_pref)
    df_temp_distribution = pd.DataFrame(temp_distribution)
    df_disp_distribution = pd.DataFrame(disp_distribution)
    df_angle_distribution = pd.DataFrame(angle_distribution)
    df_ibi_distribution = pd.DataFrame(ibi_distribution)

    order = sorted(list(val_fever_bouts.keys()))

    fig, (ax_mean, ax_median) = pl.subplots(ncols=2)
    sns.boxplot(data=df_temp_pref, x="Treatment", y="Avg. temperature [C]", ax=ax_mean, order=order, whis=np.inf)
    sns.stripplot(x="Treatment", y="Avg. temperature [C]", data=df_temp_pref, color='k', alpha=0.3, ax=ax_mean,
                  order=order)
    sns.boxplot(data=df_temp_pref, x="Treatment", y="Median temperature [C]", ax=ax_median,
                order=order, whis=np.inf)
    sns.stripplot(x="Treatment", y="Median temperature [C]", data=df_temp_pref, color='k', alpha=0.3, ax=ax_median,
                  order=order)
    sns.despine()
    fig.savefig(path.join(plot_dir, "Temperature_preference.pdf"))

    fig, ax = pl.subplots()
    sns.boxplot(data=df_temp_pref, x="Treatment", y="Variance [C**2]", ax=ax, order=order)
    sns.stripplot(x="Treatment", y="Variance [C**2]", data=df_temp_pref, color='k', alpha=0.3, ax=ax,
                  order=order)
    sns.despine()
    fig.savefig(path.join(plot_dir, "TemperatureVariance.pdf"))

    fig, ax = pl.subplots()
    sns.barplot(data=df_temp_pref, x="Treatment", y="Median temperature [C]", ax=ax, order=order,
                estimator=np.var, errorbar=('ci', 68))
    pl.ylabel("Preference variance [C**2]")
    sns.despine()
    fig.savefig(path.join(plot_dir, "InterFishVariance.pdf"))

    if len(fever_csv) > 0:
        fig, ax = pl.subplots()
        sns.boxplot(data=df_temp_pref, x="Treatment", y="dCq", ax=ax, order=order, whis=np.inf)
        sns.stripplot(data=df_temp_pref, x="Treatment", y="dCq", ax=ax, order=order, color='k')
        pl.ylabel("Isg-15 dCQ")
        pl.yticks([0.0, 0.5, 0.6, 1.0, 2.0, 3.0, 4.0])
        sns.despine()
        fig.savefig(path.join(plot_dir, "dCq.pdf"))

    # Compute p-values of all pairwise conditions, generate and save significance dataframes
    p_avg_list = []
    p_dcq_list = []
    n_avg_list = []
    n_dcq_list = []
    comparisons = []
    for i, k in enumerate(order[:-1]):
        for kk in order[i+1:]:
            grp_1 = df_temp_pref["Treatment"] == k
            grp_2 = df_temp_pref["Treatment"] == kk
            comparisons.append(f"{k} vs. {kk}")
            n_avg_list.append([
                df_temp_pref["Avg. temperature [C]"][grp_1].shape[0],
                df_temp_pref["Avg. temperature [C]"][grp_2].shape[0]
            ])
            p_avg = ranksums(df_temp_pref["Avg. temperature [C]"][grp_1],
                             df_temp_pref["Avg. temperature [C]"][grp_2]).pvalue
            p_avg_list.append(p_avg)
            if len(fever_csv) > 0:
                n_dcq_list.append([
                    valid(df_temp_pref["dCq"][grp_1]).size,
                    valid(df_temp_pref["dCq"][grp_2]).size
                ])
                p_dcq = ranksums(valid(df_temp_pref["dCq"][grp_1]), valid(df_temp_pref["dCq"][grp_2])).pvalue
                p_dcq_list.append(p_dcq)
    df_p_avg_temp = utils.generate_stat_df(np.hstack(p_avg_list), np.vstack(n_avg_list), np.hstack(comparisons))

    df_p_avg_temp.to_csv(path.join(plot_dir, "pvals_avg_temp_comparison.csv"))

    if len(fever_csv) > 0:
        df_p_dcq = utils.generate_stat_df(np.hstack(p_dcq_list), np.vstack(n_dcq_list), np.hstack(comparisons))
        df_p_dcq.to_csv(path.join(plot_dir, "pvals_dcq_comparison.csv"))

    # plot temperature distribution densities with bootstrap statistics
    fig = pf.lineplot(df_temp_distribution, "Density", "Treatment", temp_oc_bc, "Temperature [C]", np.mean)
    fig.savefig(path.join(plot_dir, "Occupancy.pdf"))
    fig = pf.lineplot(df_temp_distribution, "Cumulative density", "Treatment", temp_oc_bc, "Temperature [C]", np.mean)
    fig.savefig(path.join(plot_dir, "Cumulative Occupancy.pdf"))

    # use ks-test bootstrapping to test significance of distribution differences across fish
    p_occupancy_list = []
    n_occupancy_list = []
    comparisons = []
    for i, k in enumerate(order[:-1]):
        for kk in order[i+1:]:
            comparisons.append(f"{k} vs. {kk}")
            n_occupancy_list.append([
                len(fish_temps_raw[k]),
                len(fish_temps_raw[kk])
            ])
            p_occupancy_list.append(utils.ks_bootstrap_test_by_fish(fish_temps_raw[k], fish_temps_raw[kk], 1000)[0])
    df_p_occupancy = utils.generate_stat_df(np.hstack(p_occupancy_list), np.vstack(n_occupancy_list), np.hstack(comparisons))
    df_p_occupancy.to_csv(path.join(plot_dir, "pvals_occupancy.csv"))

    # plot bout displacement distribution densities with bootstrap statistics
    fig = pf.lineplot(df_disp_distribution, "Density", "Treatment", disp_bc, "Displacement [mm]", np.mean)
    fig.savefig(path.join(plot_dir, "Displacement.pdf"))

    # plot bout turn angle distribution densities with bootstrap statistics
    fig = pf.lineplot(df_angle_distribution, "Density", "Treatment", angle_bc, "Turn angle [deg]", np.mean)
    pl.plot([-10, -10], [0, 0.05], 'k--')
    pl.plot([10, 10], [0, 0.05], 'k--')
    fig.savefig(path.join(plot_dir, "TurnAngle.pdf"))

    # plot interbout interval distribution densities with bootstrap statistics
    fig = pf.lineplot(df_ibi_distribution, "Density", "Treatment", ibi_bc, "Interbout interval [ms]", np.mean)
    fig.savefig(path.join(plot_dir, "InterboutInterval.pdf"))

    # analyze bout features by temperature
    temp_bins = np.linspace(cold_temp, hot_temp, 15)
    temp_bincents = temp_bins[:-1] + np.diff(temp_bins) / 2
    temp_disp_distribution = {"Treatment": [], "Displacement [mm]": []}
    temp_ibi_distribution = {"Treatment": [], "Bout interval [ms]": []}
    temp_mag_distribution = {"Treatment": [], "Turn magnitude [deg]": []}

    for k in val_fever_fish:
        for bouts in val_fever_bouts[k]:
            b_displace = bouts["Displacement"]
            b_mag = np.abs(np.rad2deg(bouts["Angle change"]))
            ibi = bouts["IBI"]
            b_temps = bouts["Temperature"]
            temp_disp_distribution["Treatment"].append(k)
            temp_ibi_distribution["Treatment"].append(k)
            temp_mag_distribution["Treatment"].append(k)
            s_hist = np.histogram(b_temps, bins=temp_bins, weights=b_displace)[0].astype(float)
            m_hist = np.histogram(b_temps, bins=temp_bins, weights=b_mag)[0].astype(float)
            i_hist = np.histogram(b_temps, bins=temp_bins, weights=ibi)[0].astype(float)
            n_hist = np.histogram(b_temps, bins=temp_bins)[0].astype(float)
            temp_disp_distribution["Displacement [mm]"].append(s_hist / n_hist)
            temp_mag_distribution["Turn magnitude [deg]"].append(m_hist / n_hist)
            temp_ibi_distribution["Bout interval [ms]"].append(i_hist / n_hist)

    df_temp_disp_distribution = pd.DataFrame(temp_disp_distribution)
    df_temp_ibi_distribution = pd.DataFrame(temp_ibi_distribution)
    df_temp_mag_distribution = pd.DataFrame(temp_mag_distribution)

    fig = pf.lineplot(df_temp_disp_distribution, "Displacement [mm]", "Treatment", temp_bincents, "Temperature [C]",
                      np.nanmean)
    fig.savefig(path.join(plot_dir, "Displacement_by_temperature.pdf"))

    fig = pf.lineplot(df_temp_ibi_distribution, "Bout interval [ms]", "Treatment", temp_bincents, "Temperature [C]",
                      np.nanmean)
    fig.savefig(path.join(plot_dir, "BoutInterval_by_temperature.pdf"))

    fig = pf.lineplot(df_temp_mag_distribution, "Turn magnitude [deg]", "Treatment", temp_bincents, "Temperature [C]",
                      np.nanmean)
    fig.savefig(path.join(plot_dir, "TurnMag_by_temperature.pdf"))

    # compute overall significance of difference in kinematics by temperature by testing significance of mean squared
    # distances between the distributions by temperature
    p_displacement_list = []
    p_ibi_list = []
    p_tmag_list = []
    comparisons = []
    n_values = []
    for i, k in enumerate(order[:-1]):
        for kk in order[i+1:]:
            comparisons.append(f"{k} vs. {kk}")
            n_values.append([
                len(val_fever_fish[k]),
                len(val_fever_fish[kk]),
            ])
            p_displacement_list.append(utils.sq_diff_bs_hyp_test(df_temp_disp_distribution, "Displacement [mm]",
                                                                 "Treatment", k, kk, 1000)[0])
            p_ibi_list.append(utils.sq_diff_bs_hyp_test(df_temp_ibi_distribution, "Bout interval [ms]",
                                                        "Treatment", k, kk, 1000)[0])
            p_tmag_list.append(utils.sq_diff_bs_hyp_test(df_temp_mag_distribution, "Turn magnitude [deg]",
                                                         "Treatment", k, kk, 1000)[0])
    comparisons = np.hstack(comparisons)
    df_p_displacement = utils.generate_stat_df(np.hstack(p_displacement_list), np.vstack(n_values), comparisons)
    df_p_displacement.to_csv(path.join(plot_dir, "pvals_displacement.csv"))
    df_p_ibi = utils.generate_stat_df(np.hstack(p_ibi_list), np.vstack(n_values), comparisons)
    df_p_ibi.to_csv(path.join(plot_dir, "pvals_ibi.csv"))
    df_p_tmag = utils.generate_stat_df(np.hstack(p_tmag_list), np.vstack(n_values), comparisons)
    df_p_tmag.to_csv(path.join(plot_dir, "pvals_tmag.csv"))

    # For following analyses reduce number of bins to reduce noise
    temp_bins = np.linspace(cold_temp, hot_temp, 11)
    temp_bincents = temp_bins[:-1] + np.diff(temp_bins) / 2

    delta_cut = 0.025
    # for each condition, plot experienced bout delta-t
    dt_bins = np.linspace(-0.2, 0.2, 26)
    dt_bc = dt_bins[:-1] + np.diff(dt_bins)/2
    bout_delta_distribution = {"Treatment": [], "Proportion": []}
    for k in val_fever_fish:
        for bouts in val_fever_bouts[k]:
            bout_deltas = bouts["Prev Delta T"]
            bout_delta_distribution["Treatment"].append(k)
            bout_delta_distribution["Proportion"].append(np.histogram(bout_deltas, dt_bins)[0]/bout_deltas.size)
    df_delta_distribution = pd.DataFrame(bout_delta_distribution)
    fig = pf.lineplot(df_delta_distribution, "Proportion", "Treatment", dt_bc,
                      "Prev. swim temperature change [C]", np.nanmean)
    pl.plot([-delta_cut, -delta_cut], [0.01, 0.08], 'k--')
    pl.plot([delta_cut, delta_cut], [0.01, 0.08], 'k--')
    fig.savefig(path.join(plot_dir, "swim_temperature_change.pdf"))

    # for each condition, plot the difference between average turn magnitude after heating vs. cooling bouts
    mag_bias_distribution = {"Treatment": [], "Turn bias": []}
    for k in val_fever_fish:
        for bouts in val_fever_bouts[k]:
            b_temps = bouts["Temperature"]
            prev_dt = bouts["Prev Delta T"]
            b_mags = np.rad2deg(np.abs(bouts["Angle change"]))
            heating = prev_dt > delta_cut
            cooling = prev_dt < -delta_cut
            mag_bias_distribution["Treatment"].append(k)
            tw = np.histogram(b_temps[heating], bins=temp_bins, weights=b_mags[heating])[0].astype(float)
            tc = np.histogram(b_temps[heating], bins=temp_bins)[0]
            heating_mag = tw / tc
            tw = np.histogram(b_temps[cooling], bins=temp_bins, weights=b_mags[cooling])[0].astype(float)
            tc = np.histogram(b_temps[cooling], bins=temp_bins)[0]
            cooling_mag = tw / tc
            mag_bias_distribution["Turn bias"].append(heating_mag - cooling_mag)
    df_mag_bias = pd.DataFrame(mag_bias_distribution)
    fig = pf.lineplot(df_mag_bias, "Turn bias", "Treatment", temp_bincents, "Temperature [C]",
                      np.nanmean, add_marker=True)
    pl.plot([temp_bincents[0], temp_bincents[-1]], [0, 0], 'k--')
    fig.savefig(path.join(plot_dir, "heating_turn_bias.pdf"))

    p_tbias_list = []
    comparisons = []
    n_values = []
    for i, k in enumerate(order[:-1]):
        for kk in order[i+1:]:
            comparisons.append(f"{k} vs. {kk}")
            n_values.append([
                len(val_fever_fish[k]),
                len(val_fever_fish[kk]),
            ])
            p_tbias_list.append(utils.sq_diff_bs_hyp_test(df_mag_bias, "Turn bias",
                                                                 "Treatment", k, kk, 1000)[0])
    comparisons = np.hstack(comparisons)
    df_p_tbias = utils.generate_stat_df(np.hstack(p_tbias_list), np.vstack(n_values), comparisons)
    df_p_tbias.to_csv(path.join(plot_dir, "pvals_turnbias.csv"))

    # augment bouts with reversal information and save fish and bout dataframes
    # temp_aln_distribution = {"Treatment": [], "Gradient alignment": []}
    # for k in val_fever_bouts:
    #     for bout, fish, ename in zip(val_fever_bouts[k], val_fever_fish[k], val_fever_expnames[k]):
    #         bout = utils.augment_state_info(bout, align_thresh, max_reversal_length)
    #         grad_direction = np.array(bout["Gradient direction"])
    #         b_temps = bout["Temperature"]
    #         a_hist = np.histogram(b_temps, bins=temp_bins, weights=np.abs(grad_direction))[0].astype(float)
    #         n_hist = np.histogram(b_temps, bins=temp_bins)[0].astype(float)
    #         temp_aln_distribution["Gradient alignment"].append(a_hist / n_hist)
    #         temp_aln_distribution["Treatment"].append(k)
    #         e_base = path.split(ename)[1]
    #         if not path.exists(path.join(plot_dir, k.replace(':', '_'))):
    #             os.makedirs(path.join(plot_dir, k.replace(':', '_')))
    #         save_base = path.join(path.join(plot_dir, k.replace(':', '_')), e_base)
    #         bout.to_pickle(f"{save_base}_bout.pkl")
    #         fish.to_pickle(f"{save_base}_fish.pkl")
