"""
Supplemental analysis script for behavioral fever data
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
from scipy.stats import ranksums, gaussian_kde

if __name__ == '__main__':
    burn_in = 5*60*100  # 5-minute burn-in period

    mpl.rcParams['pdf.fonttype'] = 42

    a_parser = argparse.ArgumentParser(prog="suppl_analysis",
                                       description="Runs supplemental analysis for behavioral fever experiments")
    a_parser.add_argument("-pf", "--preproc_folder", help="Path to folder with pickle subfolders", type=str,
                          required=True)
    a_parser.add_argument("-ht", "--hot_temp", help="Warm edge temperature of gradient", type=float, default=30.0)
    a_parser.add_argument("-ct", "--cold_temp", help="Cool edge temperature of gradient", type=float, default=20.0)

    args = a_parser.parse_args()

    preproc_root = args.preproc_folder
    hot_temp = args.hot_temp
    cold_temp = args.cold_temp

    plot_dir = path.join(preproc_root, "suppl_plots")
    if not path.exists(plot_dir):
        os.makedirs(plot_dir)

    # find experimental condition subfolders
    subs = os.listdir(preproc_root)
    conditions = [s for s in subs if path.isdir(path.join(preproc_root, s)) and "plot" not in s]
    order = sorted(conditions)

    all_fish = {cond: utils.load_all_fish_data(path.join(preproc_root, cond)) for cond in conditions}

    # generate plots of individual fish gradient densities
    temp_oc_bc = np.linspace(cold_temp, hot_temp, 51)
    bw = (180000/1000)**(-1/4)  # behavior likely independent over 10s interval

    occupancies = {cond: [] for cond in conditions}
    for cond in order:
        fig = pl.figure()
        avg = np.zeros(temp_oc_bc.size)
        for fish in all_fish[cond]:
            temperatures = np.array(fish["Temperature"])[np.isfinite(np.array(fish["Temperature"]))]
            kde = gaussian_kde(temperatures, bw_method=bw)
            kde_vals = kde.evaluate(temp_oc_bc)
            avg += kde_vals
            occupancies[cond].append(kde_vals)
            pl.plot(temp_oc_bc, kde_vals, color='k')
        pl.plot(temp_oc_bc, avg / len(all_fish[cond]), "C1")
        pl.xlim(20, 30)
        pl.ylim(0, 0.7)
        pl.xticks([20, 22, 24, 26, 28, 28, 30])
        pl.xlabel("Temperature [C]")
        pl.ylabel("Kernel density")
        sns.despine(fig)
        fig.savefig(path.join(plot_dir, cond+"_ind_occupancy_kde.pdf"))

    # for each fish plot the per-minute displacement across experimental time and experiment average
    fish_minute_disp = {"Treatment": [], "Value": []}
    test_times = np.arange(30)*100*60
    test_times = test_times[test_times >= burn_in]
    test_times = test_times.astype(int)
    for cond in order:
        fig = pl.figure()
        avg_disps = np.zeros(test_times.size)
        disp_counts = np.zeros(test_times.size)
        for fish in all_fish[cond]:
            xp = np.array(fish["X Position"])
            yp = np.array(fish["Y Position"])
            tt_matched = test_times[test_times+60*100-1 < xp.size]
            disps = np.array([np.sqrt((xp[tt+60*100-1]-xp[tt])**2 + (yp[tt+60*100-1]-yp[tt])**2) for tt in tt_matched])
            fish_minute_disp["Treatment"].append(cond)
            fish_minute_disp["Value"].append(np.nanmean(disps))
            d_to_add = disps
            cnt_to_add = np.ones(disps.size)
            d_to_add[np.isnan(disps)] = 0
            cnt_to_add[np.isnan(disps)] = 0
            avg_disps[:disps.size] += d_to_add
            disp_counts[:disps.size] += cnt_to_add
            pl.scatter(tt_matched/(100*60), disps, s=2, alpha=0.25, color='k')
        pl.plot(test_times/(100*60), avg_disps/disp_counts, color='k')
        pl.xlim(4, 30)
        pl.ylim(0, 200)
        pl.xticks([5, 10, 15, 20, 25, 30])
        pl.xlabel("Time [minutes]")
        pl.ylabel("1 minute displacement [mm]")
        sns.despine()
        fig.savefig(path.join(plot_dir, cond + "_ind_minute_displace_by_time.pdf"))

    df_minute_disp = pd.DataFrame(fish_minute_disp)
    fig, ax = pl.subplots()
    sns.boxplot(data=df_minute_disp, x="Treatment", y="Value", ax=ax, order=order, whis=np.inf)
    sns.stripplot(data=df_minute_disp, x="Treatment", y="Value", ax=ax, order=order, color='k')
    pl.ylim(0, 100)
    pl.ylabel("1 minute displacement [mm]")
    sns.despine()
    fig.savefig(path.join(plot_dir, "One_minute_displace_averages.pdf"))

    # compute and save statistics on per-minute-displacements
    p_mdisp_list = []
    comparisons = []
    n_values = []
    for i, k in enumerate(order[:-1]):
        for kk in order[i+1:]:
            grp_1 = df_minute_disp["Treatment"] == k
            grp_2 = df_minute_disp["Treatment"] == kk
            comparisons.append(f"{k} vs. {kk}")
            n_values.append(np.array([df_minute_disp["Value"][grp_1].shape[0],
                                      df_minute_disp["Value"][grp_2].shape[0]]))
            p = ranksums(df_minute_disp["Value"][grp_1],
                             df_minute_disp["Value"][grp_2]).pvalue

            p_mdisp_list.append(p)
    df_p_min_disp = utils.generate_stat_df(np.hstack(p_mdisp_list), np.vstack(n_values), np.hstack(comparisons))

    df_p_min_disp.to_csv(path.join(plot_dir, "pvals_per_minute_displacement_comparison.csv"))
