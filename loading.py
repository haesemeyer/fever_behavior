"""
Routines for data loading and file identification
"""

from typing import List, Dict, Tuple
import os
from os import path

import numpy as np
import pandas as pd
from pynwb import NWBHDF5IO


def find_all_exp_paths(root_f: str) -> Tuple[List[str], List[str]]:
    """
    Recursively searches for experiments, identified by the presence of .nwb files and Q-Results
    identified by the presence of .csv files
    :param root_f: The folder from which to start the current search
    :return:
        [0]: List of all .nwb files at and below root_f with their full path
        [1]: List of all .csv files at and below root_f with their full path
    """
    try:
        objects = os.listdir(root_f)
    except PermissionError:
        return [], []
    exp_local = [path.join(root_f, o) for o in objects if ".nwb" in o and ".pkl" not in o]
    csv_local = [path.join(root_f, o) for o in objects if ".csv" in o]
    dir_local = [path.join(root_f, o) for o in objects if path.isdir(path.join(root_f, o)) and "plot" not in o]
    exp_deep = []
    csv_deep = []
    for dl in dir_local:
        e_deep, c_deep = find_all_exp_paths(dl)
        exp_deep += e_deep
        csv_deep += c_deep
    return exp_local + exp_deep, csv_local + csv_deep


def load_exp_data_from_nwb(nwb_file_path: str) -> Dict:
    """
    Loads all relevant experimental data from nwb files
    :param nwb_file_path: Full path to the experiments nwb file
    :return: Dictionary with fish-type as key and list of tuples with fish dataframe and left(1) right(2) lane as values
    """
    result_dict = {}
    df_left = None
    df_right = None
    with NWBHDF5IO(nwb_file_path, "r") as io:
        nwbf = io.read()
        if "Left fish" in nwbf.scratch:
            left_fish = nwbf.scratch['Left fish'].data
            # create dataframe for left fish
            t = nwbf.processing[f"{left_fish}: behavior left lane"]["Temperature"]["Temperatures"].data[()]
            p = nwbf.processing[f"{left_fish}: behavior left lane"]["Position"]["Chamber position"].data[()]
            px = p[:, 0]
            py = p[:, 1]
            h = nwbf.processing[f"{left_fish}: behavior left lane"]["Heading"]["Fish heading"].data[()]
            df_left = pd.DataFrame(np.c_[t, px, py, h], columns=["Temperature", "X Position", "Y Position", "Heading"])

        if "Right fish" in nwbf.scratch:
            right_fish = nwbf.scratch['Right fish'].data
            # create dataframe for right fish
            t = nwbf.processing[f"{right_fish}: behavior right lane"]["Temperature"]["Temperatures"].data[()]
            p = nwbf.processing[f"{right_fish}: behavior right lane"]["Position"]["Chamber position"].data[()]
            px = p[:, 0]
            py = p[:, 1]
            h = nwbf.processing[f"{right_fish}: behavior right lane"]["Heading"]["Fish heading"].data[()]
            df_right = pd.DataFrame(np.c_[t, px, py, h], columns=["Temperature", "X Position", "Y Position", "Heading"])

    # add to and return result dictionary
    if df_left is not None:
        result_dict[left_fish] = [(df_left, 1)]
    if df_right is not None:
        if right_fish in result_dict:
            # fish in right lane is of the same type
            result_dict[right_fish].append((df_right, 2))
        else:
            # fish in right lane has different type than fish in left lane
            result_dict[right_fish] = [(df_right, 2)]
    return result_dict


if __name__ == '__main__':
    pass
