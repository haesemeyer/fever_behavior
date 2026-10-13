"""
Routines for experimental data processing to call swim bouts etc.
"""

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from typing import Tuple


def average_angle(angles: np.ndarray) -> np.ndarray:
    """
    Computes an average angle through decomposition
    :param angles: The individual angles to average in radians
    :return: The average angle
    """
    x, y = np.cos(angles), np.sin(angles)
    return np.arctan2(np.mean(y), np.mean(x))


def arc_distance(angles: np.ndarray) -> np.ndarray:
    """
    Computes the length of the arcs between consecutive angles on the unit circle
    :param angles: n_frames long vector of the angles in radians between which to compute the arc distances
    :return: n_frames-1 long vector of arc distances
    """
    x, y = np.cos(angles), np.sin(angles)
    d = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    theta = np.arccos(1 - d**2 / 2)
    return theta  # since on the unit circle the arc-distance is equal to the angular difference


def fix_angle_trace(angles: np.ndarray, ad_thresh: float, max_ahead: int) -> np.ndarray:
    """
    Tries to fix sudden jumps in angle traces (angle differences that are implausible) by filling stretches between
    with the pre-jump angle
    :param angles: The angle trace to fix
    :param ad_thresh: The largest plausible delta-angle (arc distance on unit circle, i.e. smallest angular difference)
    :param max_ahead: The maximal amount of frames to look ahead for a jump back (longer stretches won't be fixed)
    :return: The corrected angle trace
    """
    adists = np.r_[0, arc_distance(angles)]
    angles_corr = np.full(angles.size, np.nan)
    index = 0
    while index < adists.size:
        ad = adists[index]
        if np.isnan(ad):
            angles_corr[index] = angles[index]
            index += 1
            continue
        if ad < ad_thresh:
            angles_corr[index] = angles[index]
            index += 1
        else:
            # start correction loop
            next_jump_ix = index + 1
            for i in range(max_ahead):
                if i + next_jump_ix >= adists.size:
                    # nothing we can do here, just set this one to NaN by not filling it and continue
                    break
                if adists[i + next_jump_ix] >= ad_thresh:
                    # we found a similar jump within the next ten frames fill with initial angle to correct
                    replace_angle = angles_corr[index - 1]
                    assert np.isfinite(replace_angle)
                    angles_corr[index:i + next_jump_ix + 1] = replace_angle
                    index = next_jump_ix + i
                    assert np.isfinite(angles_corr[index])
                    break
            index += 1
    return angles_corr


def pre_process_fish_data(fish_data: pd.DataFrame, frame_rate=100) -> None:
    """
    Performs filtering and interpolation on fish data for smoothening and small gap filling and adds
    fish speeds and tail vigor to the data. Converts coordinates to mm and expresses speed in mm/s
    :param fish_data: raw Fish DataFrame returned by load_exp_data_by_info
    :param px_per_mm: The spatial resolution of the acquisition in pixels per mm
    :param frame_rate: The temporal resolution of the acquisition in Hz
    """
    # Median filter position data in order to remove mis-tracks then
    # filter position data, setting sigma to 1/3 of expected swim bout length at 100 Hz
    fish_data["Raw X"] = fish_data["X Position"].copy()
    fish_data["Raw Y"] = fish_data["Y Position"].copy()
    xpos = np.array(fish_data["X Position"])
    ypos = np.array(fish_data["Y Position"])
    med_thresh = (250.0/frame_rate)**2
    for i in range(2, xpos.size-2):
        if np.sum(np.isnan(xpos[i-2:i+3])) > 2:
            continue
        # don't median filter unless there are unexpectedly large jumps - fish should never swim faster
        # than 25 cm/s, i.e. 250 mm/s
        if np.max(np.diff(xpos[i-2:i+3])**2 + np.diff(ypos[i-2:i+3])**2) < med_thresh:
            continue
        xpos[i] = np.nanmedian(xpos[i-2:i+3])
        ypos[i] = np.nanmedian(ypos[i-2:i+3])
    fish_data["X Position"] = xpos
    fish_data["Y Position"] = ypos
    fish_data["X Position"] = gaussian_filter1d(fish_data["X Position"], sigma=2)
    fish_data["Y Position"] = gaussian_filter1d(fish_data["Y Position"], sigma=2)
    # Compute instantaneous speed
    i_speed = np.sqrt(np.diff(fish_data["X Position"])**2 + np.diff(fish_data["Y Position"])**2)
    i_speed = np.r_[0, i_speed] * frame_rate
    fish_data["Instant speed"] = i_speed
    # filter out improbably heading angle jumps - 1 radians was determined as the cut-off
    # through an arc_distance histogram
    fish_data["Heading"] = fix_angle_trace(fish_data["Heading"], 1, 10)


def find_bout_start_end_by_peak(instant_speed: np.ndarray, spd_thresh: float, pk_width: int,
                                delta_thresh: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Finds peaks in the speed trace and from there extends bouts forwards and backwards while the speed is decreasing
    :param instant_speed: The instand speed trace in which to identify bouts
    :param spd_thresh: The minimal peak height value in mm/s
    :param pk_width: The minimal peakd width in frames
    :param delta_thresh: From the peak bouts are extended while the speed is at least dropping by delta_thresh
    :return:
        [0]: n_bouts long vector of starts
        [1]: n_bouts long vector of ends
    """
    # First identify peaks
    peak_indices = find_peaks(instant_speed, height=spd_thresh, width=pk_width)[0]
    n_bouts = peak_indices.size
    if n_bouts == 0:
        return np.array([]), np.array([])
    starts = np.zeros(n_bouts, int) - 1
    ends = starts.copy()
    # From each peak walk forwards and backwards - stop walking if the average drop in speed across the next
    # five frames is smaller than the threshold
    for i, pi in enumerate(peak_indices):
        s = pi
        while True:
            delta = (instant_speed[s] - instant_speed[s-5]) / 5
            s -= 1
            if np.isnan(delta):
                s = -1
                break
            if delta < delta_thresh and instant_speed[s] < spd_thresh:
                break
        e = pi
        while True:
            if e + 5 >= instant_speed.size:
                break
            delta = (instant_speed[e] - instant_speed[e + 5]) / 5
            e += 1
            if np.isnan(delta):
                e = -1
                break
            if delta < delta_thresh and instant_speed[e] < spd_thresh:
                break
        starts[i] = s
        ends[i] = e
    # identify cases where the start of the next bout has been placed before the end of the previous bout (negative IBI)
    ibi = np.r_[0, starts[1:] - ends[:-1]]
    ix_neg = np.where(ibi <= 0)[0]
    for ix in ix_neg:
        starts[ix] = ends[ix-1] + 1  # set the start of this bout to one frame after the previous bout ends
    # remove any pairs that have a -1 as this indicates that no valid value was found, also remove pairs in which the
    # length is now below 0 due to the fix above
    valid = np.logical_and(starts != -1, ends != -1)
    valid = np.logical_and(valid, (ends-starts) > 0)
    return starts[valid], ends[valid]



def identify_bouts(fish_data: pd.DataFrame, frame_rate=100) -> pd.DataFrame:
    """
    Uses instant speed to identify swim bouts and returns dataframe with bout characteristics
    :param fish_data: Fish data after pre-processing
    :param frame_rate: Acquisition frame-rate in Hz
    :return: Dataframe with bout data (start, stop, peak speed, displacement, angle-change, IBI, Temperature at bout,
     Previous bout delta T, Previous second delta T)
    """
    bout_columns = ["Original index", "Start", "Stop", "Peak speed", "Displacement", "Angle change", "IBI",
                    "Temperature", "Prev Delta T", "1s Delta T", "Delta X", "Delta Y", "Prev angle change",
                    "X Position", "Y Position", "Heading"]
    spd_trace = np.array(fish_data["Instant speed"])
    start_frames, stop_frames = find_bout_start_end_by_peak(spd_trace, 5.0, 5, 0.1)
    if start_frames.size == 0:
        return pd.DataFrame(columns=bout_columns)
    # above_th = (spd_trace > spd_thresh).astype(float)
    # starts_stops = np.r_[0, np.diff(above_th)]
    # start_frames = np.where(starts_stops > 0)[0]
    # stop_frames = np.where(starts_stops < 0)[0]
    # if start_frames.size == 0 or stop_frames.size == 0:
    #     return pd.DataFrame(columns=bout_columns)
    # if stop_frames.size > start_frames.size or stop_frames[0] < start_frames[0]:
    #     stop_frames = stop_frames[1:]  # this can only occur if the speed is high to begin with which should not happen
    # if start_frames.size > stop_frames.size:
    #     start_frames = start_frames[:stop_frames.size]
    bout_lengths = stop_frames - start_frames + 1
    valid = np.logical_and(bout_lengths >= 8, bout_lengths <= 50)  # at least 80 ms maximally 500 ms
    valid = np.logical_and(valid, stop_frames < spd_trace.size-1)  # if a bout ends right at experiment end remove
    start_frames = start_frames[valid]
    stop_frames = stop_frames[valid]
    if start_frames.size == 0:
        return pd.DataFrame(columns=bout_columns)
    p_speeds = np.full(start_frames.size, np.nan)
    displace = p_speeds.copy()
    delta_x = p_speeds.copy()
    delta_y = p_speeds.copy()
    angle_change = p_speeds.copy()
    ibi = p_speeds.copy()
    prev_ang_change = p_speeds.copy()
    temperature = p_speeds.copy()  # temperature at bout start
    prev_delta_t = p_speeds.copy()  # previous bout delta-temperature
    prev_sec_delta_t = p_speeds.copy()  # previous second delta-temperature
    x_position = p_speeds.copy()
    y_position = p_speeds.copy()
    start_heading = p_speeds.copy()
    for i, (s, e) in enumerate(zip(start_frames, stop_frames)):
        p_speeds[i] = np.max(fish_data["Instant speed"][s:e+1])
        pre_start = s-5 if s >= 5 else 0
        post_end = e+5 if (e+5) < spd_trace.size else spd_trace.size-1
        pre_slice = slice(pre_start, s)
        post_slice = slice(e+1, post_end)
        pre_x = np.mean(fish_data["X Position"][pre_slice])
        pre_y = np.mean(fish_data["Y Position"][pre_slice])
        x_position[i] = pre_x
        y_position[i] = pre_y
        post_x = np.mean(fish_data["X Position"][post_slice])
        post_y = np.mean(fish_data["Y Position"][post_slice])
        displace[i] = np.sqrt((post_x - pre_x)**2 + (post_y - pre_y)**2)
        delta_x[i] = post_x - pre_x
        delta_y[i] = post_y - pre_y
        pre_angle = average_angle(fish_data["Heading"][pre_slice])
        post_angle = average_angle(fish_data["Heading"][post_slice])
        start_heading[i] = pre_angle
        d_angle = post_angle - pre_angle
        if d_angle > np.pi:
            angle_change[i] = d_angle - 2*np.pi
        elif d_angle < -np.pi:
            angle_change[i] = d_angle + 2*np.pi
        else:
            angle_change[i] = d_angle
        temperature[i] = np.mean(fish_data["Temperature"][pre_slice])
        if s > frame_rate:
            prev_sec_delta_t[i] = temperature[i] - fish_data["Temperature"][s-frame_rate]
        if i > 0:
            prev_delta_t[i] = temperature[i] - temperature[i-1]
    # interbout intervals in ms
    ibi[1:] = (start_frames[1:] - stop_frames[:-1])/frame_rate*1000
    prev_ang_change[1:] = angle_change[:-1]
    original_index = np.arange(ibi.size).astype(int)
    # limit bouts to those that have no NaN within their frames
    val_bouts = np.logical_and(np.isfinite(p_speeds), np.isfinite(displace))
    val_bouts = np.logical_and(val_bouts, np.isfinite(angle_change))
    val_bouts = np.logical_and(val_bouts, np.isfinite(ibi))
    val_bouts = np.logical_and(val_bouts, np.isfinite(prev_ang_change))
    return pd.DataFrame(np.c_[original_index,
                              start_frames,
                              stop_frames,
                              p_speeds,
                              displace,
                              angle_change,
                              ibi,
                              temperature,
                              prev_delta_t,
                              prev_sec_delta_t,
                              delta_x,
                              delta_y,
                              prev_ang_change,
                              x_position,
                              y_position,
                              start_heading],
                        columns=bout_columns)[val_bouts]


if __name__ == '__main__':
    pass
