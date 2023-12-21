#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os
import json
import sys
import pandas as pd
from package import import_data, dtw_detection, seg_detection, plot_stepdetection, quality, download, compute_semio_val, radar_design

# if you need to access a file next to the source code, use the variable ROOT
ROOT = os.path.dirname(os.path.realpath(__file__))

# Save the current CWD
data_WD = os.getcwd()

# Change the CWD to ROOT
os.chdir(ROOT)
            

def print_seg_detection(seg_lim, freq):
    """Dump the phase segmentation computed from the trial. 

    Parameters
    ----------
        seg_lim {dataframe} -- pandas dataframe with phases events 
    """

    seg_lim_dict = {'Start': int(seg_lim[0]),
                    'U-Turn start': int(seg_lim[1]),
                    'U-Turn end': int(seg_lim[2]),
                    'End': int(seg_lim[3])}

    display_dict = {'Start_title': "Trial start",
                    'Start': "{Start}".format(**seg_lim_dict),
                    'Start_sec': "{}".format(round(seg_lim_dict['Start']/freq, 2)),
                    'U-Turn start_title': "U-turn start",
                    'U-Turn start': "{U-Turn start}".format(**seg_lim_dict),
                    'U-Turn start_sec': "{}".format(round(seg_lim_dict['U-Turn start']/freq, 2)),
                    'U-Turn end_title': "U-turn end",
                    'U-Turn end': "{U-Turn end}".format(**seg_lim_dict),
                    'U-Turn end_sec': "{}".format(round(seg_lim_dict['U-Turn end']/freq, 2)),
                    'End_title': "Trial end",
                    'End': "{End}".format(**seg_lim_dict), 
                    'End_sec': "{}".format(round(seg_lim_dict['End']/freq, 2))}
        
    info_msg = """
    {Start_title:<15}| {Start:<10}| {Start_sec:<10}
    {U-Turn start_title:<15}| {U-Turn start:<10}| {U-Turn start_sec:<10}
    {U-Turn end_title:<15}| {U-Turn end:<10}| {U-Turn end_sec:<10}
    {End_title:<15}| {End:<10}| {End_sec:<10}
    """

    # Dump information
    os.chdir(data_WD) # Get back to the normal WD

    with open("seg_lim.txt", "wt") as f:
        print(info_msg.format(**display_dict), file=f)
        

def print_steps_detection(seg_lim_full, seg_lim_corrected, steps_lim_full, steps_lim_corrected, freq):
    """Dump the trial parameters computed from the gait events detection.  

    Parameters
    ----------
        steps_lim_corrected {dataframe} -- pandas dataframe with gait events after elimination of the extra trial steps
    """

    steps_dict = {"TrialDuration": (seg_lim_full[3] - seg_lim_full[0])/freq, 
                  "LeftGaitCycles": len(steps_lim_full[steps_lim_full["Foot"]==0]), 
                  "RightGaitCycles": len(steps_lim_full[steps_lim_full["Foot"]==1]), 
                  "WalkingSpeed": (seg_lim_corrected[3] - seg_lim_corrected[0] - seg_lim_corrected[2] + seg_lim_corrected[1])/freq, 
                  "LeftGaitCyclesOk": len(steps_lim_corrected[(steps_lim_corrected["Foot"]==0) & (steps_lim_corrected["Correct"]==1)]), 
                  "RightGaitCyclesOk": len(steps_lim_corrected[(steps_lim_corrected["Foot"]==1) & (steps_lim_corrected["Correct"]==1)])
                 }

    display_dict = {'Raw': "Raw data",
                    'Corrected': "Corrected data",
                    'Number': "Number of footsteps:",
                    'TrialDuration': "Trial duration (s): {TrialDuration}".format(**steps_dict),
                    'LeftGaitCycles': "    - Left foot: {LeftGaitCycles}".format(**steps_dict),
                    'RightGaitCycles': "    - Right foot: {RightGaitCycles}".format(**steps_dict),
                    'WalkingSpeed': "WalkingSpeed (m/s): {WalkingSpeed}".format(**steps_dict),
                    'LeftGaitCyclesOk': '    - Left foot: {LeftGaitCyclesOk}'.format(**steps_dict),
                    'RightGaitCyclesOk': '    - Right foot: {RightGaitCyclesOk}'.format(**steps_dict)
                    }
    info_msg = """
    {Raw:^30}|{Corrected:^30}
    ------------------------------+------------------------------
    {TrialDuration:<30}| {WalkingSpeed:<30}
    {Number:<30}| Number of validated footsteps:
    {LeftGaitCycles:<30}| {LeftGaitCyclesOk:<30}
    {RightGaitCycles:<30}| {RightGaitCyclesOk:<30}
    """

    # dump information
    os.chdir(data_WD) # Get back to the normal WD

    with open("gait_events.txt", "wt") as f:
        print(info_msg.format(**display_dict), file=f)
                

def print_semio_parameters(parameters_dict):
    """Dump the parameters computed from the trial in a text file (trial_info.txt)

    Parameters
    ----------
    parameters_dict : dict
        Parameters of the trial.
    """

    display_dict = {'V': "V: {V}".format(**parameters_dict),
                    'StrT': "StrT: {StrT}".format(**parameters_dict),
                    'UTurn': "UTurn: {UTurn}".format(**parameters_dict),
                    'SteL': "SteL: {SteL}".format(**parameters_dict),
                    'SPARC_gyr': "SteL: {SteL}".format(**parameters_dict),
                    'LDLJAcc': "LDLJAcc: {LDLJAcc}".format(**parameters_dict),
                    'CVStrT': "CVStrT: {CVStrT}".format(**parameters_dict),
                    'CVdsT': "CVdsT: {CVdsT}".format(**parameters_dict),
                    'P1_aCC': "P1_aCC: {P1_aCC}".format(**parameters_dict),
                    'P2_aCC': "P2_aCC: {P2_aCC}".format(**parameters_dict),
                    'ML_RMS': "ML_RMS: {ML_RMS}".format(**parameters_dict),
                    'P1P2': "P1P2: {P1P2}".format(**parameters_dict),
                    'MSwTR': "MSwTR: {MSwTR}".format(**parameters_dict),
                    'AP_iHR': "AP_iHR: {AP_iHR}".format(**parameters_dict),
                    'ML_iHR': "ML_iHR: {ML_iHR}".format(**parameters_dict),
                    'CC_iHR': "CC_iHR: {CC_iHR}".format(**parameters_dict),
                    'dstT': "dstT: {dstT}".format(**parameters_dict), 
                    'sd_V': "{sd_V}".format(**parameters_dict),
                    'sd_StrT': "{sd_StrT}".format(**parameters_dict),
                    'sd_UTurn': "{sd_UTurn}".format(**parameters_dict),
                    'sd_SteL': "{sd_SteL}".format(**parameters_dict),
                    'sd_SPARC_gyr': "{sd_SteL}".format(**parameters_dict),
                    'sd_LDLJAcc': "{sd_LDLJAcc}".format(**parameters_dict),
                    'sd_CVStrT': "{sd_CVStrT}".format(**parameters_dict),
                    'sd_CVdsT': "{sd_CVdsT}".format(**parameters_dict),
                    'sd_P1_aCC': "{sd_P1_aCC}".format(**parameters_dict),
                    'sd_P2_aCC': "{sd_P2_aCC}".format(**parameters_dict),
                    'sd_ML_RMS': "{sd_ML_RMS}".format(**parameters_dict),
                    'sd_P1P2': "{sd_P1P2}".format(**parameters_dict),
                    'sd_MSwTR': "{sd_MSwTR}".format(**parameters_dict),
                    'sd_AP_iHR': "{sd_AP_iHR}".format(**parameters_dict),
                    'sd_ML_iHR': "{sd_ML_iHR}".format(**parameters_dict),
                    'sd_CC_iHR': "{sd_CC_iHR}".format(**parameters_dict),
                    'sd_dstT': "{sd_dstT}".format(**parameters_dict)
                    
                    }
    info_msg = """
    Values                        | Z-Scores
    ------------------------------+------------------------------
    {V:<30}| {sd_V:<30}
    {StrT:<30}| {sd_StrT:<30}
    {UTurn:<30}| {sd_UTurn:<30}
    {SteL:<30}| {sd_SteL:<30}
    {SPARC_gyr:<30}| {sd_SPARC_gyr:<30}
    {LDLJAcc:<30}| {sd_LDLJAcc:<30}
    {CVStrT:<30}| {sd_CVStrT:<30}
    {CVdsT:<30}| {sd_CVdsT:<30}
    {P1_aCC:<30}| {sd_P1_aCC:<30}
    {P2_aCC:<30}| {sd_P2_aCC:<30}
    {ML_RMS:<30}| {sd_ML_RMS:<30}
    {P1P2:<30}| {sd_P1P2:<30}
    {MSwTR:<30}| {sd_MSwTR:<30}
    {AP_iHR:<30}| {sd_AP_iHR:<30}
    {ML_iHR:<30}| {sd_ML_iHR:<30}
    {CC_iHR:<30}| {sd_CC_iHR:<30}
    """

    # Dump information
    os.chdir(data_WD) # Get back to the normal WD

    with open("trial_parameters.txt", "wt") as f:
        print(info_msg.format(**display_dict), file=f)
        

def print_semio_criteria(criteria_dict):
    """Dump the parameters computed from the trial in a text file (trial_info.txt)

    Parameters
    ----------
    parameters_dict : dict
        Parameters of the trial.
    """

    display_dict = {'Average Speed': "Average Speed: {Average Speed}".format(**criteria_dict),
                    'Springiness': "Springiness: {Springiness}".format(**criteria_dict),
                    'Sturdiness': "Sturdiness: {Sturdiness}".format(**criteria_dict),
                    'Smoothness': "Smoothness: {Smoothness}".format(**criteria_dict),
                    'Steadiness': "Steadiness: {Steadiness}".format(**criteria_dict),
                    'Stability': "Stability: {Stability}".format(**criteria_dict),
                    'Symmetry': "Symmetry: {Symmetry}".format(**criteria_dict),
                    'Synchronisation': "Synchronisation: {Synchronisation}".format(**criteria_dict)
                    }
    info_msg = """
    Z-Scores
    --------------------------------------------------+--------------------------------------------------
    {Average Speed:<50}| {Steadiness:<50}
    {Springiness:<50}| {Stability:<50}
    {Sturdiness:<50}| {Symmetry:<50}
    {Smoothness:<50}| {Synchronisation:<50}
    """

    # Dump information
    os.chdir(data_WD) # Get back to the normal WD

    with open("trial_criteria.txt", "wt") as f:
        print(info_msg.format(**display_dict), file=f)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description='Return a semiogram for a given trial.')
    parser.add_argument('-i0', metavar='data_lb', help='Time series for the lower back sensor.')
    parser.add_argument('-i1', metavar='data_rf', help='Time series for the right foot sensor.')
    parser.add_argument('-i2', metavar='data_lf', help='Time series for the left foot sensor.')
    
    parser.add_argument('-freq', metavar='freq',
                        help='Acquistion frequency.')
    parser.add_argument('-age', metavar='age', type=int,
                        help='Age of the subject.')
    parser.add_argument('-min_z', metavar='min_z', type=int,
                        help='Minimum for Z-score.')
    parser.add_argument('-max_z', metavar='max_z', type=int,
                        help='Maximum for Z-score.')
    args = parser.parse_args()
    print("args", args)

    freq = int(args.freq)
    #age = args.age
    age = None 

    # quality index tab
    q1 = [0, 0]  # protocol observation quality [only one u-turn, no steps outside the limits]
    q2 = [0, 0]  # intrinsic detection quality [autocorrelation coefficient, DTW coefficient]
    q3 = [0]  # extrinsic detection quality [right-left step alternation]
    
    # load data
    data_lb = import_data.import_XSens(os.path.join(data_WD, args.i0), freq)
    data_rf = import_data.import_XSens(os.path.join(data_WD, args.i1), freq)
    data_lf = import_data.import_XSens(os.path.join(data_WD, args.i2), freq)
    
    # gait events and steps detection
    steps_lim_full, q2 = dtw_detection.steps_detection_full(data_rf, data_lf, freq, output=data_WD)
    
    # phase boundaries detection and figure
    seg_lim_full, regression, q1[0] = seg_detection.seg_detection(data_lb, steps_lim_full, freq)

    # quality index and 
    steps_lim_corrected, seg_lim_corrected, q1[1], q3 = quality.correct_detection(steps_lim_full, seg_lim_full)
    quality.print_all_quality_index(q1, q2, q3, output=data_WD)

    # print phases and figure
    print_seg_detection(seg_lim_corrected, freq)
    seg_detection.plot_seg_detection(seg_lim_corrected, data_lb, regression, freq, output=data_WD)

    # print validated gait events and figure 
    print_steps_detection(seg_lim_full, seg_lim_corrected, steps_lim_full, steps_lim_corrected, freq)
    plot_stepdetection.plot_stepdetection(steps_lim_corrected, data_rf, data_lf, seg_lim_corrected, freq, output=data_WD)
    plot_stepdetection.plot_stepdetection_construction(steps_lim_corrected, data_rf, data_lf, freq, output=data_WD, corrected=True)

    # load file to be download
    download.json_report(seg_lim_corrected, steps_lim_corrected, freq, output=data_WD)

    # compute semio values (dictionnaries)
    models_folder = os.path.join(ROOT, "models")
    seg_lim_corrected = pd.DataFrame(seg_lim_corrected)
    criteria_names, criteria, parameters = compute_semio_val.compute_semio_val(age, steps_lim_corrected, seg_lim_corrected, data_lb, freq, models_folder=models_folder)
    
    # print semiogram values
    parameters_names = ["StrT", "sd_StrT", "UTurn", "sd_UTurn", "SteL", "sd_SteL",
              "SPARC_gyr", "sd_SPARC_gyr", "LDLJAcc", "sd_LDLJAcc", "CVStrT", "sd_CVStrT", "CVdsT", "sd_CVdsT",
              "P1_aCC", "sd_P1_aCC", "P2_aCC", "sd_P2_aCC", "ML_RMS", "sd_ML_RMS", "P1P2", "sd_P1P2",
              "MSwTR", "sd_MSwTR", "AP_iHR", "sd_AP_iHR", "ML_iHR", "sd_ML_iHR", "CC_iHR", "sd_CC_iHR", "dstT", "sd_dstT",
              "V", "sd_V"]
    parameters_dict = dict(zip(parameters_names, parameters))
    print_semio_parameters(parameters_dict)

    criteria_dict = dict(zip(criteria_names, criteria))
    print_semio_criteria(criteria_dict)

    # semiogram design

    print("test", args.min_z, args.max_z)
    radar_design.new_radar_superpose({"unique": criteria}, min_r=args.min_z, max_r=args.max_z, output=data_WD, name="semio")

    print("ok charge")
    sys.exit(0)
