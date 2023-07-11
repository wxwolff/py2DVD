#!/usr/bin/env python
# coding: utf-8
import os, sys, glob
import numpy as np
import pandas as pd
import seaborn as sns
import datetime
import matplotlib.pyplot as plt
import math
from zipfile import ZipFile
import twodvd as vd
import plot_2dvd as pltvd
from timeit import default_timer as timer
import warnings
warnings.filterwarnings("ignore")

###############################################################################################
def filter_drops_by_velocity_loop(DF, dv_parms):
    '''
    # Filter drops with velocities that are 50% over/under terminal velocity
    '''
    # Make an empty copy of DF
    DF1 = DF.drop(DF.index)

    # Use D/V lookup table to filter drops with bad Vs for a given D.
    diam = dv_parms['Diam']
    vter = dv_parms['Vterm']
    nd = len(diam)
    for id in range(nd):
        d = diam[id]
        v = vter[id]
        d_mask = (DF['Dmm'] >= d) & (DF['Dmm'] < d + 0.2)
        v_mask = (DF['FallSpeed'] >= v - 0.5*v) & (DF['FallSpeed'] < v + 0.5*v)

        df1 = (DF[d_mask & v_mask]).copy()
        frames = [DF1, df1]
        DF1 = pd.concat(frames)
        #print(id, d, v, len(df1['Deq']), len(DF1['Deq']))

    return DF1.sort_values('Time').reindex(copy=False)

###############################################################################################
if __name__ == "__main__":
    CAMPAIGN = 'WFF'
    inst     = '2dvd_sn70'
    year     = 2019
    month    = 7
    day      = 18
    
    #if len(sys.argv) < 4:
    #     sys.exit('Usage: ql.py YYYY MM DD')
    #year  = int(sys.argv[1])
    #month = int(sys.argv[2])
    #day   = int(sys.argv[3])

    syear  = str(year).zfill(4)
    smonth = str(month).zfill(2)
    sday   = str(day).zfill(2)
    
    # Get list of D/V pairs from Table
    parm_file = 'Tables/2dvd_diameter_50.txt'
    print('Reading ' + parm_file)
    dv_parms = vd.get_2dvd_drop_velocity_parms(parm_file)
    print(dv_parms.keys())
    print()
       
    # Load 2DVD data into a dataframe
    print("Loading raw data for " + smonth + '/' + sday + '/' + syear)
    in_dir = 'distro/' + inst.lower() + '/ascii'
    DF, filename = vd.load_2dvd_data(CAMPAIGN, inst, year, month, day, in_dir)
    fileb = os.path.basename(filename)[:-4]
    print(fileb)
    
    # Bin each drop by A. Tokay's D/V tables
    print("Rebinning drop sizes using Tokay D/V table...")
    DF_rebin = vd.rebin_dropbydrop(DF, dv_parms)

    # Filter data to remove drops with velocities over/under 50% of terminal velocity
    DF_filt = vd.filter_drops_by_velocity(DF_rebin, dv_parms)

    title = CAMPAIGN + '/' + inst + '  ' + smonth + '/' + sday + '/' + syear
    vd.plot_dv_bounds(DF, DF_filt, dv_parms, title)
    #DF.plot(x='Deq', y='FallSpeed', kind='scatter')

    # Construct 1-minute dataframe. After this call, only date, time data
    # will be filled; however, the raw data columns (and headers) with NaN will be there.
    print("Constructing daily/1-minute template...")
    DF_1min = vd.construct_1min_dataframe_template(DF_filt, syear, smonth, sday)

    # Calculate the 1-minute DSDs.  There are multiple rows (drops) in DF for some
    # minutes, so we need to gather those and then calc DSD info.
    print('Calculating DSD...')
    Rain_dict, DSD, M = vd.calculate_2dvd_dsd(DF_1min, DF_filt, dv_parms)
    
    # Merge Rain_dict, DSD and M into common dataframe    
    print('Merging Rain_dict, DSD and M into common dataframe...')
    rain_DF, dsd_DF, mom_DF = vd.construct_dsd_dataframe(DF_1min, Rain_dict, DSD, M)
    print("Returned rain_DF, dsd_DF, and mom_DF")
    
    # Write output to a CSV file for easy ingest later.
    CSV_DIR = 'CSV/' + syear + '-' + smonth + '/'
    os.makedirs(CSV_DIR, exist_ok=True)
    
    rain_file = CSV_DIR + CAMPAIGN + '_' + inst + '_' + syear + '_' + smonth + sday + '_rain.csv'
    print(f"Rain file: {rain_file}")
    rain_DF.to_csv(rain_file)

    dsd_file = CSV_DIR + CAMPAIGN + '_' + inst + '_' + syear + '_' + smonth + sday + '_DSD.csv'
    print(f"DSD file:  {dsd_file}")
    dsd_DF.to_csv(dsd_file)

    print("Done.")
