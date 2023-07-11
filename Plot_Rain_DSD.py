#!/usr/bin/env python
# coding: utf-8
import os, sys, glob
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import math
from zipfile import ZipFile
import plot_2dvd as pltvd
import twodvd as vd
import warnings
warnings.filterwarnings("ignore")
##############################################################################################
if __name__ == "__main__":

    campaign = 'WFF'
    instrument = '2dvd_sn70'
    year  = 2019
    month = 7
    day   = 18
    syear  = str(year).zfill(4)
    smonth = str(month).zfill(2)
    sday   = str(day).zfill(2)

    # CSV/2019-07/WFF_2dvd_sn70_V19199.drops_DSD.csv
    rain_dir = 'CSV/' + syear + '-' + smonth + '/'
    rain_file = f"{rain_dir}/{campaign}_{instrument}_{syear}_{smonth}{sday}_rain.csv"
    print(f"<-- {rain_file}")
    #fileb = os.path.basename(rain_file)[:-9]
    #print(fileb)
    rain_df = pd.read_csv(rain_file, index_col=0)
    rain_df.index.name = 'DateTime'

    # CSV/2019-07/WFF_2dvd_sn70_V19199.drops_DSD.csv
    dsd_dir = 'CSV/' + syear + '-' + smonth + '/'
    dsd_file = f"{dsd_dir}/{campaign}_{instrument}_{syear}_{smonth}{sday}_DSD.csv"
    print(f"<-- {dsd_file}")
    #fileb = os.path.basename(rain_file)[:-9]
    #print(fileb)
    dsd_df = pd.read_csv(dsd_file, index_col=0)
    dsd_df.index.name = 'DateTime'
    
    # Get list of D/V pairs from Table
    parm_file = 'Tables/2dvd_diameter_50.txt'
    #print(f"Reading parms file: {parm_file}")
    dv_parms = vd.get_2dvd_drop_velocity_parms(parm_file)
    print()
    
    pltvd.plot_rain_parms(rain_df, campaign, instrument, savefig=True)
    
    pltvd.plot_dsd_mesh(dsd_df, dv_parms, campaign, instrument, savefig=True)
    
    print("Done.")
