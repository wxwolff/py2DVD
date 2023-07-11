import os, sys, glob
import numpy as np
import pandas as pd
import seaborn as sns
import datetime
import matplotlib.pyplot as plt
import math
from zipfile import ZipFile
from timeit import default_timer as timer
import warnings
warnings.filterwarnings("ignore")
'''
    Written by: David B. Wolff
    Adapted from code by A. Tokay
    Date: July 10, 2023
'''
###############################################################################################
def plot_dv_bounds(DF, DF_filt, dv_parms, title):
    plt.figure(figsize=(12,6))
    ax=plt.subplot()
    plt.scatter(DF['Deq'], DF['FallSpeed'], marker='o', color='black', s=0.5, alpha=0.1)
    plt.scatter(DF_filt['Deq'], DF_filt['FallSpeed'], color='red', s=0.5, alpha=0.1)


    diam  = dv_parms['Diam']
    tvm50 = dv_parms['Vterm'] - dv_parms['Vterm']*0.5
    tv    = dv_parms['Vterm']
    tvp50 = dv_parms['Vterm'] + dv_parms['Vterm']*0.5

    plt.plot(diam, tvp50, 'b--', label='Vt + 50%')
    plt.plot(diam, tv, 'r+', label = 'Vt')
    plt.plot(diam, tvm50, 'b--', label='Vt - 50%')
    plt.plot()
    plt.fill_between(diam, tvp50, tvm50, color='grey', alpha=0.2)
    plt.legend()
    plt.grid()
    plt.xlabel('Diameter [mm]')
    plt.ylabel('Terminal Velocity [m/s]')

    plt.title(title)
    plt.show()

    return

###############################################################################################
def get_2dvd_drop_velocity_parms(parm_file):
    '''
        Read tabularized drop size and terminal velocities from A. Tokay. These will be used
        to constructed properly binned DSDs.
    '''
    dv = np.loadtxt(parm_file).T
    Diam   = dv[0] # Drop bin
    Delta  = dv[1] # Delta-D
    Vterm  = dv[2] # Theoretical terminal velocity (Vt)
    dv_parms = {'Diam': Diam, 'Delta': Delta, 'Vterm': Vterm}
    return dv_parms

###############################################################################################
def load_2dvd_data(CAMPAIGN, inst, year, month, day, in_dir):
    '''
        Read raw (ASCII) 2DVD data
    '''
    # Get day of year to id file
    dt = datetime.datetime(year, month, day)
    tt = dt.timetuple()
    doy = tt.tm_yday

    syear  = str(year).zfill(4)
    short_year = syear[2:4]
    smonth = str(month).zfill(2)
    sday   = str(day).zfill(2)
    sdoy   = str(doy).zfill(3)   # Day of year

    # Directory where input zip file is extracter
    tmp_dir = 'tmp/'
    os.makedirs(tmp_dir, exist_ok=True)

    # Locate the file
    the_date = syear + smonth + sday
    wc = in_dir + '/V' + short_year + sdoy + '.drops.zip'
    zip_files = glob.glob(wc)
    nf = len(zip_files)
    if(nf == 0):
        print('No files found in ' + wc)
        sys.exit('Bye.')
    else:
        print('Processing ' + str(nf) + ' file!')

    # Files are zipped and unzip to a VYYddd.drops.txt file, YY=short year, ddd=Day of year
    zip_file = zip_files[0]
    print(zip_file)
    with ZipFile(zip_file, 'r') as zipObj:
        zipObj.extractall(tmp_dir)

    data_file = tmp_dir + os.path.basename(zip_file)[:-4] + '.txt'
    print('<-- ' + data_file)

    # Load contents of file into DataFrame
    print("Loading drop data into DataFrame...")
    DF = pd.read_csv(data_file, header=None, skiprows=2, index_col=None, delim_whitespace=True)

    # Remove tmp data file
    #os.remove(data_file)

    # Add correct header
    hdr = ['Time', 'Deq', 'Volume', 'FallSpeed', 'Oblateness', 'Area', 'T1', 'T2',
           'A_Height', 'B_Height', 'A_Width', 'B_Width', 'A_Min', 'B_Min', 'A_Max', 'B_Max']
    DF.columns = hdr

    # Drop two columns that are useless.
    DF = DF.drop(['T1','T2'], axis=1)
    return DF, data_file

###############################################################################################
def filter_drops_by_velocity(DF, dv_parms):
    '''
        Filter drops with velocities that are 50% over/under terminal velocity
    '''
    DF1 = DF.drop(DF.index)
    diam = dv_parms['Diam']
    vter = dv_parms['Vterm']
    nd = len(diam)
    for id in range(nd):
        d = diam[id]
        v = vter[id]
        d_mask = (DF['Deq'] > d) & (DF['Deq'] < d + 0.2)
        v_mask = (DF['FallSpeed'] > v - 0.5*v) & (DF['FallSpeed'] < v + 0.5*v)

        df1 = (DF[d_mask & v_mask]).copy()
        frames = [DF1, df1]
        DF1 = pd.concat(frames)
        #print(id, d, v, len(df1['Deq']), len(DF1['Deq']))

    return DF1.sort_values('Time').reindex(copy=False)

###############################################################################################
def rebin_dropbydrop(DF_filt, dv_parms):
    '''
        For each drop, determine it's bin location from a text table provided by A. Tokay. We will retain
        all of the drops and the measured drops sizes and terminal velocities, but will add two
        columns: Dmm and Vt, which are the binned drop sizes and terminal velocities. These bins will
        only be used in the DSD construction, not in calculating the integral parameters.
    '''

    DF_rebin = DF_filt.copy()
    DF_rebin['Dmm'] = DF_rebin['Deq']
    DF_rebin['Vt']  = DF_rebin['FallSpeed']

    diams = dv_parms['Diam']
    vterms = dv_parms['Vterm']
    nd = len(diams)
    for n in range(nd):
        diam = diams[n]
        vt = vterms[n]
        mask = (DF_rebin['Deq'] >= diam) & (DF_rebin['Deq'] < diam+0.2)
        #print(diam, vt)
        DF_rebin['Dmm'][mask] = diam
        DF_rebin['Vt'][mask]  = vt

    return DF_rebin

##############################################################################################
def construct_1min_dataframe_template(DF_rebin, year, month, day):
    '''
        Construct a one-minute template DF
    '''
    Year   = []
    Month  = []
    Day    = []
    Hour   = []
    Minute = []
    Second = []
    MOTD   = [] # Minute of the day
    HOTD   = [] # Minute of the day

    nr = 1440
    hour = np.zeros(nr, dtype='int32')
    mint = np.zeros(nr, dtype='int32')
    for ir in range(nr):
        hour[ir] = ir/60
        if(ir+1 < 1440):
            mint[ir+1] = mint[ir] + 1
            if(mint[ir+1] == 60):
                mint[ir+1] = 60 - mint[ir+1]
        #print(ir, hour[ir], mint[ir])

    for t in DF_rebin['Time']:
        s = str(t)
        Year.append(year)
        Month.append(month)
        Day.append(day)
        Hour.append(s[:2])
        Minute.append(s[3:5])
        Second.append(s[6:8])
        MOTD.append(str(int(s[:2])*60 + int(s[3:5])))
        t = float(s[3:5]) + float(s[6:8])/60
        HOTD.append( str('{:.3f}'.format(t)) )
        #HOTD.append( str(t) )

    # Parse 1st column time to get hour, minute, second.
    # Add date and time columns to master DF.
    DF_rebin.insert(0, 'Year', Year)
    DF_rebin.insert(1, 'Month', Month)
    DF_rebin.insert(2, 'Day', Day)
    DF_rebin.insert(3, 'Hour', Hour)
    DF_rebin.insert(4, 'Minute', Minute)
    DF_rebin.insert(5, 'Second', Second)
    DF_rebin.insert(6, 'MOTD', MOTD)
    DF_rebin.insert(7, 'Hour of the Day', HOTD)

    # Create new 1-minute DataFrame
    hdr_list = ['Year', 'Month', 'Day', 'Hour', 'Minute', 'MOTD', 'Hour of the Day']
    #print(hdr_list)

    # Create empty 1440 row dataframe (headers are set)
    DF_1min = pd.DataFrame(data=None,index=range(1440), columns=hdr_list)

    nr = 1440
    for ir in range(nr):
        DF_1min['Year'][ir]   = year
        DF_1min['Month'][ir]  = month
        DF_1min['Day'][ir]    = day
        DF_1min['Hour'][ir]   = hour[ir]
        DF_1min['Minute'][ir] = mint[ir]
        # Dropping Seconds column [0, 5]
        DF_1min['MOTD'][ir]   = ir
        DF_1min['Hour of the Day'][ir] = hour[ir] + mint[ir]/60

    return DF_1min

##############################################################################################
def calculate_2dvd_dsd(DF_1min, DF_rebin, dv_parms):
    '''
        Calculated on minute average integral parameters, DSD and moments.

    '''
    pi = np.pi
    dT = 60   # Integration time in seconds
    nr = 1440 # Minutes per day
    nm = 8    # Moments

    Diams = dv_parms['Diam']
    nd = len(Diams)   # Number of bins in DSD (from Table file)

    NDrops  = np.zeros(nr)       # Total drops per record
    Conc    = np.zeros(nr)       # Concentration
    LWC     = np.zeros(nr)       # Liquid Water Content
    Zm      = np.zeros(nr)       # Reflectivity [mm**6/m**3]
    dBZ     = np.zeros(nr)       # Reflectivity [dBZ]
    Rain    = np.zeros(nr)       # Rain Rate [mm/hr]
    Dm      = np.zeros(nr)       # Mass-weight mean diameter
    Accum   = np.zeros(nr)       # Rain Accumulation [mm]
    Sigma_M = np.zeros(nr)       # Variance of mass spectrum
    Dmax    = np.zeros(nr)       # Max diameter [mm]
    x3      = np.zeros(nr)       # 3rd moment of DSD
    x4      = np.zeros(nr)       # 4th moment of DSD
    M       = np.zeros([nr, nm]) # Moments of DSD
    DSD     = np.zeros([nr, nd]) # Drop Size Distribution

    for imin in range(nr):    # nr is the # of minutes/day
        minute = str(imin).zfill(2)
        Drops = DF_rebin[DF_rebin['MOTD']== str(minute)]
        ndrops = len(Drops)

        # Go through all of the drops this minute and calculate the
        # integrated parameters.
        if(ndrops > 0):
            NDrops[imin] = ndrops
            #print('MOTD, # drops: ', imin, ndrops)

            # Calculate integrated DSD parms for this minut
            for ir in range(ndrops):
                Deq        = Drops['Deq'].iloc[ir]       # Measured equivalent diameter
                Vterm      = Drops['FallSpeed'].iloc[ir] # Measured fall speed
                Dmm        = Drops['Dmm'].iloc[ir]       # Rebinned diameter
                Vt         = Drops['Vt'].iloc[ir]        # Rebinned terminal velocity
                Area       = Drops['Area'].iloc[ir]      # Measured area
                Vol        = Drops['Volume'].iloc[ir]    # Measured drop volume
                Dmax[imin] = Deq.max()

                bot        = Area * Vterm * dT
                bot1       = Area * Vt * dT

                #print(imin, ir, ndrops, bot, Vterm, bot1, Vt, Deq, Dmm)

                Rain[imin] += Vol * dT / Area
                Conc[imin] += 1.e6/bot1
                LWC[imin]  += 1e-3*(pi * Dmm**3.)/6 * (1.e6/bot1)
                Zm[imin]   += (Dmm**6 * 1.e6)/bot

                #print(imin, ir, Vt, Vterm, Vol, dT, Area, LWC[imin])
                # *** Calculate moments
                x3[imin]   += (Dmm**3 * 1.e6)/bot1
                x4[imin]   += (Dmm**4 * 1.e6)/bot1
                for im in range(nm):
                    M[imin, im] += (10**6 * ndrops * Dmm**im)/bot1

                # Calculate DSD
                for ibin in range(nd-1):
                    if((Dmm >= Diams[ibin]) & (Dmm < Diams[ibin+1])):
                        DSD[imin, ibin] += 1.e6/(bot1*0.2)

            dBZ[imin] = np.log10(Zm[imin])
            Dm[imin] = x4[imin]/x3[imin]

    # *** Calculate the rain accumulation
    Accum[0] = Rain[0]/60.
    for ir in range(1, nr):
        Accum[ir] = Accum[ir-1] + Rain[ir]/60

    Rain_dict = {'dBZ':        dBZ,
                 'Zlin':       Zm,
                 'Rain':       Rain,
                 'Accum':      Accum,
                 'Conc':       Conc,
                 'LWC':        LWC,
                 'Dm':         Dm,
                 'Dmax':       Dmax,
                 'TotalDrops': NDrops
                }

    return Rain_dict, DSD, M

##############################################################################################
def construct_dsd_dataframe(DF_1min, Rain_dict, DSD, M):
    '''
        Construct a new one-minte (1440 row) dataframe to hold our integral parameters, DSD
        and moments.
    '''
    DSD_shape = DSD.shape
    nbins = DSD_shape[1]
    M_shape = M.shape
    nmom = M_shape[1]

    time_hdr = DF_1min.columns[0:7]
    # Construct DF with 1-minute integral parameters
    rain_DF = pd.DataFrame.from_dict(Rain_dict)
    rain_DF.insert(0, time_hdr[0], DF_1min['Year'])
    rain_DF.insert(1, time_hdr[1], DF_1min['Month'])
    rain_DF.insert(2, time_hdr[2], DF_1min['Day'])
    rain_DF.insert(3, time_hdr[3], DF_1min['Hour'])
    rain_DF.insert(4, time_hdr[4], DF_1min['Minute'])
#     rain_DF.insert(5, time_hdr[5], DF_1min['MOTD'])
#     rain_DF.insert(6, time_hdr[6], DF_1min['Hour of the Day'])

    # Replace date columns with DateTime index
    rain_DF.index = pd.to_datetime(rain_DF['Year'].astype(str) + '-' +
                              rain_DF['Month'].astype(str) + '-' +
                              rain_DF['Day'].astype(str) + 'T' +
                              rain_DF['Hour'].astype(str) + ':' +
                              rain_DF['Minute'].astype(str))
    rain_DF = rain_DF.drop(columns = ['Year','Month','Day','Hour','Minute'])

    # Set up columns for DSD (1-col per drop size)
    D_hdr = []
    for id in range(nbins):
        D_hdr.append('DSD_' + str(id).zfill(2))

    # Set up columns for Moments (1-col per moment [0,7])
    M_hdr = []
    for im in range(nmom):
        M_hdr.append('Moment_' + str(im))

    # Returned dataframe has 1440 rows (minutes) and columns for
    # for integral parameters, DSD and moments.
    dsd_DF = pd.DataFrame(DSD, columns=D_hdr)
    dsd_DF.index = rain_DF.index.values

    mom_DF = pd.DataFrame(M, columns=M_hdr)
    mom_DF = rain_DF.index.values

    return rain_DF, dsd_DF, mom_DF

