import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
##############################################################################################
def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)

##############################################################################################
def oneD2twoD(vector,shape2,axis):
    if axis == 0:
        matrix = np.zeros((shape2,len(vector)))
        for h in np.arange(shape2):
            matrix[h]= vector
    elif axis == 1:
        matrix = np.zeros((len(vector),shape2))
        for h in np.arange(shape2):
            matrix[:,h]= vector
    else:
        raise ValueError("Wrong axis")
    return matrix

##############################################################################################
def set_plot_size_parms():
    SMALL_SIZE  = 14
    MEDIUM_SIZE = 18
    BIGGER_SIZE = 20
    plt.rc('font',   size=MEDIUM_SIZE)       # controls default text sizes
    plt.rc('axes',   titlesize=MEDIUM_SIZE)  # fontsize of the axes title
    plt.rc('axes',   labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick',  labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('ytick',  labelsize=MEDIUM_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=MEDIUM_SIZE)   # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    return

##############################################################################################
def plot_dsd_mesh(dsd_df, dv_parms, campaign, instrument, savefig=False):
    # Read the diameter bins and terminal velocity tables.

    dt = pd.to_datetime(dsd_df.index)
    hour = dt.hour
    diam = dv_parms['Diam']
    x = hour
    y = diam
    X, Y = np.meshgrid(x, y)
    Z = np.log10(dsd_df.T)

    xlabel = 'Hour [UTC]'
    xticks = np.arange(0,25,3)
    xlim = [0,24]
    myColor = 'blue'
    linewidth = 1.25

    # Extract dsd from DF as an numpy array
    FIGSIZE=[12, 8]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    fontsize = 12
    ax = plt.subplot()
    cmap = 'jet'
    cmap = plt.get_cmap(cmap)
    cmap.set_bad('w', 1.)

    bins=np.arange(0, 7, 0.5)
    dsd_cmap=discrete_cmap(14, base_cmap='jet')
    pm = ax.pcolormesh(X, Y, Z, cmap=dsd_cmap, vmin=0, vmax=7)

    cb = plt.colorbar(pm, ticks=bins, pad=0.06, extend='max')
    cb.set_ticks(bins)
    cb.set_label('Log$_{10}$ [Drops per m$^3$ mm$^{-1}$]', size=14)

    ax.set_xlabel(xlabel, size=fontsize*1.5)
    ax.tick_params(labelsize=fontsize*1.2)
    ax.tick_params(which='both', width=2)
    ax.tick_params(which='major', length=6)
    ax.tick_params(which='minor', length=3)
    ax.set_xticks(np.arange(0,25,3))

    ax.set_ylabel('Diameter [mm]', size=fontsize*1.5)
    ax.set_yticks(np.arange(0, 11))
    ax.set_ylim((0, 10))
    
    ax.grid(True)

    year  = str(dt.year[0])
    month = str(dt.month[0]).zfill(2)
    day   = str(dt.day[0]).zfill(2)
    
    title = f"{campaign}/{instrument}: {month}/{day}/{year}"
    ax.set_title(title, fontsize=24)

    png_dir = f"Plots/{year}-{month}/DSD"
    os.makedirs(png_dir, exist_ok=True)
    
    png_file = f"{png_dir}/{campaign}_{instrument}_{year}_{month}{day}_dsd.png"
    
    if(savefig):
        print(f"Saving figure: {png_file}")
        plt.savefig(png_file, dpi=100)
        plt.close()
        plt.show()
    return

    ax.grid()
##############################################################################################
def plot_rain_parms(df, campaign, instrument, savefig=False):
    nrows=3; ncols=2
    xticks = np.arange(0,25,3)
    xlim = (0,24)
    lw = 1.25
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12,8))

    dt = pd.to_datetime(df.index)
    hour = dt.hour + dt.minute/60

    ax = axes[0,0]
    field = 'Rain'
    ax.plot(hour, df[field], color='blue', label='Rain Rate [mm/hr]', linewidth=lw)
    ax.plot(hour, df['Accum'], color='red', label='Rain Accumulation [mm]', alpha=0.25, linewidth=lw)
    ax.set_xlabel('Hour')
    ax.set_ylabel('Rain Rate/Accum')
    ax.set_xticks(xticks)
    ax.set_xlim(xlim)
    ax.set_ylim(0,50)
    ax.set_yticks(np.arange(0, 51, 10))
    ax.grid(True)
    ax.legend(loc='upper right')

    ax = axes[0,1]
    field = 'dBZ'
    ax.plot(hour, df[field], color='blue', label='Reflectivity [dBZ]', linewidth=lw)
    ax.set_xlabel('Hour')
    ax.set_ylabel('Reflectivity')
    ax.set_xticks(xticks)
    ax.set_xlim(xlim)
    ax.set_yticks(np.arange(0, 61, 10))
    ax.grid(True)
    ax.legend(loc='upper right')

    ax = axes[1,0]
    field = 'LWC'
    ax.plot(hour, df[field], color='blue', label='LWC [g/m^3]', linewidth=lw)
    ax.set_xlabel('Hour')
    ax.set_ylabel('Liquid Water Content')
    ax.set_xticks(xticks)
    ax.set_xlim(xlim)
    ax.set_yticks(np.arange(0, 7))
    ax.set_ylim(0,6)
    ax.grid(True)
    ax.legend(loc='upper right')

    ax = axes[1,1]
    field = ''
    ax.plot(hour, df['Dmax'], color='red', label='Dmax', alpha=0.5, linewidth=lw)
    ax.plot(hour, df['Dm'], color='blue', label='Dm', linewidth=lw)
    ax.set_xlabel('Hour')
    ax.set_ylabel('Dm/Dmax')
    ax.set_xticks(xticks)
    ax.set_xlim(xlim)
    ax.set_yticks(np.arange(0, 9))
    ax.set_ylim(0, 8)
    ax.grid(True)
    ax.legend(loc='upper right')

    ax = axes[2,0]
    field = 'TotalDrops'
    ax.plot(hour, df[field], color='blue', label='Total Drops', linewidth=lw)
    ax.set_xlabel('Hour')
    ax.set_ylabel('Total Drops')
    ax.set_xticks(xticks)
    ax.set_xlim(xlim)
    ax.set_yticks(np.arange(0, 5001, 1000))
    ax.set_ylim(0, 5000)
    ax.grid(True)
    ax.legend(loc='upper right')

    ax = axes[2,1]
    field = 'Conc'
    ax.plot(hour, df[field], color='blue', label='Concentration', linewidth=lw)
    ax.set_xlabel('Hour')
    ax.set_ylabel('Concentration')
    ax.set_xticks(xticks)
    ax.set_xlim(xlim)
    ax.set_yticks(np.arange(0, 5001, 1000))
    ax.set_ylim(0, 5000)
    ax.grid(True)
    ax.legend(loc='upper right')
    
    plt.tight_layout()

    year  = str(dt.year[0])
    month = str(dt.month[0]).zfill(2)
    day   = str(dt.day[0]).zfill(2)
    
    png_dir = f"Plots/{year}-{month}/Rain"
    os.makedirs(png_dir, exist_ok=True)
    
    png_file = f"{png_dir}/{campaign}_{instrument}_{year}_{month}{day}_rain.png"
    
    if(savefig):
        print(f"Saving figure: {png_file}")
        plt.savefig(png_file, dpi=100)
        plt.close()
    else:
        plt.show()
    return

##############################################################################################
def plot_dv_parms(dv_parms):
    #dv_values
    #fig = plot.subplot()
    #dv_values.info()
    sns.lineplot(x='DropBinSize', y='Vt_Measured', color='r', data=dv_values)
    sns.lineplot(x='DropBinSize', y='Vt', color='g', data=dv_values)
    sns.lineplot(x='DropBinSize', y='Vt_gt50', dashes=True, color='b', data=dv_values)
    sns.lineplot(x='DropBinSize', y='Vt_lt50', dashes=True, color='b', data=dv_values)
    plt.legend(labels=['Vt','Vt_Measured','Vt_g50','Vt_lt50'])
    return dv_values

##############################################################################################
def plot_parsivel_matrix(pmatrix):
    fig = plt.subplot()
    plt.imshow(pmatrix, cmap='Accent')
    plt.title('Parsivel Matrix')
    return

