## Plotting code associated with WSU calculations.
## Originally this code was in several separate files, but I have
## consolidated to one to keep things more consistent.

import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
import re
import math
from astropy.table import Table, QTable, join
import ipdb


## imsize vs. nchan overview plots
## ------------------------------

def make_beamsperfov_vs_nchan(result, filename,
                              array = '12m',
                              cube_limit = 40, #GB
                              max_cube_limit = 60, #GB
                              addconfigs = True,
                              mous_list = None,
                              cycle = '7',
                              plt_title=None,
                              markersize=90):
    '''
    Make plot of resolution elements per fov vs. nchan

    Date        Programmer      Description of Changes
    ----------------------------------------------------------------------
    9/14/2022   A.A. Kepey      Original Code
    
    '''

    plt.figure(figsize=(12,10),edgecolor='white',facecolor='white')

    # select array
    if  (array == '12m') or (array == '7m'):
        mosaic = (result['array'] == array) & (result['is_mosaic'] == 'T') 
        sp = (result['array'] == array) & (result['is_mosaic'] == 'F') 
    else:
        print('array does not exist: ' + array)

    # set color of background points
    if mous_list:
        sp_color = 'lightgray'
        mosaic_color = 'darkgray'
    else:
        sp_color='darkslateblue'
        mosaic_color='seagreen'

    # plot points
    plt.scatter(result[sp]['points_per_fov'],result[sp]['spw_nchan']+50,
                marker='.',label='single pointing',facecolor=sp_color)
    plt.scatter(result[mosaic]['points_per_fov'],result[mosaic]['spw_nchan']-50,
                marker='.',label='mosaics',facecolor=mosaic_color)
        
    # highlight particular MOUSes if desired
    if mous_list:     
        # add points showing some specific data sets
        for mous in mous_list:

            #idx = result['member_ous_uid'] == mous['member_ous_uid']            
            #prop = np.unique(result[idx]['proposal_id'].data[0])[0]

            if 'markersize' in mous.keys():
                markersize = mous['markersize']

                
            plt.scatter(mous['points_per_fov'],mous['spw_nchan'],label=mous['proposal_id'],
                        marker=mous['marker'],s=markersize)

            
    # fix up plot
    plt.xticks(size=18)
    plt.xlabel('Resolution elements per FOV',size=18)
    plt.xscale('log')
    plt.yticks(size=18)
    plt.ylabel('Number of channels',size=18)
    plt.grid(which='major')
    plt.grid(which='minor',linestyle=':')

    if plt_title is None:
        plt.title(array + ' Cube sizes in ALMA Cycle '+cycle,size=24)
    else:
        plt.title(plt_title,size=24)

    # TODO --  better to do this with getting axis values, but going with hardcode for now
    # calculate line for maximum number of channels
    points_per_fov = np.linspace(100,2e6,1000) 
    nchan_max = calc_nchan_max_points_per_fov(cube_limit, points_per_fov,chan_limit=7680.0)

    # calculate maximum mitigation limit -- depends on nbin which may not be able to be set
    nchan_max_fail_nbin1 = calc_nchan_max_points_per_fov(max_cube_limit, points_per_fov, 
                                                         nbin=1.0, pixels_per_beam=9.0, frac_fov=0.22,
                                                         chan_limit=7680.0)

    nchan_max_fail_nbin2 = calc_nchan_max_points_per_fov(max_cube_limit, points_per_fov, 
                                                         nbin=2.0, pixels_per_beam=9.0, frac_fov=0.22,
                                                         chan_limit=7680.0)

    # add mitigation lines
    plt.plot(points_per_fov,nchan_max,color='darkorange',label='mitigation threshold', linewidth=3)
    plt.plot(points_per_fov,nchan_max_fail_nbin1, color='firebrick',label='mitigation limit (nbin=1)',linewidth=3,
             linestyle=':')
    plt.plot(points_per_fov,nchan_max_fail_nbin2, color='firebrick',label='mitigation limit (nbin=2)',linewidth=3)

    # add configs
    if addconfigs:
        config_dict = configuration_info()
        for config in config_dict.keys():
            plt.axvline(config_dict[config]['points_per_fov'], color= 'black')
            plt.text(config_dict[config]['points_per_fov'] * 0.60, 5500, config + '\n(SF)',size=15)
            

    # add legend
    plt.legend(loc='upper left',prop={'size':14})

    # save plot
    if filename:
        plt.savefig(filename,facecolor='white',edgecolor='white')

def make_imsize_vs_nchan(result,
                         chan_type='wsu_nchan_final_stepped',
                         cube_limit = 40, #GB
                         max_cube_limit = 60, #GB
                         mit_limits = False,
                         log_imsize = np.arange(1.5,4.1,0.1),
                         log_nchan = np.arange(3.0,6.1,0.1),
                         band2_specscan=True,
                         core_theory=False,
                         title=None,
                         pltname=None,
                         xlim=(1.0,4.5),
                         ylim=(1.5,6.2),
                         nspw=10
                         ):
    '''
    '''

    imsize = np.logspace(1.5,4.0,100)
    mit_threshold = calc_nchan_max(imsize,cube_limit)
    mit_limit = calc_nchan_max(imsize,max_cube_limit)
    


    single_7m = (result['array'] == '7m') & (result['mosaic'] == 'F')
    single_12m = (result['array'] == '12m') & (result['mosaic'] == 'F')
    mosaic_7m = (result['array'] == '7m') & (result['mosaic'] == 'T')
    mosaic_12m = (result['array'] == '12m') & (result['mosaic'] == 'T')

    
    plt.figure(figsize=(8,7),edgecolor='white', facecolor='white')
    plt.subplot(111,position=[0.1,0.1,0.6,0.7])
        
    plt.scatter(np.log10(result['imsize'][single_7m]),np.log10(result[chan_type][single_7m]),
                label='SF 7m', marker='o',color='lightgrey')
    plt.scatter(np.log10(result['imsize'][single_12m]),np.log10(result[chan_type][single_12m]),
                label='SF 12m',marker='^',color='lightgrey')
    
    plt.scatter(np.log10(result['imsize'][mosaic_7m]),np.log10(result[chan_type][mosaic_7m]),
                label='Mosaic 7m',marker='o',color='darkgrey')
    plt.scatter(np.log10(result['imsize'][mosaic_12m]),np.log10(result[chan_type][mosaic_12m]),
                label='Mosaic 12m',marker='^',color='darkgrey')


    plt.plot(np.log10(imsize),np.log10(mit_threshold),color='darkmagenta',linewidth=3,
         label='current mitigation \n threshold')
    plt.plot(np.log10(imsize),np.log10(mit_limit),color='midnightblue',linewidth=3,
             label='current mitigation \n limit')

    if mit_limits:
        mit_limit_2x = calc_nchan_max(imsize,max_cube_limit*2.0)
        plt.plot(np.log10(imsize),np.log10(mit_limit_2x),color='forestgreen',linewidth=3,
                 linestyle="-.",
                 label='2x mitigation \n limit')
        
        mit_limit_10x = calc_nchan_max(imsize,max_cube_limit*10.0)
        plt.plot(np.log10(imsize),np.log10(mit_limit_10x),color='darkorange',linewidth=3,
                 linestyle='-.',
                 label='10x mitigation \n limit')
        
        mit_limit_100x = calc_nchan_max(imsize,max_cube_limit*100.0)
        plt.plot(np.log10(imsize),np.log10(mit_limit_100x),color='firebrick',linewidth=3,
                 linestyle="-.",
                 label='100x mitigation \n limit')
        
        mit_limit_500x = calc_nchan_max(imsize,max_cube_limit*500.0)
        plt.plot(np.log10(imsize),np.log10(mit_limit_500x),color='firebrick',linewidth=3,
                 linestyle="-.",
                 label='500x mitigation \n limit')


    if band2_specscan:

        plt.scatter(np.log10(band2specscan['imsize']), np.log10(band2specscan['nchan']/nspw), label="Band 2 Spectral Scan", 
                    color='magenta',marker='*',s=200,edgecolor='black')
        
        plt.scatter(np.log10(band2specscan_500MBs['imsize']), np.log10(band2specscan_500MBs['nchan']/nspw), 
                    label="Band 2 Spectral Scan \n (500MB/s cap)", 
                    color='magenta',marker='^',s=150,edgecolor='black')
        
        plt.scatter(np.log10(band2specscan_160MBs['imsize']), np.log10(band2specscan_160MBs['nchan']/nspw), 
                    label="Band 2 Spectral Scan \n (160MB/s cap)", 
                    color='magenta',marker='s',s=150,edgecolor='black')
        
    if core_theory:
        nrecs_jao = calc_minor(nproc=8, mem_per_proc=32,plotit=False)
        mit2x_proc = calc_minor(nproc=24, mem_per_proc=32,plotit=False)
        mit10x_proc = calc_minor(nproc=150, mem_per_proc=32,plotit=False)
        mit100x_proc = calc_minor(nproc=1500, mem_per_proc=32, plotit=False)
        mit500x_proc = calc_minor(nproc=7000, mem_per_proc=32, plotit=False)
    

        plt.contour(log_imsize,log_nchan,np.sqrt(nrecs_jao) , levels=[146],colors='black', label='8 nodes/32GB')
        plt.contour(log_imsize,log_nchan,np.sqrt(mit2x_proc) , levels=[146], colors='red', label='24 nodes/32GB')
        plt.contour(log_imsize,log_nchan,np.sqrt(mit10x_proc) , levels=[146], colors='green', label='150 nodes/32GB')
        plt.contour(log_imsize,log_nchan,np.sqrt(mit100x_proc) , levels=[146], colors='orange', label='1500 nodes/32GB')
        plt.contour(log_imsize,log_nchan,np.sqrt(mit500x_proc) , levels=[146], colors='blue', label='7000 nodes/32GB')
    


    plt.xlim(xlim)
    plt.ylim(ylim)
    
    plt.xlabel('log(imsize [pixels])',size=14)
    plt.ylabel('log(nchan)',size=14)
    if title:
        plt.title(title,size=16)
    plt.grid(which='both')
    plt.legend(loc='upper right',bbox_to_anchor=(1.5,1.0),borderpad=1.2)

    if pltname:
        plt.savefig(pltname)

def make_imsize_vs_nchan_hist2d(result, chan_type = 'wsu_nchan_final_stepped',
                                log_imsize_range = (1.4, 4.1),
                                log_imsize_step = 0.1,
                                log_nchan_range = (3.0, 6.1),
                                log_nchan_step = 0.1,
                                title='ALMA Cycle 7',
                                cmap='winter_r',
                                band2_specscan=True,
                                mit_limits=True,
                                pltname=None,
                                nspw=10,
                                **kwargs):
    '''


    '''

    # current limits
    cubelimit = 40 #GB
    maxcube = 60 # GB

    # calculate thresholds
    mit_imsize = np.logspace(log_imsize_range[0],log_imsize_range[1],1000)
    mit_threshold = calc_nchan_max(mit_imsize, cubelimit)

    mit_limit = calc_nchan_max(mit_imsize, maxcube)
    mit_limit_2x = calc_nchan_max(mit_imsize, maxcube*2.0)
    mit_limit_10x = calc_nchan_max(mit_imsize, maxcube*10.0)
    mit_limit_100x = calc_nchan_max(mit_imsize, maxcube*100.0)
    mit_limit_500x = calc_nchan_max(mit_imsize, maxcube*500.0)
    
    # make figure
    plt.figure(figsize=(10,8),edgecolor='white',facecolor='white')
    ax = plt.subplot(111,position=[0.1,0.1,0.6,0.8])
    
    # create bins -- add extra end point
    log_imsize_bins = np.arange(log_imsize_range[0], log_imsize_range[1] + log_imsize_step, log_imsize_step)
    log_nchan_bins = np.arange(log_nchan_range[0], log_nchan_range[1] + log_nchan_step, log_nchan_step)
    
    # plot histogram
    h, xedges, yedges,image = ax.hist2d(np.log10(result['imsize']),
                                        np.log10(result[chan_type]),
                                        bins=[log_imsize_bins,log_nchan_bins],cmap=cmap,cmin=0.5,**kwargs)


    if band2_specscan:

        color_specscan='black'
        #ax.axhline(np.log10(band2specscan_160MBs[nchanval]),label="Data Rate capped at 160MB/s",
        #                    color=color_specscan, linewidth=4, linestyle='-')        
        ax.axhline(np.log10(band2specscan_500MBs['nchan']/nspw),label="Data rate = \n 500 MB/s",
                            color=color_specscan, linewidth=4, linestyle='-')
        ax.axhline(np.log10(band2specscan['nchan']/nspw),label="Data rate = \n 1800 MB/s",
                   color=color_specscan, linewidth=4, linestyle='--')

        ax.axhline(np.log10(80*14880/nspw), label="Max data rate = \n 3600 MB/s",
                   linewidth=4,linestyle=':',color=color_specscan)

    if mit_limits:
        linewidth=4
        #ax.plot(np.log10(mit_imsize),np.log10(mit_threshold),color='magenta',linewidth=linewidth,
        #        linestyle='-',
        #        label='current mitigation \n threshold')
        ax.plot(np.log10(mit_imsize),np.log10(mit_limit),color='darkorange',linewidth=linewidth,
                linestyle='-',
                label='current mitigation \n limit')
        #ax.plot(np.log10(mit_imsize),np.log10(mit_limit_2x),color='gray',linewidth=3,
        #         linestyle="-",
        #         label='2x mitigation \n limit')
        ax.plot(np.log10(mit_imsize),np.log10(mit_limit_10x),color='darkorange',linewidth=linewidth,
                linestyle='--',
                label='10x mitigation \n limit')
        ax.plot(np.log10(mit_imsize),np.log10(mit_limit_100x),color='darkorange',linewidth=linewidth,
                linestyle="-.",
                label='100x mitigation \n limit')
        ax.plot(np.log10(mit_imsize),np.log10(mit_limit_500x),color='darkorange',linewidth=linewidth,
                linestyle=":",
                label='500x mitigation \n limit')
        
    ax.set_xlabel('log imsize',size=14)
    ax.set_ylabel('log nchan',size=14)
    ax.set_title(title)

    ax.legend(loc='upper right',bbox_to_anchor=(1.5,1.0),borderpad=1.2,handlelength=6)

    if pltname:
        plt.savefig(pltname)
    
    return h, xedges, yedges


def calc_cumulative_hist(h,
                         maxcube=60, #GB
                         mit_limits = np.array([1,2,10,100,500]),
                         nprocs = np.array([8, 24, 150, 1500, 7000]),
                         log_imsize_range = (1.4, 4.1),
                         log_imsize_step = 0.1,
                         log_nchan_range = (3.0, 6.1),
                         log_nchan_step = 0.1,
                         ylim=(0.65,1.01),
                         title = 'ALMA Cycle 7',
                         pltname=None):
    '''
    calculate the cumulative histogram of the size of data we would be able to process.

    Date        Programmer      Description of Changes
    ----------------------------------------------------------------------    
    9/16/2022   A.A. Kepley     Original Code
    
    '''


    # calculate imsize ranges
    log_imsize = np.arange(log_imsize_range[0],log_imsize_range[1], log_imsize_step)
    log_nchan = np.arange(log_nchan_range[0],log_nchan_range[1], log_nchan_step)

    # calculate cube size
    imsize, nchan = np.meshgrid( 10**(log_imsize), 10**(log_nchan))
    cube_size = calc_cube_size(imsize,nchan)
    
    # calculate numbers
    num_total = np.nansum(np.transpose(h))

    num_below_limit = np.empty(len(mit_limits))
    frac_below_limit = np.empty(len(mit_limits))

    print("Limit   Number    Fraction")
    
    # calculate fractions and number with different limits
    for i in np.arange(len(mit_limits)):
        num_below_limit[i] = np.nansum(np.transpose(h) * np.where(cube_size < mit_limits[i] * maxcube,1,0))
        frac_below_limit[i] = num_below_limit[i] / num_total
        print(mit_limits[i], num_below_limit[i],frac_below_limit[i])
        
        
    fig = plt.figure(figsize=(10,8),edgecolor='white',facecolor='white')
    ax = plt.subplot(111)

    ax.semilogx(mit_limits,frac_below_limit)
    ax.set_ylim(ylim)
    ax.tick_params(axis='y',labelsize=14)
    ax.grid()

    ax.set_xticks(mit_limits)
    ax.set_xticklabels([str(x) for x in mit_limits],fontsize=14)

    ax.axhline(1.0, color='black')
    
    ax.set_xlabel('Increase in Mitigation Limit',size=16)
    ax.set_ylabel('Fraction of Data Below Limit',size=16)
    ax.set_title(title,size=20)

    
    ax1 = ax.twiny()
    
    #ipdb.set_trace()
    
    ax1.set_xlim(ax.get_xlim())
    ax1.set_xscale('log')
    ax1.set_xticks(mit_limits)

    ax1.set_xticklabels([str(i) for i in nprocs],size=14)
    ax1.set_xlabel("Estimated Number of Cores (32GB/core)", size=14)
    
    if pltname:
        fig.savefig(pltname)


## WSU Use case plots        
## -----------------

def make_visrate_vs_imsize_plot(result,
                                visrate_type='vis_rate_typical_final_finest',
                                band2spec_scan=True,
                                band2spec_scan_visrate='vis_rate_typical',
                                title=None,
                                pltname=None):
    '''
    plot visrate vs. imsize

    Date        Programmer      Description of Changes
    ----------------------------------------------------------------------
    9/23/2022   A.A. Kepley     Original Code
    
    '''

    single_7m = (result['array'] == '7m') & (result['mosaic'] == 'F')
    single_12m = (result['array'] == '12m') & (result['mosaic'] == 'F')
    mosaic_7m = (result['array'] == '7m') & (result['mosaic'] == 'T')
    mosaic_12m = (result['array'] == '12m') & (result['mosaic'] == 'T')

    plt.figure(figsize=(9,8),facecolor='white',edgecolor='white')
    plt.subplot(111,position=[0.1,0.1,0.65,0.8])
    
    plt.scatter(np.log10(result[visrate_type][single_7m]),np.log10(result['imsize'][single_7m]),
                label='SF 7m')
    plt.scatter(np.log10(result[visrate_type][single_12m]),np.log10(result['imsize'][single_12m]),
                label='SF 12m')
    
    plt.scatter(np.log10(result[visrate_type][mosaic_7m]),
                np.log10(result['imsize'][mosaic_7m]),
                label='Mosaic 7m')
    plt.scatter(np.log10(result[visrate_type][mosaic_12m]),
                np.log10(result['imsize'][mosaic_12m]),
                label='Mosaic 12m')

    if band2spec_scan:
        if band2spec_scan_visrate in band2specscan.keys():
            plt.scatter(np.log10(band2specscan[band2spec_scan_visrate]), 
                        np.log10(band2specscan['imsize']), 
                        label="Band 2 Spectral Scan", 
                        color='magenta',marker='*',s=200, edgecolor='black')

        if band2spec_scan_visrate in band2specscan_500MBs.keys():    
            plt.scatter(np.log10(band2specscan_500MBs[band2spec_scan_visrate]), 
                        np.log10(band2specscan_500MBs['imsize']), 
                        label="Band 2 Spectral Scan \n (500 MB/s cap)", 
                        color='magenta',marker='^',s=150, edgecolor='black')

        if band2spec_scan_visrate in band2specscan_160MBs.keys():  
            plt.scatter(np.log10(band2specscan_160MBs[band2spec_scan_visrate]), 
                        np.log10(band2specscan_160MBs['imsize']), 
                        label="Band 2 Spectral Scan \n (160 MB/s cap)", 
                        color='magenta',marker='s',s=150, edgecolor='black')
    

    plt.ylim(1.0,4.5)
    plt.xlim(-2.0,4.5)
    plt.xlabel('log10(Visibility Rate [GVis/hr])',size=14)
    plt.ylabel('log10(imsize)',size=14)

    if title:
        plt.title(title,size=16)

    plt.legend(loc='upper right',bbox_to_anchor=(1.41,1.0),borderpad=1.2)
    
    if pltname:
        plt.savefig(pltname)



def make_freq_vs_fracbw(result,
                        frac_bw_type='frac_bw_initial',
                        band2spec_scan=True,
                        title=None,
                        pltname=None):
    '''
    plot fractional bandwidth vs. frequency

    Date        Programmer      Description of Changes
    --------------------------------------------------------
    9/23/2022   A.A. Kepley     Original Code
    
    '''

    single_7m = (result['array'] == '7m') & (result['mosaic'] == 'F')
    single_12m = (result['array'] == '12m') & (result['mosaic'] == 'F')
    mosaic_7m = (result['array'] == '7m') & (result['mosaic'] == 'T')
    mosaic_12m = (result['array'] == '12m') & (result['mosaic'] == 'T')

    plt.figure(figsize=(8,7),facecolor='white',edgecolor='white')

    plt.subplot(111,position=[0.1,0.1,0.60,0.75])


    plt.scatter(np.log10(result['wsu_freq'][single_7m]),np.log10(result[frac_bw_type][single_7m]),
                label='SF 7m')
    plt.scatter(np.log10(result['wsu_freq'][single_12m]),np.log10(result[frac_bw_type][single_12m]),
                label='SF 12m')
    
    plt.scatter(np.log10(result['wsu_freq'][mosaic_7m]),np.log10(result[frac_bw_type][mosaic_7m]),
                label='Mosaic 7m')
    plt.scatter(np.log10(result['wsu_freq'][mosaic_12m]),np.log10(result[frac_bw_type][mosaic_12m]),
                label='Mosaic 12m')


    if band2spec_scan:
        plt.scatter(np.log10(band2specscan['freq']), 
                    np.log10(band2specscan['frac_bw']), 
                    label="Band 2 Spectral Scan", 
                    color='magenta',marker='*',s=200, edgecolor='black')
    
    plt.xlabel('log10(Frequency [GHz])',size=14)
    plt.ylabel('log10(Fractional BW)',size=14)


    if title:
        plt.title(title)
    
    plt.legend(loc='upper right',bbox_to_anchor=(1.49,1.0),borderpad=1.2)

    if pltname:
        plt.savefig(pltname)


## Nchan derivation info
# ---------------------
        
def make_velocity_bar(result,
                      vel_label = 'velocity_resolution',
                      vel_breakpoints = [0.1,0.5,2,10],
                      band_list=[3,4,5,6,7,8,9,10],
                      title=None,
                      pltname=None):
    '''
    Purpose: Make bar chart showing percentage in each velocity bin

    Date        Programmer      Description of Changes
    ----------------------------------------------------
    9/30/2022   A.A. Kepley     Original Code
    '''

    # count up things.
    total = len(result[vel_label])

    nbands = len(band_list)
    
    nbins = len(vel_breakpoints) + 1
    count_arr = np.zeros((nbands, nbins))
    label_arr = np.zeros((nbands, nbins),dtype=object)

    for k in range(nbands):
        idx = (result[vel_label] < vel_breakpoints[0]) & (result['band_list'] == band_list[k])
        count_arr[k, 0] = len(result[idx])
        label_arr[k, 0] = "< {:4.2f} km/s".format(vel_breakpoints[0])

        #ipdb.set_trace()
        for i in np.arange(0,nbins-2):
            idx = (result[vel_label] >= vel_breakpoints[i]) & (result[vel_label] < vel_breakpoints[i+1]) & (result['band_list'] == band_list[k])
            count_arr[k,i+1] = len(result[idx])
            label_arr[k,i+1] = "{:4.2f} - \n {:4.2f} km/s".format(vel_breakpoints[i], vel_breakpoints[i+1])
            #ipdb.set_trace()
            
        count_arr[k, -1] = len(result[((result[vel_label] >= vel_breakpoints[-1]) & (result['band_list'] == band_list[k]) )])
        label_arr[k, -1] = ">= {:4.2f} km/s".format(vel_breakpoints[-1])
        #ipdb.set_trace()
        

    frac_arr = count_arr/np.sum(count_arr,axis=0)
        
    # make plot
    fig = plt.figure(figsize=(10,8),edgecolor='white',facecolor='white')
    ax = fig.add_subplot(111)


    yoffset = np.zeros(nbins) 
    for i in np.arange(0,nbands):        
        ax.bar(np.arange(0,nbins),count_arr[i,:]/total,width=0.8, label= "Band {:2d}".format(band_list[i]),bottom=yoffset)
        yoffset = yoffset + count_arr[i,:]/total 

        #ax.bar(np.arange(0,nbins),frac_arr[i,:],width=0.8, label= "Band {:2d}".format(band_list[i]),bottom=yoffset)
        #yoffset = yoffset + frac_arr[i,:] 

    ax.set_xticks(ticks=np.arange(nbins))
    ax.set_xticklabels(label_arr[0,:],size=14)


    ax.tick_params(axis='y',labelsize=14)
    ax.set_ylabel('Fraction of Total SPWs',size=16)

    ax.legend()
    
    if title:
        ax.set_title(title,size=20)

    if pltname:
        plt.savefig(pltname)

    print(total)
    print(count_arr)

    return count_arr
                      

## Pipeline runtime breakdown
## -------------------------


def plot_timedist(mytab,plot_title='Test',figname=None):
    '''
    Purpose: plot distribution of pipeline run times

    Date        Programmer      Description of Changes
    ---------------------------------------------------
    1/30/2023   A.A. Kepely     Original Code
    
    '''

    hours_per_day = 24.0 #h
    hours_per_week = hours_per_day * 7.0
    hours_per_month = hours_per_day * 30.0
    hours_per_year = hours_per_day * 365.0

    mybins = 500
    
    plt.hist(np.log10(mytab['pl_totaltime']),
             bins=mybins,
             cumulative=-1, histtype='step',
             log=True,
             density=True,
             label='Total PL time')
    plt.hist(np.log10(mytab['pl_caltime']),cumulative=-1, histtype='step',
             bins=mybins,
             log=True,
             density=True,
             label='Total PL calibration time')
    plt.hist(np.log10(mytab['pl_imgtime']),cumulative=-1, histtype='step',
             bins=mybins,
             log=True,
             density=True,
             label='Total PL image time')

    plt.axvline(np.log10(hours_per_week),color='black',linestyle=':')
    plt.axvline(np.log10(hours_per_month),color='black',linestyle='--')

    plt.text(np.log10(hours_per_week)+0.05,0.1,'1 week',rotation=90)
    plt.text(np.log10(hours_per_month)+0.05,0.1,'1 month',rotation=90)

    plt.grid(which='both',axis='both',linestyle=':')
    
    #plt.axhline(0.5,color='gray',linestyle='--')
    

    plt.title(plot_title)
    plt.legend(loc='lower left')

    plt.xlabel('Log10(Duration in Hours)')
    plt.ylabel('Fraction of Longer Durations')

    if figname:
        plt.savefig(figname)


def plot_imgtime_breakdown(mytab, plot_title='Test',figname=None):
    '''
    Purpose: plot breakdown between cube imaging, agg cont,
    findcont

    Date        Programmer      Description of Changes
    ---------------------------------------------------
    1/30/2023   A.A. Kepley     Original Code

    '''

    hours_per_day = 24.0
    hours_per_week = hours_per_day * 7.0
    hours_per_month = hours_per_day * 30.0
    hours_per_year = hours_per_day * 365.0

    mybins = 500
    

    plt.hist(np.log10(mytab['pl_imgtime']),cumulative=-1, histtype='step',
             bins=mybins,
             log=True,
             density=True,
             label='Total PL image time',
             color='green')
    
    plt.hist(np.log10(mytab['pl_cubetime']),cumulative=-1, histtype='step',
             bins=mybins,
             log=True,
             density=True,
             label='Cube image time',
             color='orange')

    plt.hist(np.log10(mytab['pl_fctime']),cumulative=-1, histtype='step',
             bins=mybins,
             log=True,
             density=True,
             label='Findcont time')


    plt.hist(np.log10(mytab['pl_aggtime']),cumulative=-1, histtype='step',
             bins=mybins,
             log=True,
             density=True,
             label='aggcont time')

    plt.axvline(np.log10(hours_per_week),color='black',linestyle=':')
    plt.axvline(np.log10(hours_per_month),color='black',linestyle='--')

    plt.text(np.log10(hours_per_week)+0.05,0.1,'1 week',rotation=90)
    plt.text(np.log10(hours_per_month)+0.05,0.1,'1 month',rotation=90)


    plt.grid(which='both',axis='both',linestyle=':')
    #plt.axhline(0.5,color='gray',linestyle='--')
    

    plt.title(plot_title)
    plt.legend(loc='lower left')

    plt.xlabel('Log10(Duration in Hours)')
    plt.ylabel('Fraction of Longer Durations')

    if figname:
        plt.savefig(figname)

def plot_wsu_pl_time(mydb,pl_type='caltime',
                     plot_title='Cycle 7 & 8 predictions',
                     figname=None):
    '''
    Purpose: calculate pipeline calibration time

    Date        Programmer      Description of Changes
    ---------------------------------------------------
    1/31/2023   A.A. Kepley     Original Code
    '''

    hours_per_day = 24.0
    hours_per_week = hours_per_day * 7.0
    hours_per_month = hours_per_day * 30.0
    hours_per_year = hours_per_day * 365.0

    mybins = 500

    plt.hist(np.log10(mydb['pl_'+pl_type]),cumulative=-1,histtype='step',
             bins = mybins,
             log=True,
             density=True,
             color='#1f77b4',
             label='BLC')


    if pl_type in ['imgtime','totaltime']:        
        plt.hist(np.log10(mydb['wsu_pl_'+pl_type+'_early_mit']),cumulative=-1,histtype='step',
                 bins = mybins,
                 log=True,
                 density=True,
                 color='#ff7f0e',
                 linestyle='-',
                 label='early WSU (mitigated)')
    

    plt.hist(np.log10(mydb['wsu_pl_'+pl_type+'_early']),cumulative=-1,histtype='step',
             bins = mybins,
             log=True,
             density=True,
             color='#ff7f0e',
             linestyle=':',
             label='early WSU')

        
    plt.hist(np.log10(mydb['wsu_pl_'+pl_type+'_later_2x']),cumulative=-1,histtype='step',
             bins = mybins,
             log=True,
             density=True,
             color='#2ca02c',
             label='Later WSU (2x)')
    
    plt.hist(np.log10(mydb['wsu_pl_'+pl_type+'_later_4x']),cumulative=-1,histtype='step',
             bins = mybins,
             log=True,
             density=True,
             color='#d62728',
             label='Later WSU (4x)')

    
        
    plt.axvline(np.log10(hours_per_week),color='black',linestyle=':')
    plt.axvline(np.log10(hours_per_month),color='black',linestyle='--')
    plt.axvline(np.log10(hours_per_year),color='black')

    plt.text(np.log10(hours_per_week)+0.05,0.3,'1 week',rotation=90)
    plt.text(np.log10(hours_per_month)+0.05,0.3,'1 month',rotation=90)
    plt.text(np.log10(hours_per_year)+0.05,0.3,'1 year',rotation=90)

    plt.axhline(0.1,color='gray',linestyle=':')
    plt.text(0,0.1,'90% take less time')

    plt.axhline(0.05,color='gray',linestyle=':')
    plt.text(0,0.05,'95% smaller')

    plt.axhline(0.01,color='gray',linestyle=':')
    plt.text(0,0.01,'99% smaller')

    plt.grid(which='both',axis='both',linestyle=':')
    
    plt.title(plot_title)
    plt.legend(loc='lower left')

    plt.xlabel('Log10(Duration in Hours)')
    plt.ylabel('Fraction of Longer Durations')

    if figname:
        plt.savefig(figname)



        
    
def plot_cal_img_time(mytab,plot_title='Cycle'):
    '''
    Purpose: Plot fraction of time spent on in calibration and imaging portions
    of the pipeline

    Date        Programmer      Description of Changes
    ----------------------------------------------------------------------
    1/23/2023   A.A. Kepley     Original Code
    '''

    idx = (mytab['procedure'] == 'hifa_caliamge') | (mytab['procedure'] == 'hifa_calimage_renorm')
    
    frac_cal = mytab['pl_caltime'][idx]/mytab['pl_totaltime'][idx]
    frac_imgtime = mytab['pl_imgtime'][idx]/mytab['pl_totaltime'][idx]
    frac_cubetime = mytab['pl_cubetime'][idx]/mytab['pl_totaltime'][idx]
    frac_aggtime = mytab['pl_aggtime'][idx]/mytab['pl_totaltime'][idx]
    frac_fctime = mytab['pl_fctime'][idx]/mytab['pl_totaltime'][idx]

    mylabels = ['Calibration','Imaging', 'Cubes',' Agg Cont','Findcont']

    mydata = [frac_cal,frac_imgtime,frac_cubetime,frac_aggtime,frac_fctime]

    fig, ax = plt.subplots(1,1,figsize=(8,6))

    parts = ax.violinplot(mydata,showmedians=True)

    ax.set_xticks(np.arange(1,len(mylabels)+1),labels=mylabels)
    ax.set_title(plot_title)


def plot_casa_time(mytab, plot_title='Cycle 7', figname=None):
    '''
    Purpose: Plot fraction of time spent on CASA in pipeline runs

    Date        Programmer      Description of Changes
    ----------------------------------------------------------------------
    1/23/2023   A.A. Kepley     Original Code
    '''


    
    frac_casa = (mytab['casatasks'] + mytab['casatools'])/mytab['pipetime']
    frac_tasks = (mytab['casatasks'] / mytab['pipetime'])
                  
    frac_tools = (mytab['casatools'] /mytab['pipetime'])
    
    mylabels = ['CASA', 'Tasks', 'Tools']
    
    fig, ax = plt.subplots(1,1,figsize=(8,6))

    parts = ax.violinplot([frac_casa,frac_tasks,frac_tools], showmedians=True)

    ## Can use parts to set colors
    
    ax.set_xticks(np.arange(1,len(mylabels)+1),labels=mylabels)
    ax.set_title(plot_title)

    ax.set_ylabel('Fraction of pipeline run time')
    
    if figname:
        plt.savefig(figname)
    


def plot_casa_task_time(mytab, plot_title='Cycle 7', figname=None):
    '''
    Purpose: Plot fraction of time spent on different CASA tasks

    Date        Programmer      Description of Changes
    ----------------------------------------------------------------------
    1/23/2023   A.A. Kepley     Original Code
    '''

    mycols_tasks = ['importasdm','flagdata','listobs','plotms','clearstat','flagcmd','gencal','plotbandpass','wvrgcal','gaincal','bandpass','setjy','flagmanager','applycal','fluxscale','tclean','exportfits','mstransform','imhead','immoments','imstat','imsubimage','makemask','immath','uvcontfit','visstat']


    mydata = []
    for mycol in mycols_tasks:
        mydata.append(mytab[mycol]/mytab['pipetime'])

        
    fig, ax = plt.subplots(1,1,figsize=(12,6))
    fig.subplots_adjust(bottom=0.2,top=0.92)
    parts = ax.violinplot(mydata,showmedians=True, widths=0.9)
    
    ax.set_xticks(np.arange(1,len(mycols_tasks)+1),labels=mycols_tasks,rotation=90)

    ax.set_title(plot_title)

    ax.set_ylabel('Fraction of pipeline run time')
    
    if figname:
        plt.savefig(figname)
    
def plot_casa_tool_time(mytab, plot_title='Cycle 7'):
    '''
    Purpose: Plot fraction of time spent on different CASA tasks

    Date        Programmer      Description of Changes
    ----------------------------------------------------------------------
    1/23/2023   A.A. Kepley     Original Code
    '''

    mycols_tasks = ['imager.selectvis','imager.advise','imager.apparentsens', 'ia.getprofile']


    mydata = []
    for mycol in mycols_tasks:
        mydata.append(mytab[mycol]/mytab['pipetime'])

        
    fig, ax = plt.subplots(1,1,figsize=(8,6))
    parts = ax.violinplot(mydata,showmedians=True, widths=0.5)
    
    ax.set_xticks(np.arange(1,len(mycols_tasks)+1),labels=mycols_tasks,rotation=90)

    ax.set_title(plot_title)

        

## WSU Data property breakdown
## -------------------------

def plot_cubesize_comparison(mydb,
                             plot_title='Cube Size',
                             figname=None):
    '''
    Purpose: compare the cubesize distribution
    between WSU and BLC

    Date        Programmer      Description of Changes
    ---------------------------------------------------
    1/31/2023   A.A. Kepley     Original Code
    '''

    maxcubesize = 40
    cubesizelimit = 60
    
    mybins = 500

    plt.hist(np.log10(mydb['mitigatedcubesize']),cumulative=-1,histtype='step',
             bins=mybins,
             log=True,
             density=True,
             color='#1f77b4',
             linestyle='-',
             label="BLC (mitigated)")

    plt.hist(np.log10(mydb['predcubesize']), cumulative=-1, histtype='step',
             bins=mybins,
             log=True,
             density=True,
             color='#1f77b4',
             linestyle=':',
             label="BLC (unmitigated)")


    plt.hist(np.log10(mydb['wsu_cubesize_stepped2_mit']), cumulative=-1, histtype='step',
             bins=mybins,
             log=True,
             density=True,
             color='#ff7f0e',
             label="WSU (mitigated)")


    plt.hist(np.log10(mydb['wsu_cubesize_stepped2']), cumulative=-1, histtype='step',
             bins=mybins,
             log=True,
             density=True,
             color='#ff7f0e',
             linestyle=':',
             label="WSU (unmitigated)")

    plt.axvline(np.log10(maxcubesize), color='black', linestyle=':')
    plt.axvline(np.log10(cubesizelimit), color='black', linestyle='-')
    
    plt.text(np.log10(maxcubesize)-0.1,0.5,'40GB',horizontalalignment='right')
    plt.text(np.log10(cubesizelimit)+0.1,0.5,'60GB',horizontalalignment='left')

    plt.axhline(0.1,color='gray',linestyle=':')
    plt.text(0,0.1,'90% smaller')

    plt.axhline(0.05,color='gray',linestyle=':')
    plt.text(0,0.05,'95% smaller')

    plt.axhline(0.01,color='gray',linestyle=':')
    plt.text(0,0.01,'99% smaller')

    plt.grid(which='both',axis='both',linestyle=':')
    
    plt.xlabel('Log10(Cubesize in GB)')
    plt.ylabel('Fraction of Larger Cubes')

    plt.title(plot_title)
    plt.legend(loc='lower left')

    if figname:
        plt.savefig(figname)
    
def plot_productsize_comparison(mydb,
                                plot_title='Product Size',
                                figname=None):
    '''
    Purpose: compare the productsize distribution
    between WSU and BLC

    Date        Programmer      Description of Changes
    --------------------------------------------------
    1/31/2023   A.A. Kepley     Original Code

    '''

    maxproductsize = 500 #GB
    mybins = 500

    plt.hist(np.log10(mydb['mitigatedprodsize']),cumulative=-1,histtype='step',
             bins=mybins,
             log=True,
             density=True,
             color='#1f77b4',
             linestyle='-',
             label="BLC (mitigated)")

    plt.hist(np.log10(mydb['initialprodsize']), cumulative=-1, histtype='step',
             bins=mybins,
             log=True,
             density=True,
             color='#1f77b4',
             linestyle=':',
             label="BLC (unmitigated)")


    plt.hist(np.log10(mydb['wsu_productsize_early_stepped2_mit']), cumulative=-1, histtype='step',
             bins=mybins,
             log=True,
             density=True,
             color='#ff7f0e',
             label="Early WSU (mitigated)")

    plt.hist(np.log10(mydb['wsu_productsize_early_stepped2']), cumulative=-1, histtype='step',
             bins=mybins,
             log=True,
             density=True,
             color='#ff7f0e',
             linestyle=':',
             label="Early WSU")

    plt.hist(np.log10(mydb['wsu_productsize_later_2x_stepped2']), cumulative=-1, histtype='step',
             bins=mybins,
             log=True,
             density=True,
             color='#2ca02c',
             label="Later WSU (2x)")

    plt.hist(np.log10(mydb['wsu_productsize_later_4x_stepped2']), cumulative=-1, histtype='step',
             bins=mybins,
             log=True,
             density=True,
             color='#d62728',
             label="Later WSU (4x)")

    plt.axvline(np.log10(maxproductsize), color='black', linestyle=':')

    
    plt.text(np.log10(maxproductsize)-0.1,0.5,'500GB',horizontalalignment='right')

    plt.axhline(0.1,color='gray',linestyle=':')
    plt.text(0,0.1,'90% smaller')

    plt.axhline(0.05,color='gray',linestyle=':')
    plt.text(0,0.05,'95% smaller')

    plt.axhline(0.01,color='gray',linestyle=':')
    plt.text(0,0.01,'99% smaller')

    plt.grid(which='both',axis='both',linestyle=':')
    
    plt.xlabel('Log10(Product size in GB)')
    plt.ylabel('Fraction of Larger Products')

    plt.title(plot_title)
    plt.legend(loc='lower left')

    if figname:
        plt.savefig(figname)
    

def plot_datavol_comparison(mydb,
                            plot_title="Visibility Data Volume",
                            figname=None):
    '''
    Purpose: compare the nvis distribution
    between WSU and BLC
    
    Date        Programmer      Description of Changes
    --------------------------------------------------
    1/31/2023   A.A. Kepley     Original Code
    '''

    mybins = 500

    plt.hist(np.log10(mydb['blc_datavol_typical_total']),cumulative=-1,histtype='step',
             bins=mybins,
             log=True,
             density=True,
             color='#1f77b4',             
             label="BLC")
    
    plt.hist(np.log10(mydb['wsu_datavol_early_stepped2_typical_total']), cumulative=-1,histtype='step',
             bins=mybins,
             log=True,
             density=True,
             color='#ff7f0e',
             label="early WSU")

    plt.hist(np.log10(mydb['wsu_datavol_later_2x_stepped2_typical_total']), cumulative=-1,histtype='step',
             bins=mybins,
             log=True,
             density=True,
             color='#2ca02c',
             label="later WSU (2x)")

    plt.hist(np.log10(mydb['wsu_datavol_later_4x_stepped2_typical_total']), cumulative=-1,histtype='step',
             bins=mybins,
             log=True,
             density=True,
             color='#d62728',
             label="later WSU (4x)")
        

    plt.axhline(0.1,color='gray',linestyle=':')
    plt.text(0,0.1,'90% smaller')

    plt.axhline(0.05,color='gray',linestyle=':')
    plt.text(0,0.05,'95% smaller')

    plt.axhline(0.01,color='gray',linestyle=':')
    plt.text(0,0.01,'99% smaller')

    plt.grid(which='both',axis='both',linestyle=':')
    
    plt.xlabel('Log10(Visibility Data Volume (GB))')
    plt.ylabel('Fraction of Larger Data')

    plt.title(plot_title)
    plt.legend(loc='lower left')

    if figname:
        plt.savefig(figname)

    
        
def plot_datarate_result_hist(mydb,
                              bin_min=-1,
                              bin_max=-1,   
                              nbin=10,
                              data_val = 'blc_datarate_typical', 
                              title='',
                              pltname=None):
    '''
    Purpose: create plots and calculate result table 

    Inputs: mydb with weights, data rates, and system performance

    Output:
        if pltname:
                plot showing fraction at each data rate

        if tblname:
                csv file giving fraction, data rate in each bin

    Date        Programmer      Description of Changes
    ------------------------------------------------------------------------
    2/27/2023   A.A. Kepley     Original Code

    
    '''
    import re
    
    if data_val not in mydb.columns:
        print("Column not found in database: "+data_val)
        return

    if bin_min < 0:
        bin_min = np.min(mydb[data_val]).value

    if bin_max < 0:
        bin_max = np.max(mydb[data_val]).value
    
    # nbin+1 not nbin to take care of fence post 
    mybins = np.linspace(bin_min,bin_max,nbin+1)

    # select color
    if data_val == 'blc_datarate_typical':
        mycolor = '#1f77b4'
    elif data_val == 'wsu_datarate_early_stepped2_typical':
        mycolor = '#ff7f0e'
    elif data_val == 'wsu_datarate_later_2x_stepped2_typical':
        mycolor = '#2ca02c'
    elif data_val == 'wsu_datarate_later_4x_stepped2_typical':
        mycolor = '#d62728'
    else:
        print("Data value not found. Using default color.")
        mycolor = 'black'
    
    fig, ax1 = plt.subplots()
    
    myhist = ax1.hist(mydb[data_val].value,bins=mybins,
                      log=True,
                      color=mycolor,
                      weights=mydb['weights_all'])

    ax1.set_xlabel('Data Rate (GB/s)')
    ax1.set_ylabel('log(Fraction of time)')

    ax1.set_title(title)
    
    if pltname:
        plt.savefig(pltname)
        

        
def plot_soc_result_hist(mydb,
                         bin_min=-1,
                         bin_max=-1,   
                         nbin=10,
                         data_val = 'blc_sysperf_typical', 
                         title='',
                         pltname=None):
    '''
    Purpose: create plots and calculate result table 

    Inputs: mydb with weights, data rates, and system performance

    Output:
        if pltname:
                plot showing fraction at each data rate

        if tblname:
                csv file giving fraction, data rate in each bin

    TODO: ADD WEIGHTED AVERAGE TO PLOTS TO SHOW???

    Date        Programmer      Description of Changes
    ------------------------------------------------------------------------
    2/27/2023   A.A. Kepley     Original Code

    '''
    import re
    
    if data_val not in mydb.columns:
        print("Column not found in database: "+data_val)
        return

    if bin_min < 0:
        bin_min = np.min(mydb[data_val]).value

    if bin_max < 0:
        bin_max = np.max(mydb[data_val]).value
    
    # nbin+1 not nbin to take care of fence post 
    mybins = np.linspace(bin_min,bin_max,nbin+1)

    # select color
    if data_val == 'blc_sysperf_typical':
        mycolor = '#1f77b4'
    elif data_val == 'wsu_sysperf_early_stepped2_typical':
        mycolor = '#ff7f0e'
    elif data_val == 'wsu_sysperf_later_2x_stepped2_typical':
        mycolor = '#2ca02c'
    elif data_val == 'wsu_sysperf_later_4x_stepped2_typical':
        mycolor = '#d62728'
    else:
        print("Data value not found. Using default color.")
        mycolor = 'black'
    
    fig, ax1 = plt.subplots()
    
    myhist = ax1.hist(mydb[data_val].value,bins=mybins,
                      log=True,
                      color=mycolor,
                      weights=mydb['weights_all'])

    ax1.set_xlabel('System Performance (PFLOP/s)')
    ax1.set_ylabel('Fraction of time')

    ax1.set_title(title)
    
    if pltname:
        plt.savefig(pltname)

    

def plot_soc_result_cumulative(mydb,
                               plot_title="System Performance",
                               figname=None):

    '''
    Purpose: create cumulative histograms showing size of compute

    Inputs: mydb with weights, data rates, and system performance

    Output:
        if pltname
                plot showing fraction at each data rate


    Date        Programmer      Description of Changes
    --------------------------------------------------
    4/20/2023   A.A. Kepley     Original Code
    
    '''

    mybins = 500

    plt.hist(np.log10(mydb['blc_sysperf_typical']), cumulative=-1, histtype='step',
             bins=mybins,
             log=True,
             density=True,
             color='#1f77b4',             
             label='BLC')

    
    plt.hist(np.log10(mydb['wsu_sysperf_early_stepped2_typical']), cumulative=-1, histtype='step',
             bins=mybins,
             log=True,
             density=True,
             color='#ff7f0e',
             label='early WSU')

    plt.hist(np.log10(mydb['wsu_sysperf_later_2x_stepped2_typical']), cumulative=-1, histtype='step',
             bins=mybins,
             log=True,
             density=True,
             color='#2ca02c',
             label='later WSU (2x)')
    

    plt.hist(np.log10(mydb['wsu_sysperf_later_4x_stepped2_typical']), cumulative=-1, histtype='step',
             bins=mybins,
             log=True,
             density=True,
             color = '#d62728',
             label='later WSU (4x)')

    plt.axhline(0.1,color='gray',linestyle=':')
    plt.text(0,0.1,'90% smaller')

    plt.axhline(0.05,color='gray',linestyle=':')
    plt.text(0,0.05,'95% smaller')

    plt.axhline(0.01,color='gray',linestyle=':')
    plt.text(0,0.01,'99% smaller')

    plt.grid(which='both',axis='both',linestyle=':')
    
    plt.xlabel('log (System Performance (PFLOP/s))')
    plt.ylabel('Fraction of Larger Data')

    plt.title(plot_title)
    plt.legend(loc='lower left')

    if figname:
        plt.savefig(figname)
