from astropy.table import Table, QTable, join
import numpy as np
import astropy.units as u
from astropy import constants as const
import math
import ipdb

def calc_talon_specwidth(specwidth):
    '''
    calculate the values that TALON would use

    Input:
    * specwidth in kHz
    
    Output:
    * channel size
    * number of channels averaged
    '''
    import math
    
    talon_chan = 13.5 #kHz

    if specwidth > talon_chan:
        chan_avg = float(math.floor(specwidth / talon_chan))
        if chan_avg < 1.0:
            chan_avg = 1.0
        specwidth_talon = talon_chan * chan_avg
    
    return specwidth_talon, chan_avg


def create_database(cycle7tab):
    '''
    
    create database of cycle7 parameters for WSU size of computing estimate
    
    '''
    
    # get MOUS list
    mousList = np.unique(cycle7tab['member_ous_uid'])
     
    # setup variables to hold values.
    #-----------------

    # overall info
    if_mous_list = []
    proposal_id_list = []
    array_list = []

    # basic observing info
    ntarget_list = []
    target_name_list = []
    npol_list = []
    band_list_array = []

    # image info
    s_fov_list = []
    s_resolution_list = []
    mosaic_list = []
    pb_list = []
    imsize_list = []
    cell_list = []

    # blc info
    blc_specwidth = []
    blc_freq = []
    blc_nchan_agg = []
    blc_nchan_max = []
    blc_nspw = []
    blc_bw_agg = []
    blc_bw_max = []
    blc_vel_res = []

    # WSU info
    wsu_npol_list = []
    wsu_bandwidth_final = []
    wsu_bandwidth_initial = []
    wsu_bandwidth_spw = []

    wsu_nspw_initial = []
    wsu_nspw_final = []

    wsu_tint_list = [] 
    wsu_freq_list = []

    wsu_specwidth_finest = []
    wsu_chanavg_finest = []
    wsu_velres_finest = []

    wsu_specwidth_stepped = []
    wsu_chanavg_stepped = []
    wsu_velres_stepped = []

    wsu_specwidth_stepped2 = []
    wsu_chanavg_stepped2 = []
    wsu_velres_stepped2 = []


    # number of  antennas assumed for data rate calculations
    nant_typical_list = []
    nant_array_list = []
    nant_all_list = []


    # fill in values
    for mymous in mousList:
        idx_mous = cycle7tab['member_ous_uid'] == mymous
    
        # 12m or 7m
        array = np.unique(cycle7tab[idx_mous]['array'])
        # skip TP data
        if array == 'TP':
            continue

        # Otherwise continue
        if len(array) > 1:
            print("more than one array found for " + mymous)

        # get number of targets and names
        mytargets = np.unique(cycle7tab[idx_mous]['target_name'])
        ntarget = np.unique(cycle7tab[idx_mous]['ntarget'])

        # loop over targets and extract info
        for target_name in mytargets:
            idx = (cycle7tab['member_ous_uid'] == mymous) & (cycle7tab['target_name'] == target_name)

            # MOUS
            #if_mous_list = np.append(if_mous_list,mymous)
            if_mous_list.append(mymous)
            
            # targetname
            #target_name_list = np.append(target_name_list,target_name)
            target_name_list.append(target_name)
            
            # n targets
            #ntarget_list = np.append(ntarget_list,ntarget)
            ntarget_list.append(ntarget)
            
            # proposal id 
            proposal_id = np.unique(cycle7tab[idx]['proposal_id'])
            #proposal_id_list = np.append(proposal_id_list,proposal_id)
            proposal_id_list.append(proposal_id)
            
            # array info
            #array_list = np.append(array_list,array)
            array_list.append(array)
            
            if array == '12m':
                nant_typical = 47
                nant_array = 54
                nant_all = 66 #12m+7m+TP
                tint = 3.024 #s
            elif array == '7m':
                nant_typical = 10
                nant_array = 12
                nant_all = 16 # total power plus 7m
                tint = 10.08 #s
            #nant_typical_list = np.append(nant_typical_list,nant_typical)
            #nant_array_list = np.append(nant_array_list,nant_array)
            #nant_all_list = np.append(nant_all_list, nant_all)
            #tint_list = np.append(tint_list,tint)

            nant_typical_list.append(nant_typical)
            nant_array_list.append(nant_array)
            nant_all_list.append(nant_all)
            wsu_tint_list.append(tint)
            
            # FOV
            s_fov = np.mean(cycle7tab[idx]['s_fov']) 
            #s_fov_list = np.append(s_fov_list,s_fov)
            s_fov_list.append(s_fov)
            
            
            # Resolution
            s_resolution = np.mean(cycle7tab[idx]['s_resolution'])
            #s_resolution_list = np.append(s_resolution_list, s_resolution)
            s_resolution_list.append(s_resolution)
            
            # mosaic
            mosaic = np.unique(cycle7tab[idx]['is_mosaic'])
            if len(mosaic) > 1:
                print("mosaic and single pointings in same MOUS " + mymous + ". Setting mosaic to True")
                mosaic = 'T'
            #mosaic_list = np.append(mosaic_list,mosaic)
            mosaic_list.append(mosaic)
            
            # imsize
            imsize = np.mean(cycle7tab[idx]['imsize'])
            #imsize_list = np.append(imsize_list,imsize)
            imsize_list.append(imsize)
            
            # pb
            pb = np.mean(cycle7tab[idx]['pb'])
            #pb_list = np.append(pb_list,pb)
            pb_list.append(pb)
            
            # cell
            cell = np.mean(cycle7tab[idx]['cell'])
            #cell_list = np.append(cell_list,cell)
            cell_list.append(cell)
            
            
          

            # BLC info
            # ---------

            # polarization states
            pol_states = np.unique(cycle7tab[idx]['pol_states'])
            if len(pol_states) > 1:
                print("print multiple polarization setups in same MOUS " + mymous)
            npol = len(pol_states.data[0].split('/')[1:-1])
            #npol_list = np.append(npol_list,npol)
            npol_list.append(npol)
            
            specwidth_finest = min(cycle7tab[idx]['spw_specwidth']) #kHz
            blc_specwidth.append(specwidth_finest)

            freq = np.mean(cycle7tab[idx]['spw_freq']) #GHz
            blc_freq.append(freq) 
            
            vel_res =  min(((cycle7tab[idx]['spw_specwidth']*1e3) / (cycle7tab[idx]['spw_freq']*1e9)) * const.c.to('km/s')).value #km/s
            blc_vel_res.append(vel_res)

            nchan_agg = sum(cycle7tab[idx]['spw_nchan'])
            blc_nchan_agg.append(nchan_agg)
            
            nchan_max = max(cycle7tab[idx]['spw_nchan'])
            blc_nchan_max.append(nchan_max)
                        
            nspw = len(cycle7tab[idx])
            blc_nspw.append(nspw)
            
            # get maximum spectral window
            bw_max = max(cycle7tab[idx]['bandwidth'])/1e9 #bandwidth in Hz
            blc_bw_max.append(bw_max)

            # total aggregate bandwidth -- does NOT account for overlapping windows
            bw_agg = np.sum(cycle7tab[idx]['bandwidth'])/1e9 # bandwidth in Hz 
            blc_bw_agg.append(bw_agg)
            
            # WSU Frequency
            # -------------

            # Assuming WSU center frequency is the same as the BLC center frequency
            wsu_freq_list.append(freq)

            # WSU polarization
            # ----------------

            # assuming all single pol will switch to dual pol
            if npol == 1:
                wsu_npol = 2
            wsu_npol_list.append(wsu_npol)

            
            # WSU spectral resolution
            # -----------------------------
            
            # I believe that spec_width is what i want because that is the spectral 
            # resolution which is greater than the channel spacing for cases where 
            # averaging isn't happening for the channels
            
            ## finest
            (specwidth_finest_talon, chanavg_finest_talon) = calc_talon_specwidth(specwidth_finest)
            #wsu_specwidth_finest = np.append(wsu_specwidth_finest, specwidth_finest_talon )
            wsu_specwidth_finest.append(specwidth_finest_talon)
            #wsu_chanavg_finest = np.append(wsu_chanavg_finest, chanavg_finest_talon)
            wsu_chanavg_finest.append(chanavg_finest_talon)
        
            velres_finest_tmp = (specwidth_finest_talon*1e3/(freq*1e9)) * const.c.to('km/s').value
            wsu_velres_finest.append(velres_finest_tmp)

            
            ## stepped -- 4 steps
            if vel_res > 10.0 :
                vel_res_tmp = 10.0 # km/s
            elif vel_res > 1.0:
                vel_res_tmp = 1.0 # km/s
            elif vel_res > 0.1:
                vel_res_tmp = 0.1 # km/s
            else:
                vel_res_tmp = 0.1
               
            specwidth_tmp = (vel_res_tmp / const.c.to('km/s').value) * np.mean(cycle7tab[idx]['spw_freq']*1e9)/1e3 #kHz
            (specwidth_stepped_talon,chanavg_stepped_talon) = calc_talon_specwidth(specwidth_tmp)
            #wsu_specwidth_stepped = np.append(wsu_specwidth_stepped,specwidth_stepped_talon)
            wsu_specwidth_stepped.append(specwidth_stepped_talon)
            #wsu_chanavg_stepped = np.append(wsu_chanavg_stepped, chanavg_stepped_talon)
            wsu_chanavg_stepped.append(chanavg_stepped_talon)

            velres_stepped_tmp = (specwidth_stepped_talon*1e3/(freq*1e9)) * const.c.to('km/s').value
            wsu_velres_stepped.append(velres_stepped_tmp)

            
            ## stepped -- 5 steps
            # finer coverage around 1km/s. At band 6 projects often are slightly over 1 km/s to get full bandwidth.
            if vel_res > 10.0 :
                vel_res_tmp = 10.0 # km/s
            elif vel_res > 2.0 :
                vel_res_tmp = 2.0 # km/s
            elif vel_res > 0.5:
                vel_res_tmp = 0.5 # km/s
            elif vel_res > 0.1:
                vel_res_tmp = 0.1 # km/s
            else:
                #vel_res_tmp = 0.1
                vel_res_tmp = vel_res
                
            specwidth_tmp = (vel_res_tmp / const.c.to('km/s').value) * np.mean(cycle7tab[idx]['spw_freq']*1e9)/1e3 #kHz
            (specwidth_stepped2_talon,chanavg_stepped2_talon) = calc_talon_specwidth(specwidth_tmp)
            #wsu_specwidth_stepped2 = np.append(wsu_specwidth_stepped2,specwidth_stepped2_talon)
            wsu_specwidth_stepped2.append(specwidth_stepped2_talon)
            #wsu_chanavg_stepped2 = np.append(wsu_chanavg_stepped2, chanavg_stepped2_talon)
            wsu_chanavg_stepped2.append(chanavg_stepped2_talon) 

            velres_stepped2_tmp = (specwidth_stepped2_talon * 1e3 / (freq*1e9)) * const.c.to('km/s').value
            wsu_velres_stepped2.append(velres_stepped2_tmp)
                                    
            # WSU BW
            # ------------

            # each spw likely to have 1.6GHz BW
            #wsu_bandwidth_spw = np.append(wsu_bandwidth_spw,1.6) 
            wsu_bandwidth_spw.append(1.6)
            
            # everything will have 16GHz eventually and 10 spws

            #wsu_bandwidth_final = np.append(wsu_bandwidth_final, 16.0)
            wsu_bandwidth_final.append(16.0)
            wsu_nspw_final.append(10)

            # but at beginning only band 6 and band 2 will be upgraded. Band 2 is under dev now, so no band 2 in cycle 7.
            band_list = np.unique(cycle7tab[idx]['band_list'])
            if len(band_list) > 1:
                print("multiple bands in same MOUS " + mymous)
            #band_list_array = np.append(band_list_array, band_list)
            band_list_array.append(band_list) ## is append going to cause problems here
            
            if band_list == 6:
                #wsu_bandwidth_initial = np.append(wsu_bandwidth_initial, 16.0)
                wsu_bandwidth_initial.append(16.0)
                wsu_nspw_initial.append(10)
            elif (band_list >= 3) & (band_list <= 8) & (band_list != 6):
                #wsu_bandwidth_initial = np.append(wsu_bandwidth_initial, 8.0)
                wsu_bandwidth_initial.append(8.0)
                wsu_nspw_initial.append(5)
            elif (band_list >= 9 & band_list <= 10):
                #wsu_bandwidth_initial = np.append(wsu_bandwidth_initial, 16.0)
                wsu_bandwidth_initial.append(16.0)
                wsu_nspw_initial.append(10)
            else:
                print('Band not recognized for MOUS: ' + mymous)


            
 
    # put appropriate units on quantities.
    s_fov_list = np.array(s_fov_list) * u.deg
    s_resolution_list = np.array(s_resolution_list) * u.arcsec


    blc_specwidth = np.array(blc_specwidth) * u.kHz
    blc_freq = np.array(blc_freq) * u.GHz
    blc_vel_res = np.array(blc_vel_res) * u.km / u.s
    blc_nchan_agg = np.array(blc_nchan_agg)
    blc_nchan_max = np.array(blc_nchan_max)
    blc_bw_max = np.array(blc_bw_max) * u.GHz
    blc_bw_agg = np.array(blc_bw_agg) * u.GHz
    blc_nspw = np.array(blc_nspw)
    
    wsu_bandwidth_initial = np.array(wsu_bandwidth_initial) * u.GHz
    wsu_bandwidth_final = np.array(wsu_bandwidth_final) * u.GHz
    wsu_bandwidth_spw = np.array(wsu_bandwidth_spw) * u.GHz

    wsu_specwidth_finest = np.array(wsu_specwidth_finest) * u.kHz
    wsu_specwidth_stepped = np.array(wsu_specwidth_stepped) * u.kHz
    wsu_specwidth_stepped2 = np.array(wsu_specwidth_stepped2) * u.kHz

    wsu_velres_finest = np.array(wsu_velres_finest) * u.km / u.s
    wsu_velres_stepped = np.array(wsu_velres_stepped) * u.km / u.s
    wsu_velres_stepped2 = np.array(wsu_velres_stepped2) * u.km / u.s
    
    wsu_freq_list = np.array(wsu_freq_list) * u.GHz
    wsu_tint_list = np.array(wsu_tint_list) * u.s

    #ipdb.set_trace()
    
    # put together table
    if_mous_tab = QTable([np.squeeze(if_mous_list), np.squeeze(proposal_id_list), np.squeeze(array_list), 
                          np.squeeze(nant_typical_list), np.squeeze(nant_array_list), np.squeeze(nant_all_list), 
                          np.squeeze(band_list_array), np.squeeze(ntarget_list), np.squeeze(target_name_list),
                          np.squeeze(s_fov_list), np.squeeze(s_resolution_list), np.squeeze(mosaic_list),
                          np.squeeze(imsize_list), np.squeeze(pb_list), np.squeeze(cell_list),
                          np.squeeze(npol_list),np.squeeze(blc_nspw),
                          np.squeeze(blc_specwidth),np.squeeze(blc_freq), np.squeeze(blc_vel_res),
                          np.squeeze(blc_nchan_agg),np.squeeze(blc_nchan_max),np.squeeze(blc_bw_max),np.squeeze(blc_bw_agg),
                          np.squeeze(wsu_freq_list),np.squeeze(wsu_npol_list),
                          np.squeeze(wsu_bandwidth_initial), np.squeeze(wsu_bandwidth_final), np.squeeze(wsu_bandwidth_spw),
                          np.squeeze(wsu_nspw_initial), np.squeeze(wsu_nspw_final),
                          np.squeeze(wsu_specwidth_finest), np.squeeze(wsu_chanavg_finest), np.squeeze(wsu_velres_finest),
                          np.squeeze(wsu_specwidth_stepped), np.squeeze(wsu_chanavg_stepped), np.squeeze(wsu_velres_stepped),
                          np.squeeze(wsu_specwidth_stepped2), np.squeeze(wsu_chanavg_stepped2), np.squeeze(wsu_velres_stepped2),
                          np.squeeze(wsu_tint_list)],
                         names=('mous','proposal_id','array',
                                'nant_typical','nant_array','nant_all',
                                'band','ntarget','target_name',
                                's_fov','s_resolution','mosaic',
                                'imsize','pb','cell',
                                'blc_npol','blc_nspw',
                                'blc_specwidth','blc_freq','blc_velres','blc_nchan_agg','blc_nchan_max','blc_bandwidth_max','blc_bandwidth_agg',
                                'wsu_freq','wsu_npol',
                                'wsu_bandwidth_initial','wsu_bandwidth_final','wsu_bandwidth_spw',
                                'wsu_nspw_initial','wsu_nspw_final',
                                'wsu_specwidth_finest','wsu_chanavg_finest', 'wsu_velres_finest',
                                'wsu_specwidth_stepped','wsu_chanavg_stepped', 'wsu_velres_stepped',
                                'wsu_specwidth_stepped2','wsu_chanavg_stepped2','wsu_velres_stepped2',                                
                                'wsu_tint'))
    
    
    # calculate number of channels per spw
    # ------------------------------------
        
    # figure out max allowed channels for 1.6 GHz spw
    nchan_max_talon_spw = 14880 * 8 # for 1.6 GHz spw with 8 FS
    nchan_max_spw_finest = np.floor(nchan_max_talon_spw / if_mous_tab['wsu_chanavg_finest']) # max channels if averaged
    nchan_max_spw_stepped = np.floor(nchan_max_talon_spw / if_mous_tab['wsu_chanavg_stepped']) # max channels if averaged
    nchan_max_spw_stepped2 = np.floor(nchan_max_talon_spw / if_mous_tab['wsu_chanavg_stepped2']) # max channels if averaged

    # calculate nchan for 1.6 GHz spw and finest channels
    if_mous_tab['wsu_nchan_spw_finest'] = np.floor((if_mous_tab['wsu_bandwidth_spw']/if_mous_tab['wsu_specwidth_finest']).decompose())
    # reduce nchan to less than max if necessary
    idx = if_mous_tab['wsu_nchan_spw_finest'] > nchan_max_spw_finest 
    if np.sum(idx) > 0:
        print("SPW BW, finest: Adjusting number of channels to meet TALON max: " + str(np.sum(idx)))
        #ipdb.set_trace()
        if_mous_tab['wsu_nchan_spw_finest'][idx] = nchan_max_spw_finest[idx] 

    # calculate nchan for 1.6 GHz spw and stepped channels    
    if_mous_tab['wsu_nchan_spw_stepped'] = np.floor((if_mous_tab['wsu_bandwidth_spw']/if_mous_tab['wsu_specwidth_stepped']).decompose())
    # reduce nchan to less than max if necessary
    idx = if_mous_tab['wsu_nchan_spw_stepped'] > nchan_max_spw_stepped 
    if np.sum(idx) > 0:
        print("SPW BW, stepped: Adjusting number of channels to meet TALON max: " + str(np.sum(idx)))
        #ipdb.set_trace()
        if_mous_tab['wsu_nchan_spw_stepped'][idx] = nchan_max_spw_stepped[idx] 
    
    # calculate nchan for 1.6 GHz spw and stepped2 channels
    if_mous_tab['wsu_nchan_spw_stepped2'] = np.floor((if_mous_tab['wsu_bandwidth_spw']/if_mous_tab['wsu_specwidth_stepped2']).decompose())
    idx = if_mous_tab['wsu_nchan_spw_stepped2'] > nchan_max_spw_stepped2 
    if np.sum(idx) > 0:
        print("SPW BW, stepped2: Adjusting number of channels to meet TALON max: " + str(np.sum(idx)))
        #ipdb.set_trace()
        if_mous_tab['wsu_nchan_spw_stepped2'][idx] = nchan_max_spw_stepped2[idx] 
    
    # fractional bandwidth
    # ---------------------
    
    if_mous_tab['frac_bw_initial'] = if_mous_tab['wsu_bandwidth_initial']/if_mous_tab['wsu_freq']
    if_mous_tab['frac_bw_final'] = if_mous_tab['wsu_bandwidth_final']/if_mous_tab['wsu_freq']
    if_mous_tab['frac_bw_spw'] = if_mous_tab['wsu_bandwidth_spw']/if_mous_tab['wsu_freq']

    
    ## TODO?
    ## move the calculations below to another bit of code that explicitly calculates data rates, visibility rates, and total number of visibilities?
        

    # # calculate number of baselines for each case.
    # # --------------------------------------------    

    # if_mous_tab['nbase_typical'] = if_mous_tab['nant_typical'] * (if_mous_tab['nant_typical'] -1 )/2.0
    # if_mous_tab['nbase_array'] = if_mous_tab['nant_array'] * (if_mous_tab['nant_array'] -1 )/2.0
    # if_mous_tab['nbase_all'] = if_mous_tab['nant_all'] * (if_mous_tab['nant_all'] - 1)/2.0

    # # calculate visibility rate (GVis/Hr)
    # # -----------------------------------
    
    # ## per SPW & typical number of antennas
    # if_mous_tab['vis_rate_typical_final_finest'] = (2.0 * if_mous_tab['npol'] * if_mous_tab['nbase_typical'] * if_mous_tab['wsu_nchan_spw_finest']  /1e9) / (if_mous_tab['tint'].to('hr'))
    # if_mous_tab['vis_rate_typical_final_stepped'] = (2.0 * if_mous_tab['npol'] * if_mous_tab['nbase_typical'] * if_mous_tab['wsu_nchan_spw_stepped']  /1e9) / (if_mous_tab['tint'].to('hr'))
    # if_mous_tab['vis_rate_typical_final_stepped2'] = (2.0 * if_mous_tab['npol'] * if_mous_tab['nbase_typical'] * if_mous_tab['wsu_nchan_spw_stepped2']  /1e9) / (if_mous_tab['tint'].to('hr'))
    
    # ## per SPW & all antennas in array
    # if_mous_tab['vis_rate_array_final_finest'] = (2.0 * if_mous_tab['npol'] * if_mous_tab['nbase_array'] * if_mous_tab['wsu_nchan_spw_finest']  /1e9) / (if_mous_tab['tint'].to('hr'))
    # if_mous_tab['vis_rate_array_final_stepped'] = (2.0 * if_mous_tab['npol'] * if_mous_tab['nbase_array'] * if_mous_tab['wsu_nchan_spw_stepped']  /1e9) / (if_mous_tab['tint'].to('hr'))
    # if_mous_tab['vis_rate_array_final_stepped2'] = (2.0 * if_mous_tab['npol'] * if_mous_tab['nbase_array'] * if_mous_tab['wsu_nchan_spw_stepped2']  /1e9) / (if_mous_tab['tint'].to('hr'))

    # ## per spw & all antennas together
    # if_mous_tab['vis_rate_all_final_finest'] = (2.0 * if_mous_tab['npol'] * if_mous_tab['nbase_all'] * if_mous_tab['wsu_nchan_spw_finest']  /1e9) / (if_mous_tab['tint'].to('hr'))
    # if_mous_tab['vis_rate_all_final_stepped'] = (2.0 * if_mous_tab['npol'] * if_mous_tab['nbase_all'] * if_mous_tab['wsu_nchan_spw_stepped']  /1e9) / (if_mous_tab['tint'].to('hr'))
    # if_mous_tab['vis_rate_all_final_stepped2'] = (2.0 * if_mous_tab['npol'] * if_mous_tab['nbase_all'] * if_mous_tab['wsu_nchan_spw_stepped2']  /1e9) / (if_mous_tab['tint'].to('hr'))
    

    return if_mous_tab


def add_tos_to_db(orig_db, tos_db):
    '''
    Purpose: Add time on source for sources and calibrators to data base. Needed
    for size of compute estimate.

    Date        Programmer      Description of Changes
    ----------------------------------------------------------------------
    11/25/2022  A.A. Kepley     Original Code
    '''

    new_db = join(orig_db,tos_db,keys=['mous','target_name','proposal_id','array','band','ntarget'], join_type='left')
    new_db_grouped = new_db.group_by('mous')
    
    return new_db_grouped

def calc_wsu_cal_tos():
    '''
    Purpose: Adjust calibrator TOS for the WSU

    Things to thing about:
    -- no changes -- just scale TOS by relevant factors (what are these)
    -- modest changes
             -- phase calibrator -- bigger changes (~1km/s)
             -- bandpass calibrator -- Do we need to do the full resolution here or can we average?
             -- check source -- ??
             -- polarization -- ??


    Need to review what's in the proposer's guide here to see what they are using now
    to see what might make sense initially.
    
    '''

    pass

def calc_nvis(mydb):
    '''
    Purpose: calculate number of visibilities produced for the observations.

    For now, just do for the 
    
    Date        Progammer       Description of Changes
    --------------------------------------------------
    11/25/2022  A.A. Kepley     Original Code
    '''

    ## target only
    ## ------------

    # This is what gets us what we need for the compute estimate. I think the spw version is probably
    
    
    # typical array & initial BW
    mydb['nvis_typical_initial_finest_target'] = mydb['vis_rate_typical_inital_finest'] * (mydb['target_time']/3600.0)
    mydb['nvis_typical_initial_stepped_target'] = mydb['vis_rate_typical_inital_stepped'] * (mydb['target_time']/3600.0)
    mydb['nvis_typical_initial_stepped2_target'] = mydb['vis_rate_typical_inital_stepped2'] * (mydb['target_time']/3600.0)

    # typical array & final BW
    mydb['nvis_typical_final_finest_target'] = mydb['vis_rate_typical_final_finest'] * (mydb['target_time']/3600.0)
    mydb['nvis_typical_final_stepped_target'] = mydb['vis_rate_typical_final_stepped'] * (mydb['target_time']/3600.0)
    mydb['nvis_typical_final_stepped2_target'] = mydb['vis_rate_typical_final_stepped2'] * (mydb['target_time']/3600.0)

    # typical array & spw
    mydb['nvis_typical_spw_finest_target'] = mydb['vis_rate_typical_spw_finest'] * (mydb['target_time']/3600.0)
    mydb['nvis_typical_spw_stepped_target'] = mydb['vis_rate_typical_spw_stepped'] * (mydb['target_time']/3600.0)
    mydb['nvis_typical_spw_stepped2_target'] = mydb['vis_rate_typical_spw_stepped2'] * (mydb['target_time']/3600.0)

    # whole array & initial BW
    mydb['nvis_array_initial_finest_target'] = mydb['vis_rate_array_inital_finest'] * (mydb['target_time']/3600.0)
    mydb['nvis_array_initial_stepped_target'] = mydb['vis_rate_array_inital_stepped'] * (mydb['target_time']/3600.0)
    mydb['nvis_array_initial_stepped2_target'] = mydb['vis_rate_array_inital_stepped2'] * (mydb['target_time']/3600.0)

    # whole array & final BW
    mydb['nvis_array_final_finest_target'] = mydb['vis_rate_array_final_finest'] * (mydb['target_time']/3600.0)
    mydb['nvis_array_final_stepped_target'] = mydb['vis_rate_array_final_stepped'] * (mydb['target_time']/3600.0)
    mydb['nvis_array_final_stepped2_target'] = mydb['vis_rate_array_final_stepped2'] * (mydb['target_time']/3600.0)

    # whole array & spw 
    mydb['nvis_array_spw_finest_target'] = mydb['vis_rate_array_spw_finest'] * (mydb['target_time']/3600.0)
    mydb['nvis_array_spw_stepped_target'] = mydb['vis_rate_array_spw_stepped'] * (mydb['target_time']/3600.0)
    mydb['nvis_array_spw_stepped2_target'] = mydb['vis_rate_array_spw_stepped2'] * (mydb['target_time']/3600.0)

    # all antennas & initial BW
    mydb['nvis_all_initial_finest_target'] = mydb['vis_rate_all_inital_finest'] * (mydb['target_time']/3600.0)
    mydb['nvis_all_initial_stepped_target'] = mydb['vis_rate_all_inital_stepped'] * (mydb['target_time']/3600.0)
    mydb['nvis_all_initial_stepped2_target'] = mydb['vis_rate_all_inital_stepped2'] * (mydb['target_time']/3600.0)

    # all antennas & final BW
    mydb['nvis_all_final_finest_target'] = mydb['vis_rate_all_final_finest'] * (mydb['target_time']/3600.0)
    mydb['nvis_all_final_stepped_target'] = mydb['vis_rate_all_final_stepped'] * (mydb['target_time']/3600.0)
    mydb['nvis_all_final_stepped2_target'] = mydb['vis_rate_all_final_stepped2'] * (mydb['target_time']/3600.0)

    # all antennas & spw
    mydb['nvis_all_spw_finest_target'] = mydb['vis_rate_all_spw_finest'] * (mydb['target_time']/3600.0)
    mydb['nvis_all_spw_stepped_target'] = mydb['vis_rate_all_spw_stepped'] * (mydb['target_time']/3600.0)
    mydb['nvis_all_spw_stepped2_target'] = mydb['vis_rate_all_spw_stepped2'] * (mydb['target_time']/3600.0)

    ## calibrators
    ## -----------

    ## TODO
    
    return mydb
    
    pass



