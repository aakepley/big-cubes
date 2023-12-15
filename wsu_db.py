from astropy.table import Table, QTable, join, unique, vstack
import numpy as np
import astropy.units as u
from astropy import constants as const
import math
import ipdb
import matplotlib.pyplot as plt

# define Gvis unit. if statement needed to cover reload.
# Gvis needs to be in globals to get everything to match in vstack up top.
# Otherwise it's not in the right name space (i.e., only in local namespace)
if 'gvis' not in globals():
    gvis = u.def_unit('Gvis',namespace=globals())
    u.add_enabled_units([gvis])


#wsu_chanavg_min[band][chanavg]
wsu_chanavg_min = {1:1.0, # band 1
                   2:2.0, # band 2 AKA band 2 (low)
                   3:3.0, # band 3 AKA band 2 (high)
                   4:5.0, # band 4
                   5:6.0, # band 5
                   6:8.0, # band 6
                   7:10.0, # band 7
                   8:15.0, # band 8
                   9:20.0, # band 9
                   10:32.0} # band 10
    
def calc_talon_specwidth(specwidth, band, blc_velres):
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

    # array case
    if type(specwidth) == np.ndarray:

        chan_avg = np.floor(specwidth / talon_chan)

        # fix up resulting channel averages
        chan_avg[chan_avg < 1.0] = 1.0 # can't go less than 1
        
        idx = (chan_avg < wsu_chanavg_min[band]) & (blc_velres > 0.095) & (blc_velres <= 0.5)
        chan_avg[idx] = wsu_chanavg_min[band] # if in the 0.1 to 0.5 bin, then set to the minimum.

        # calculate the spec width
        specwidth_talon = talon_chan * chan_avg
        
    # single value case
    else:        
        chan_avg = float(math.floor(specwidth / talon_chan))

        # fix up the resulting channel averages
        if chan_avg < 1.0:
            chan_avg = 1.0 # can't go less than 1
        if ((chan_avg < wsu_chanavg_min[band]) & (blc_velres > 0.095) & (blc_velres <= 0.5)):
            chan_avg = wsu_chanavg_min[band]
        
        specwidth_talon = talon_chan * chan_avg
            
    return specwidth_talon, chan_avg


def create_database(cycle7tab):
    '''
    
    Purpose: create database of cycle7 parameters for WSU size of computing estimate

    Date        Programmer      Description of Changes
    ----------------------------------------------------------------------
    7/2022?     A.A. Kepley     Original Code
    1/4/2023    A.A. Kepley     Updating with latest WSU terminology and additional BLC info
    '''
    
    # get MOUS list
    mousList = np.unique(cycle7tab['member_ous_uid'])
     
    # setup variables to hold values.
    #-----------------

    ## NOTE TO SELF: dictionary probably would have been better strategy here.
    
    # overall info
    if_mous_list = []
    proposal_id_list = []
    schedblock_name_list = []
    array_list = []
    science_keyword_list = []
    scientific_category_list = []

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
    wsu_bandwidth_later_2x = []
    wsu_bandwidth_later_4x = []
    wsu_bandwidth_early = []
    wsu_bandwidth_spw = []

    wsu_nspw_early = []
    wsu_nspw_later_2x = []
    wsu_nspw_later_4x = []

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


    ## NOTE: could potentially improve this by using group_by
    
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
            if_mous_list.append(mymous)
            
            # targetname
            target_name_list.append(target_name)
            
            # n targets
            ntarget_list.append(ntarget)
            
            # proposal id 
            proposal_id = np.unique(cycle7tab[idx]['proposal_id'])
            proposal_id_list.append(proposal_id)

            # scheduling block info
            schedblock_name = np.unique(cycle7tab[idx]['schedblock_name'])
            schedblock_name_list.append(schedblock_name)
            
            # array info & WSU tint
            array_list.append(array)
            
            if array == '12m':
                nant_typical = 47
                nant_array = 50 # corresponds to peak data rate calculations
                nant_all = 66 #12m+7m+TP
                #wsu_tint = 3.024 #s
                wsu_tint = 3.072 #s #updated value from Crystal on 2023/10/02
            elif array == '7m':
                nant_typical = 10
                nant_array = 12 # corresponds to peak data rate calculations
                nant_all = 16 # total power plus 7m
                #wsu_tint = 10.08 #s
                #wsu_tint = 10.24 #s # updated value from Crystal on 2023/10/12
                wsu_tint = 9.984 #s # updated value from Crystal on 2023/11/1

            nant_typical_list.append(nant_typical)
            nant_array_list.append(nant_array)
            nant_all_list.append(nant_all)
            wsu_tint_list.append(wsu_tint)
            
            # science keyword info
            ## These are the keywords selected by the PI
            science_keyword = np.unique(cycle7tab[idx]['science_keyword'])
            science_keyword_list.append(science_keyword)

            # scientific category
            ## These do not correspond to the 5 science categories exactly.
            ## Categories 1 (Cosmology and the high redshift universe) and 2 (galaxies and galactic nuclei)
            ## are separated depending on input keyword. I think I should be able to map to the
            ## five "official" categories based on the keywords.
            scientific_category = np.unique(cycle7tab[idx]['scientific_category'])
            scientific_category_list.append(scientific_category)

            # FOV
            s_fov = np.mean(cycle7tab[idx]['s_fov']) 
            s_fov_list.append(s_fov)
            
            # Resolution
            s_resolution = np.mean(cycle7tab[idx]['s_resolution'])
            s_resolution_list.append(s_resolution)
            
            # mosaic
            mosaic = np.unique(cycle7tab[idx]['is_mosaic'])
            if len(mosaic) > 1:
                print("mosaic and single pointings in same MOUS " + mymous + ". Setting mosaic to True")
                mosaic = 'T'
            mosaic_list.append(mosaic)
            
            # imsize
            imsize = np.mean(cycle7tab[idx]['imsize'])
            imsize_list.append(imsize)
            
            # pb
            pb = np.mean(cycle7tab[idx]['pb'])
            pb_list.append(pb)
            
            # cell
            cell = np.mean(cycle7tab[idx]['cell'])
            cell_list.append(cell)


            # BLC info
            # ---------

            # polarization states
            pol_states = np.unique(cycle7tab[idx]['pol_states'])
            if len(pol_states) > 1:
                print("print multiple polarization setups in same MOUS " + mymous)
            npol = len(pol_states.data[0].split('/')[1:-1])
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
            ## TODO -- check this assumption with Crystal.
            if npol == 1:
                wsu_npol = 2
            else:
                wsu_npol = npol
            wsu_npol_list.append(wsu_npol)


            # WSU band
            # ---------

            ## moved here because I need for TALON specwidth calculation
            ## to enforce nbin minimum.
            
            # get band information
            band = np.unique(cycle7tab[idx]['band_list'])
            if len(band) > 1:
                print("multiple bands in same MOUS " + mymous)
            band_list_array.append(band) ## is append going to cause problems here

            
            # WSU spectral resolution
            # -----------------------------
            
            # I believe that spec_width is what i want because that is the spectral 
            # resolution which is greater than the channel spacing for cases where 
            # averaging isn't happening for the channels
            
            ## finest
            (specwidth_finest_talon, chanavg_finest_talon) = calc_talon_specwidth(specwidth_finest, band[0], vel_res)
            wsu_specwidth_finest.append(specwidth_finest_talon)
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
            (specwidth_stepped_talon,chanavg_stepped_talon) = calc_talon_specwidth(specwidth_tmp, band[0], vel_res)
            wsu_specwidth_stepped.append(specwidth_stepped_talon)
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
            (specwidth_stepped2_talon,chanavg_stepped2_talon) = calc_talon_specwidth(specwidth_tmp, band[0], vel_res)
            wsu_specwidth_stepped2.append(specwidth_stepped2_talon)
            wsu_chanavg_stepped2.append(chanavg_stepped2_talon) 

            velres_stepped2_tmp = (specwidth_stepped2_talon * 1e3 / (freq*1e9)) * const.c.to('km/s').value
            wsu_velres_stepped2.append(velres_stepped2_tmp)
                                    
            # WSU BW
            # ------------

            #spw_bw = 1.6 # GHz             # based on 1st F at antenna
            spw_bw = 2.0 # GHz          # The 1st F is no more, so moving to 2 GHz.
            wsu_bandwidth_spw.append(spw_bw)

            # but at beginning only band 6 and band 2 will be upgraded. Band 2 is under dev now, so no band 2 in cycle 7.
            if band == 6:
                bw = 16.0
            elif band == 3: ## Looks like upper end of band 2 will be competitive with current band 3 so assuming that band 3 -> upper band 2
                bw = 16.0
            #elif (band >= 3) & (band <= 7) :
            elif (band >= 4) & (band <= 7) : ## Band 3 -> upper band 2
                bw = 8.0
            # adding band 8 to early WSU
            elif (band >= 8 & band <= 10): 
            #elif (band >= 9 & band <= 10):
                bw = 16.0
            else:
                print('Band not recognized for MOUS: ' + mymous)
                
            wsu_bandwidth_early.append(bw)
            wsu_nspw_early.append(round(bw/spw_bw))
            
            # 2x BW
            bw = 16.0 # GHz
            wsu_bandwidth_later_2x.append(bw)
            wsu_nspw_later_2x.append(round(bw/spw_bw))

            # 4x BW -- assumes band 1 won't be upgraded to 4x.
            if band == 1:
                bw = 16.0 # GHz
            else:
                bw = 32.0 # GHz
            wsu_bandwidth_later_4x.append(bw)
            wsu_nspw_later_4x.append(round(bw/spw_bw))                
 
    # put appropriate units on quantities.
    pb_list = np.array(pb_list) * u.arcsec
    cell_list = np.array(cell_list) * u.arcsec
    
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
    
    wsu_bandwidth_early = np.array(wsu_bandwidth_early) * u.GHz
    wsu_bandwidth_later_2x = np.array(wsu_bandwidth_later_2x) * u.GHz
    wsu_bandwidth_later_4x = np.array(wsu_bandwidth_later_4x) * u.GHz
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
    if_mous_tab = QTable([np.squeeze(if_mous_list), np.squeeze(proposal_id_list), np.squeeze(schedblock_name_list),np.squeeze(array_list), 
                          np.squeeze(science_keyword_list), np.squeeze(scientific_category_list),
                          np.squeeze(nant_typical_list), np.squeeze(nant_array_list), np.squeeze(nant_all_list), 
                          np.squeeze(band_list_array), np.squeeze(ntarget_list), np.squeeze(target_name_list),
                          np.squeeze(s_fov_list), np.squeeze(s_resolution_list), np.squeeze(mosaic_list),
                          np.squeeze(imsize_list), np.squeeze(pb_list), np.squeeze(cell_list),
                          np.squeeze(npol_list),np.squeeze(blc_nspw),
                          np.squeeze(blc_specwidth),np.squeeze(blc_freq), np.squeeze(blc_vel_res),
                          np.squeeze(blc_nchan_agg),np.squeeze(blc_nchan_max),np.squeeze(blc_bw_max),np.squeeze(blc_bw_agg),
                          np.squeeze(wsu_freq_list),np.squeeze(wsu_npol_list),
                          np.squeeze(wsu_bandwidth_early), np.squeeze(wsu_bandwidth_later_2x), np.squeeze(wsu_bandwidth_later_4x), np.squeeze(wsu_bandwidth_spw), 
                          np.squeeze(wsu_nspw_early), np.squeeze(wsu_nspw_later_2x), np.squeeze(wsu_nspw_later_4x),
                          np.squeeze(wsu_specwidth_finest), np.squeeze(wsu_chanavg_finest), np.squeeze(wsu_velres_finest),
                          np.squeeze(wsu_specwidth_stepped), np.squeeze(wsu_chanavg_stepped), np.squeeze(wsu_velres_stepped),
                          np.squeeze(wsu_specwidth_stepped2), np.squeeze(wsu_chanavg_stepped2), np.squeeze(wsu_velres_stepped2),
                          np.squeeze(wsu_tint_list)],
                         names=('mous','proposal_id','schedblock_name','array',
                                'science_keyword','scientific_category',
                                'nant_typical','nant_array','nant_all',
                                'band','ntarget','target_name',
                                's_fov','s_resolution','mosaic',
                                'imsize','pb','cell',
                                'blc_npol','blc_nspw',
                                'blc_specwidth','blc_freq','blc_velres',
                                'blc_nchan_agg','blc_nchan_max','blc_bandwidth_max','blc_bandwidth_agg',
                                'wsu_freq','wsu_npol',
                                'wsu_bandwidth_early','wsu_bandwidth_later_2x','wsu_bandwidth_later_4x','wsu_bandwidth_spw',
                                'wsu_nspw_early','wsu_nspw_later_2x', 'wsu_nspw_later_4x',
                                'wsu_specwidth_finest','wsu_chanavg_finest', 'wsu_velres_finest',
                                'wsu_specwidth_stepped','wsu_chanavg_stepped', 'wsu_velres_stepped',
                                'wsu_specwidth_stepped2','wsu_chanavg_stepped2','wsu_velres_stepped2',                              
                                'wsu_tint'))
    
    
    # calculate number of channels per spw
    # ------------------------------------
        
    # figure out max allowed channels for 1.6 GHz spw
    #nchan_max_talon_spw = 14880 * 8 # for 1.6 GHz spw with 8 FS
    nchan_max_talon_spw = 14880 * 10 # for 2.0 GHz spw with 10 FS
    nchan_max_spw_finest = np.floor(nchan_max_talon_spw / if_mous_tab['wsu_chanavg_finest']) # max channels if averaged
    nchan_max_spw_stepped = np.floor(nchan_max_talon_spw / if_mous_tab['wsu_chanavg_stepped']) # max channels if averaged
    nchan_max_spw_stepped2 = np.floor(nchan_max_talon_spw / if_mous_tab['wsu_chanavg_stepped2']) # max channels if averaged

    # calculate nchan for spw and finest channels
    if_mous_tab['wsu_nchan_spw_finest'] = np.floor((if_mous_tab['wsu_bandwidth_spw']/if_mous_tab['wsu_specwidth_finest']).decompose())
    # reduce nchan to less than max if necessary
    idx = if_mous_tab['wsu_nchan_spw_finest'] > nchan_max_spw_finest 
    if np.sum(idx) > 0:
        print("SPW BW, finest: Adjusting number of channels to meet TALON max: " + str(np.sum(idx)))
        #ipdb.set_trace()
        if_mous_tab['wsu_nchan_spw_finest'][idx] = nchan_max_spw_finest[idx] 

    # calculate nchan for spw and stepped channels    
    if_mous_tab['wsu_nchan_spw_stepped'] = np.floor((if_mous_tab['wsu_bandwidth_spw']/if_mous_tab['wsu_specwidth_stepped']).decompose())
    # reduce nchan to less than max if necessary
    idx = if_mous_tab['wsu_nchan_spw_stepped'] > nchan_max_spw_stepped 
    if np.sum(idx) > 0:
        print("SPW BW, stepped: Adjusting number of channels to meet TALON max: " + str(np.sum(idx)))
        #ipdb.set_trace()
        if_mous_tab['wsu_nchan_spw_stepped'][idx] = nchan_max_spw_stepped[idx] 
    
    # calculate nchan for spw and stepped2 channels
    if_mous_tab['wsu_nchan_spw_stepped2'] = np.floor((if_mous_tab['wsu_bandwidth_spw']/if_mous_tab['wsu_specwidth_stepped2']).decompose())
    idx = if_mous_tab['wsu_nchan_spw_stepped2'] > nchan_max_spw_stepped2 
    if np.sum(idx) > 0:
        print("SPW BW, stepped2: Adjusting number of channels to meet TALON max: " + str(np.sum(idx)))
        #ipdb.set_trace()
        if_mous_tab['wsu_nchan_spw_stepped2'][idx] = nchan_max_spw_stepped2[idx] 

    ## Calculate the aggregate number of channels
    ## ------------------------------------------
    for veltype in ['finest','stepped','stepped2']:
        if_mous_tab['wsu_nchan_agg_'+veltype+'_early'] = if_mous_tab['wsu_nchan_spw_'+veltype] * if_mous_tab['wsu_nspw_early']
        if_mous_tab['wsu_nchan_agg_'+veltype+'_later_2x'] = if_mous_tab['wsu_nchan_spw_'+veltype] * if_mous_tab['wsu_nspw_later_2x']
        if_mous_tab['wsu_nchan_agg_'+veltype+'_later_4x'] = if_mous_tab['wsu_nchan_spw_'+veltype] * if_mous_tab['wsu_nspw_later_4x']
        
    # fractional bandwidth
    # ---------------------
    
    if_mous_tab['wsu_frac_bw_early'] = if_mous_tab['wsu_bandwidth_early']/if_mous_tab['wsu_freq']
    if_mous_tab['wsu_frac_bw_later_2x'] = if_mous_tab['wsu_bandwidth_later_2x']/if_mous_tab['wsu_freq']
    if_mous_tab['wsu_frac_bw_later_4x'] = if_mous_tab['wsu_bandwidth_later_4x']/if_mous_tab['wsu_freq']
    if_mous_tab['wsu_frac_bw_spw'] = if_mous_tab['wsu_bandwidth_spw']/if_mous_tab['wsu_freq']
    

    # calculate number of baselines for each case.
    # --------------------------------------------    

    for myarray in ['typical','array','all']:
        if_mous_tab['nbase_'+myarray] = if_mous_tab['nant_'+myarray] * (if_mous_tab['nant_'+myarray] -1 )/2.0    
        
    return if_mous_tab


def create_tp_database(cycle7tab):
    '''
    Purpose: create database of parameters for WSU data rate estimate for
    total power data. Requested by Bunyo

    Date        Programmer      Description of Changes
    --------------------------------------------------
    11/20/2023  A.A. Kepley     Original Code
    '''

    # get MOUS list
    mousList = np.unique(cycle7tab['member_ous_uid'])
     
    # setup variables to hold values.
    #-----------------

    ## NOTE TO SELF: dictionary probably would have been better strategy here.
    
    # overall info
    if_mous_list = []
    proposal_id_list = []
    schedblock_name_list = []
    array_list = []
    science_keyword_list = []
    scientific_category_list = []

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
    wsu_bandwidth_later_4x = []
    wsu_bandwidth_early = []
    wsu_bandwidth_spw = []

    wsu_nspw_early = []
    wsu_nspw_later_4x = []

    wsu_freq_list = []

    wsu_specwidth_stepped2 = []
    wsu_chanavg_stepped2 = []
    wsu_velres_stepped2 = []

    # fill in values
    for mymous in mousList:
        idx_mous = cycle7tab['member_ous_uid'] == mymous

        array = np.unique(cycle7tab[idx_mous]['array'])

        # skip 12m and 7m
        if array != 'TP':
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
            if_mous_list.append(mymous)
            
            # targetname
            target_name_list.append(target_name)
            
            # n targets
            ntarget_list.append(ntarget)
            
            # proposal id 
            proposal_id = np.unique(cycle7tab[idx]['proposal_id'])
            proposal_id_list.append(proposal_id)

            # scheduling block info
            schedblock_name = np.unique(cycle7tab[idx]['schedblock_name'])
            schedblock_name_list.append(schedblock_name)
            
            # array info 
            array_list.append(array)

            # science keyword info
            ## These are the keywords selected by the PI
            science_keyword = np.unique(cycle7tab[idx]['science_keyword'])
            science_keyword_list.append(science_keyword)

            # scientific category
            ## These do not correspond to the 5 science categories exactly.
            ## Categories 1 (Cosmology and the high redshift universe) and 2 (galaxies and galactic nuclei)
            ## are separated depending on input keyword. I think I should be able to map to the
            ## five "official" categories based on the keywords.
            scientific_category = np.unique(cycle7tab[idx]['scientific_category'])
            scientific_category_list.append(scientific_category)

            # FOV
            s_fov = np.mean(cycle7tab[idx]['s_fov']) 
            s_fov_list.append(s_fov)
            
            # Resolution
            s_resolution = np.mean(cycle7tab[idx]['s_resolution'])
            s_resolution_list.append(s_resolution)
            
            # mosaic
            mosaic = np.unique(cycle7tab[idx]['is_mosaic'])
            if len(mosaic) > 1:
                print("mosaic and single pointings in same MOUS " + mymous + ". Setting mosaic to True")
                mosaic = 'T'
            mosaic_list.append(mosaic)
            
            # imsize
            imsize = np.mean(cycle7tab[idx]['imsize'])
            imsize_list.append(imsize)
            
            # pb
            pb = np.mean(cycle7tab[idx]['pb'])
            pb_list.append(pb)
            
            # cell
            cell = np.mean(cycle7tab[idx]['cell'])
            cell_list.append(cell)
            
            # BLC info
            # ---------

            # polarization states
            pol_states = np.unique(cycle7tab[idx]['pol_states'])
            if len(pol_states) > 1:
                print("print multiple polarization setups in same MOUS " + mymous)
            npol = len(pol_states.data[0].split('/')[1:-1])
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
            ## TODO -- check this assumption with Crystal.
            if npol == 1:
                wsu_npol = 2
            else:
                wsu_npol = npol
            wsu_npol_list.append(wsu_npol)

            ## WSU band
            # -----------
            
            ## moved here because I need for the TALON specwidth
            ## calculation to enforce nbin minimum
                        
            # get band information
            band_list = np.unique(cycle7tab[idx]['band_list'])
            if len(band_list) > 1:
                print("multiple bands in same MOUS " + mymous)
            band_list_array.append(band_list) ## is append going to cause problems here


            # WSU spectral resolution
            # -----------------------------

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
            (specwidth_stepped2_talon,chanavg_stepped2_talon) = calc_talon_specwidth(specwidth_tmp, band_list[0], vel_res)
            wsu_specwidth_stepped2.append(specwidth_stepped2_talon)
            wsu_chanavg_stepped2.append(chanavg_stepped2_talon) 

            velres_stepped2_tmp = (specwidth_stepped2_talon * 1e3 / (freq*1e9)) * const.c.to('km/s').value
            wsu_velres_stepped2.append(velres_stepped2_tmp)

            # WSU BW
            # ------------
            

            #spw_bw = 1.6 # GHz             # based on 1st F at antenna
            spw_bw = 2.0 # GHz          # The 1st F is no more, so moving to 2 GHz.
            wsu_bandwidth_spw.append(spw_bw)

            # but at beginning only band 6 and band 2 will be upgraded. Band 2 is under dev now, so no band 2 in cycle 7.
            if band_list == 6:
                bw = 16.0
            elif band_list == 3: ## Looks like upper end of band 2 will be competitive with current band 3 so assuming that band 3 -> upper band 2
                bw = 16.0
            #elif (band_list >= 3) & (band_list <= 7) :
            elif (band_list >= 4) & (band_list <= 7) : ## Band 3 -> upper band 2
                bw = 8.0
            # adding band 8 to early WSU
            elif (band_list >= 8 & band_list <= 10): 
            #elif (band_list >= 9 & band_list <= 10):
                bw = 16.0
            else:
                print('Band not recognized for MOUS: ' + mymous)
                
            wsu_bandwidth_early.append(bw)
            wsu_nspw_early.append(round(bw/spw_bw))
            
            # 4x BW -- assumes band 1 won't be upgraded to 4x.
            if band_list == 1:
                bw = 16.0 # GHz
            else:
                bw = 32.0 # GHz
            wsu_bandwidth_later_4x.append(bw)
            wsu_nspw_later_4x.append(round(bw/spw_bw))                

    # put appropriate units on quantities.
    pb_list = np.array(pb_list) * u.arcsec
    cell_list = np.array(cell_list) * u.arcsec
    
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
    
    wsu_bandwidth_early = np.array(wsu_bandwidth_early) * u.GHz
    wsu_bandwidth_later_4x = np.array(wsu_bandwidth_later_4x) * u.GHz
    wsu_bandwidth_spw = np.array(wsu_bandwidth_spw) * u.GHz

    wsu_specwidth_stepped2 = np.array(wsu_specwidth_stepped2) * u.kHz

    wsu_velres_stepped2 = np.array(wsu_velres_stepped2) * u.km / u.s
    
    wsu_freq_list = np.array(wsu_freq_list) * u.GHz

    # put together table
    if_mous_tab = QTable([np.squeeze(if_mous_list), np.squeeze(proposal_id_list), np.squeeze(schedblock_name_list),np.squeeze(array_list), 
                          np.squeeze(science_keyword_list), np.squeeze(scientific_category_list),

                          np.squeeze(band_list_array), np.squeeze(ntarget_list), np.squeeze(target_name_list),
                          np.squeeze(s_fov_list), np.squeeze(s_resolution_list), np.squeeze(mosaic_list),
                          np.squeeze(imsize_list), np.squeeze(pb_list), np.squeeze(cell_list),
                          np.squeeze(npol_list),np.squeeze(blc_nspw),
                          np.squeeze(blc_specwidth),np.squeeze(blc_freq), np.squeeze(blc_vel_res),
                          np.squeeze(blc_nchan_agg),np.squeeze(blc_nchan_max),np.squeeze(blc_bw_max),np.squeeze(blc_bw_agg),
                          np.squeeze(wsu_freq_list),np.squeeze(wsu_npol_list),
                          np.squeeze(wsu_bandwidth_early), np.squeeze(wsu_bandwidth_later_4x), np.squeeze(wsu_bandwidth_spw), 
                          np.squeeze(wsu_nspw_early), np.squeeze(wsu_nspw_later_4x),
                          np.squeeze(wsu_specwidth_stepped2), np.squeeze(wsu_chanavg_stepped2), np.squeeze(wsu_velres_stepped2)],
                         names=('mous','proposal_id','schedblock_name','array',
                                'science_keyword','scientific_category',
                                'band','ntarget','target_name',
                                's_fov','s_resolution','mosaic',
                                'imsize','pb','cell',
                                'blc_npol','blc_nspw',
                                'blc_specwidth','blc_freq','blc_velres',
                                'blc_nchan_agg','blc_nchan_max','blc_bandwidth_max','blc_bandwidth_agg',
                                'wsu_freq','wsu_npol',
                                'wsu_bandwidth_early','wsu_bandwidth_later_4x','wsu_bandwidth_spw',
                                'wsu_nspw_early', 'wsu_nspw_later_4x',
                                'wsu_specwidth_stepped2','wsu_chanavg_stepped2','wsu_velres_stepped2'))                              
    

    # calculate number of channels per spw
    # ------------------------------------
        
    # figure out max allowed channels for 1.6 GHz spw
    #nchan_max_talon_spw = 14880 * 8 # for 1.6 GHz spw with 8 FS
    nchan_max_talon_spw = 14880 * 10 # for 2.0 GHz spw with 10 FS
    nchan_max_spw_stepped2 = np.floor(nchan_max_talon_spw / if_mous_tab['wsu_chanavg_stepped2']) # max channels if averaged
    
    # calculate nchan for spw and stepped2 channels
    if_mous_tab['wsu_nchan_spw_stepped2'] = np.floor((if_mous_tab['wsu_bandwidth_spw']/if_mous_tab['wsu_specwidth_stepped2']).decompose())
    idx = if_mous_tab['wsu_nchan_spw_stepped2'] > nchan_max_spw_stepped2 
    if np.sum(idx) > 0:
        print("SPW BW, stepped2: Adjusting number of channels to meet TALON max: " + str(np.sum(idx)))
        #ipdb.set_trace()
        if_mous_tab['wsu_nchan_spw_stepped2'][idx] = nchan_max_spw_stepped2[idx] 


    ## Calculate the aggregate number of channels
    ## ------------------------------------------
    #for veltype in ['finest','stepped','stepped2']:
    for veltype in ['stepped2']:
        if_mous_tab['wsu_nchan_agg_'+veltype+'_early'] = if_mous_tab['wsu_nchan_spw_'+veltype] * if_mous_tab['wsu_nspw_early']
        if_mous_tab['wsu_nchan_agg_'+veltype+'_later_4x'] = if_mous_tab['wsu_nchan_spw_'+veltype] * if_mous_tab['wsu_nspw_later_4x']
        
    # fractional bandwidth
    # ---------------------
    
    if_mous_tab['wsu_frac_bw_early'] = if_mous_tab['wsu_bandwidth_early']/if_mous_tab['wsu_freq']
    if_mous_tab['wsu_frac_bw_later_4x'] = if_mous_tab['wsu_bandwidth_later_4x']/if_mous_tab['wsu_freq']
    if_mous_tab['wsu_frac_bw_spw'] = if_mous_tab['wsu_bandwidth_spw']/if_mous_tab['wsu_freq']
    

    return if_mous_tab

    
def add_blc_ntunings(orig_db, ntunings_file):
    '''
    Purpose: add number of BLC tunings to data base

    Date        Programmer      Description of Changes
    ----------------------------------------------------------------------
    1/13/2023   A.A. Kepely     Original Code
    '''

    ntunings_db = Table.read(ntunings_file,encoding='utf-8-sig')

    orig_db['blc_ntunings'] =  np.ones(len(orig_db))

    match = False
    
    for line in ntunings_db:
        idx = (line['proposal_id'] == orig_db['proposal_id']) & (line['sb_name'] == orig_db['schedblock_name'])

        if np.any(idx):
            print('Match found for ' + line['proposal_id']  + ', ' + line['sb_name'])
            orig_db['blc_ntunings'][idx] = line['n_tunings']            
            match = True

    if not match:
        print('No matches found. Are you using the right file')
        

        
def add_l80(orig_db,l80_file=None):
    '''
    Purpose: add L80 to data base

    Date        Programmer      Description of Changes
    ----------------------------------------------------------------------
    1/11/2023   A.A. Kepley     Original Code
    '''

    # read in l80 file
    if not bool(l80_file):
        print("Need to give L80 file")
        return
    
    l80_tab = Table.read(l80_file)
    l80_tab.rename_column('Member ous id','mous')
    l80_tab.rename_column('L80 BL','L80')
    l80_tab.rename_column('ALMA source name','target_name')
    
    new_db = join(orig_db,l80_tab,keys=('mous','target_name'),join_type='left')
    new_db['L80'].unit = u.m
    new_db.remove_column('Project code')
    new_db.remove_column('Array')

    return new_db

    
def add_blc_tint(orig_db, breakpt_12m=3000.0 * u.m):
    '''
    Purpose: Add baseline correlator integration time

    Date        Programmer      Description of Changes
    --------------------------------------------------
    1/11/2023   A.A. Kepley     Original Code   
    
    '''

    # default tint values
    tint_7m = 10.1 #s
    tint_12m_short = 6.05 #s
    tint_12m_long = 3.024 #s ## Matches IST data rate memo assumptions. also see values of 2.02 for some projects. 
    
    orig_db['blc_tint'] = np.ones(len(orig_db)) * u.s

    # set 7m
    idx = orig_db['array'] == '7m'
    orig_db['blc_tint'][idx] = tint_7m * orig_db['blc_tint'][idx] 
    
    # add 12m values
    idx = (orig_db['array'] == '12m') & (orig_db['L80'] > breakpt_12m)
    orig_db['blc_tint'][idx] = tint_12m_long * orig_db['blc_tint'][idx]

    idx = (orig_db['array'] == '12m') & (orig_db['L80'] <= breakpt_12m )
    orig_db['blc_tint'][idx] = tint_12m_short * orig_db['blc_tint'][idx]


def add_blc_tint_from_db(orig_db, csvfile):
    '''
    Purpose: add actual BLC tint from csv file with info

    Date        Programmer      Description of Changes
    ----------------------------------------------------------------------
    1/28/2023   A.A. Kepley     Original Code
    '''

    tint_db = Table.read('data/SB_MetaData_C7_C8_2023Jan27.csv',encoding='utf-8-sig')

    new_tab = join(orig_db, tint_db['Project','SB_name','Integration'],
                   join_type='left',
                   keys_left=('proposal_id','schedblock_name'),keys_right=('Project','SB_name'))

    if 'blc_tint' in new_tab.columns:
        new_tab.remove_column('blc_tint')
        
    new_tab.rename_column('Integration','blc_tint')
    new_tab['blc_tint'].unit = u.s

    
    return new_tab

    

    

def add_tos_to_db(orig_db, tos_db):
    '''
    Purpose: Add time on source for sources and calibrators to data base. Needed
    for size of compute estimate.

    Date        Programmer      Description of Changes
    ----------------------------------------------------------------------
    11/25/2022  A.A. Kepley     Original Code
    '''

    new_db = join(orig_db,tos_db,keys=['mous','target_name','proposal_id','array','band','ntarget'], join_type='left')
    #new_db_grouped = new_db.group_by('mous') ## IS THIS DOING ANYTHING??
    
    #return new_db_grouped
    return new_db
    
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

def add_rates_to_db(mydb, wsu_only=False, permous=False):
    '''
    Purpose: Add data rates and associated quantities to data base

    Date        Programmer      Description of Changes
    ----------------------------------------------------------------------
    1/4/2023    A.A. Kepley     Original Code
    '''

    from large_cubes import calc_mfs_size, calc_cube_size
    
    # calculate mfs size
    # This will not change with WSU
    mydb['mfssize'] = calc_mfs_size(mydb['imsize'])

    array_list = ['typical','array']
    #array_list = ['typical']
    #velres_list = ['stepped','stepped2']
    velres_list = ['stepped2']

    # if per mous/source, don't want to multiple by ntarget
    # if per mous, do want to multiply by ntarget
    if permous:
        ntarget = mydb['ntarget']
    else:
        ntarget = np.full(len(mydb),1.0)
        
    # calculate cube sizes, data rates, and data volumes.

    for velres in velres_list:

        # calculate cube sizes
        # only depends on number of channels per spw
        mydb['wsu_cubesize_'+velres] = calc_cube_size(mydb['imsize'], mydb['wsu_nchan_spw_'+velres])

        for stage in ['early','later_2x','later_4x']:
            
            # calculate product size
            # depends on number of cubes
            mydb['wsu_productsize_'+stage+'_'+velres] = 2.0 * (mydb['wsu_cubesize_'+velres] + mydb['mfssize']) * mydb['wsu_nspw_'+stage] * ntarget

            for array in array_list:
                # calculate data rates and data volumes for the visibilities
                calc_rates_for_db(mydb,array=array,correlator='wsu',stage=stage,velres=velres,permous=permous)                

                if not wsu_only:
                    # calculate BLC correlator values
                    calc_rates_for_db(mydb,array=array,correlator='blc',stage='',velres='')

    if not wsu_only:

        # calculate cube sizes
        mydb['blc_cubesize'] = calc_cube_size(mydb['imsize'], mydb['blc_nchan_max'])
        mydb['blc_cubesize_sum'] = calc_cube_size(mydb['imsize'], mydb['blc_nchan_agg'])
        mydb['blc_productsize'] = 2.0 * (mydb['blc_cubesize_sum'] + mydb['mfssize']  * mydb['blc_nspw'])
        
                
    
def calc_rates_for_db(mydb, array='typical',correlator='wsu',stage='early', velres='stepped2', permous=False):
    '''
    Purpose: calculate data rates for a specific case
    
    Date        Progammer       Description of Changes
    ----------------------------------------------------------------------
    1/4/2023    A.A. Kepley     Original Code
    
    '''

    
    Nbyte = 2.0 # cross-corrs
    Napc = 1.0 # offline WVR correlators
    Nant = mydb['nant_'+array]

    if correlator == 'wsu':
        Nspws = mydb[correlator+'_nspw_'+stage]
        Nchan_per_spw = mydb[correlator+'_nchan_spw_'+velres]
        Nchannels = Nspws * Nchan_per_spw
        mylabel = stage+'_'+velres+'_'+array
    elif correlator == 'blc':
        # use aggregate number of channels. Need to divide by number of tunings to take care of spectral
        # scan case.
        Nchannels = mydb[correlator+'_nchan_agg'] /  mydb[correlator+'_ntunings']
        mylabel = array
    else:
        print('Correlator name not found: ' + correlator)
                                        
        
    Tintegration = mydb[correlator+'_tint']

    Npols = mydb[correlator+'_npol']
    
    mydb[correlator+'_datarate_'+mylabel] = calc_datarate(Nbyte, Napc, Nant, Nchannels, Npols, Tintegration) #GB/s
    mydb[correlator+'_visrate_'+mylabel] = calc_visrate(Nant, Npols, Nchannels, Tintegration)  #Gvis/hr

    if not permous:
        mydb[correlator+'_datavol_'+mylabel+'_target'] = mydb[correlator+'_datarate_'+mylabel] * mydb['target_time']  # GB/s * s = GB
    mydb[correlator+'_datavol_'+mylabel+'_target_tot'] = mydb[correlator+'_datarate_'+mylabel] * mydb['target_time_tot']  # GB/s * s = GB

    mydb[correlator+'_datavol_'+mylabel+'_cal'] = mydb[correlator+'_datarate_'+mylabel] * mydb['cal_time'] # GB/s * s = GB
    mydb[correlator+'_datavol_'+mylabel+'_total'] = mydb[correlator+'_datarate_'+mylabel] * mydb['time_tot'] # GB/s * s = GB

    if not permous:
        mydb[correlator+'_nvis_'+mylabel+'_target'] = mydb[correlator+'_visrate_'+mylabel] * (mydb['target_time'].to(u.hr))  # Gvis/hr * hr = Gvis
    mydb[correlator+'_nvis_'+mylabel+'_target_tot'] = mydb[correlator+'_visrate_'+mylabel] * (mydb['target_time_tot'].to(u.hr)) # Gvis/hr * hr = Gvis
        
    mydb[correlator+'_nvis_'+mylabel+'_cal'] = mydb[correlator+'_visrate_'+mylabel]  * (mydb['cal_time'].to(u.hr)) # Gvis/hr * hr = Gvis
    mydb[correlator+'_nvis_'+mylabel+'_total'] = mydb[correlator+'_visrate_'+mylabel] * (mydb['time_tot'].to(u.hr))# Gvis/hr * hr = Gvis
    

def calc_datarate(Nbyte, Napc, Nant, Nchannels, Npols, Tintegration):
    '''
    Purpose: calculate data rate based on the following equation:


    Output Data Rate = (( 2 Nbyte x Napc x Nant(Nant-1)/2 + 4 Nant ) x Nchannels x Npols) / Tintegration

    Nbyte = 2 for cross-corrs (16-bit) and Nbyte = 4 for autocorrs (32-bit) -- assume Nbyte = 2
    Napc = number of WVR streams = 1
    Nant = number of antennas
    Nchannels = number of channels = nspws * nchan_per_spw
    Npols = number of polarizations
    Tintegration = visibility integration time = 3.024s for 12m and 10.08 for 7m

    Date        Programmer      Description of Changes
    --------------------------------------------------
    1/4/2023    A.A. Kepley     Original Code    
    '''

    datarate = (( 2.0 * Nbyte * Napc * Nant * (Nant-1.0)/2.0 + 4 * Nant) * Nchannels * Npols) * u.GB / Tintegration / 1e9 # GB/s
    
    return datarate
    

def calc_visrate(Nant, Npols, Nchannels, Tintegration):
    '''
    Purpose: calculate the visibility rate for each line in the dta base

    Tintegration is assumed to be in seconds
    
    Date        Programmer      Description of Changes
    ----------------------------------------------------
    1/4/2023    A.A. Kepley     Original Code
    '''

    Nbase =  Nant * (Nant-1.0)/2.0

    visrate = (2.0 * Npols * Nbase * Nchannels /1e9) * gvis / (Tintegration.to(u.hr)) # GVis/Hr

    return visrate
    

def calc_frac_time(mydb, cycle='c7'):
    '''
    Purpose: calculate fraction time for each source

    Each row in input table is mous/source

    To get MOUS time would have to group by MOUS and do some table magic.

    Date        Programmer      Description of Changes
    --------------------------------------------------
    1/9/2023    A.A. Kepley     Original Code
    '''

    mydb['frac_'+cycle+'_target_time'] = mydb['target_time'] / np.sum(mydb['target_time']) # per source

    
def create_per_mous_db(mydb):
    '''

    Purpose: create a per mous table. Fancy way using table
    groupings failed, so doing this the old fashioned way with a loop.

    Input: data base with mous/src per line

    Output: data base with mous per line
    

    Date        Programmer      Description of Changes
    ----------------------------------------------------------------------
    1/10/2023   A.A. Kepley     Original Code
    '''

    from statistics import mode
    import re


    # get groups
    mydb_by_mous = mydb.group_by('mous')

    # create output dictionary
    newdb_dict = {}
    for mykey in mydb_by_mous.keys():
        newdb_dict[mykey] = []

    # add variable to turn off messages after first round.
    keymsg = True

    # iterate over groups and calculate aggregate values
    for mygroup in mydb_by_mous.groups:

        
        for mykey in mygroup.keys():
            
            # take max
            if ((mykey in ['s_fov','s_resolution', 'imsize','pb','mfssize']) or
                (re.match('wsu_cubesize',mykey)) or
                (re.match('wsu_nchan',mykey)) or
                (re.match('wsu_datarate',mykey)) or
                (re.match('wsu_visrate',mykey))):
                
                myval = np.max(mygroup[mykey])
                newdb_dict[mykey].append(myval)

            # take min
            elif ((mykey in ['cell','blc_specwidth','blc_velres']) or
                (re.match('wsu_specwidth',mykey)) or
                (re.match('wsu_velres',mykey))):
                myval = np.min(mygroup[mykey])
                newdb_dict[mykey].append(myval)
                    
            # take sum
            elif (re.match('wsu_productsize',mykey)):
                myval = np.sum(mygroup[mykey])
                newdb_dict[mykey].append(myval)

            elif (re.match('blc_productsize',mykey)):
                myval = np.sum(mygroup[mykey])
                newdb_dict[mykey].append(myval)

                
            # take mean
            elif mykey in ['blc_freq','wsu_freq']:
                myval = np.mean(mygroup[mykey])
                newdb_dict[mykey].append(myval)

            # take mode
            elif (re.match('wsu_chanavg',mykey)):
                myval = mode(mygroup[mykey])
                newdb_dict[mykey].append(myval)

            # take first value
            else:
                if keymsg:
                    print('Taking first value. Key aggregation not specified: ' + mykey)

                newdb_dict[mykey].append(mygroup[mykey][0])

        # don't display message after first group since all groups are the same.
        keymsg = False


    # create dictionary
    mous_db = QTable(newdb_dict)

    # remove target specific keys because they aren't relevant to a per mous data base
    for mykey in mous_db.keys():
        if re.search('_target$',mykey):
            mous_db.remove_column(mykey)
            
        if mykey == 'target_time':
            mous_db.remove_column(mykey)

        if mykey == 'target_name':
            mous_db.remove_column(mykey)

    # recalculate target total
            
    return mous_db

def aggregate_tdump_db(mydb):
    '''
    Purpose: aggregate the tdump time db for Bunyo

    Input: original data base

    Output: data base with aggregated quantities.

    Date        Programmer      Description of Changes
    --------------------------------------------------
    11/21/2023  A.A. Kepley     Original Code

    '''

    from statistics import mode
    import re

    # get groups
    mydb_by_sb = mydb.group_by(['PRJ_CODE','SB_NAME'])

    # set list of keys I want
    keylist = ['PRJ_CODE','SB_NAME','CYCLE',
               'ESTIMATED_EXECUTION_TIME_PRJ','ESTIMATED_EXECUTION_TIME_PRJ_UNIT',
               'ACASPEC_INTEGRATION_DURATION','EXECOUNT','N_TDUMP']    

    # create output dictionary. only need some keys
    newdb_dict = {}
    for mykey in keylist:        
        newdb_dict[mykey] = []

    for mygroup in mydb_by_sb.groups:
        for mykey in keylist:
            if (mykey in ['SB_NAME','CYCLE',
                          'ESTIMATED_EXECUTION_TIME_PRJ',
                          'ESTIMATED_EXECUTION_TIME_PRJ_UNIT',
                          'EXECOUNT',
                          'PRJ_CODE']):

                myval = np.unique(mygroup[mykey])

                if len(myval) > 1:
                    print('multiple values')
                    print(mykey, myval)

                # take first value
                # but also need to get 0th element
                # in case of single value.
                myval = myval[0]

                # add to dictionary
                newdb_dict[mykey].append(myval)

                
            elif mykey in ['ACASPEC_INTEGRATION_DURATION']:
                myval = np.unique(mygroup[mykey])
                
                # note how many dump times
                newdb_dict['N_TDUMP'].append(len(myval))

                # the below is a bit hacky but seems to work.
                idx = myval > 100
                myval[idx] = myval[idx]/1000.0 # convert 144 ms to s

                # take the minimum integration time.
                # might want to change to the max
                myval = np.min(myval)

                # add to dictionary
                newdb_dict[mykey].append(myval)

    
    sb_db = QTable(newdb_dict)

    sb_db.rename_column('PRJ_CODE','proposal_id')
    sb_db.rename_column('SB_NAME','schedblock_name')

    return sb_db
                


def apply_mitigations(mydb,
                      maxcubesize=40 * u.GB,
                      maxcubelimit=60 * u.GB,
                      maxproductsize = 500*u.GB):
    '''
    Purpose: apply the maximum mitigation to the data base

    Inputs: per mous data base (because mitigations are per mous)

    ** this means that I need to account for both number of spws & number of sources**

    Date        Programmer      Description of Changes
    ----------------------------------------------------------------------
    1/26/2023   A.A. Kepley     Original Code
    '''

    from large_cubes import calc_mfs_size, calc_cube_size

    mydb['imsize_mit'] = mydb['imsize']

    ## if either productsize or cubesize is greater than limit reduce to smallest possible cubesize
    mos_idx = ( (mydb['mosaic'] =='T') &
                ((mydb['wsu_cubesize_stepped2'] > maxcubesize) |
                 (mydb['wsu_productsize_early_stepped2'] > maxproductsize)))
    
    sf_idx =  ( (mydb['mosaic'] =='F') &
                ((mydb['wsu_cubesize_stepped2'] > maxcubesize) |
                 (mydb['wsu_productsize_early_stepped2'] > maxproductsize)))
    

    # For mosaics, only pixels per beam mitigation is possible
    mydb['imsize_mit'][mos_idx] = mydb['imsize'][mos_idx] * (3.0/5.0)

    # For SF, both pixels per beam and FOV mitigation is possible
    # 0.47 is from reducing FOV from 0.2 to 0.7 (linear, not area)
    mydb['imsize_mit'][sf_idx] = mydb['imsize'][sf_idx] * 0.47 * (3.0/5.0)   

    mydb['wsu_cubesize_stepped2_mit'] = calc_cube_size(mydb['imsize_mit'], mydb['wsu_nchan_spw_stepped2'])

    mydb['wsu_mfssize_mit'] = calc_mfs_size(mydb['imsize_mit'])                                                       

    for stage in ['early']:                                                       
        mydb['wsu_productsize_'+stage+'_stepped2_mit'] = 2.0 * (mydb['wsu_cubesize_stepped2_mit'] + mydb['wsu_mfssize_mit']) * mydb['wsu_nspw_'+stage] * mydb['ntarget']


def get_pipeinfo(mypkl):
    '''
    Purpose: fix up pandas generated database to play nicely with

    Inputs:
      -- location of pickle file

    Date        Programmer      Description of Changes
    ----------------------------------------------------------------------
    1/24/2023   A.A. Kepley     Original Code
    '''

    import pickle
    import pandas as pd
    from large_cubes import fix_mous_col
    
    # read in pickle
    mit = pickle.load(open(mypkl,'rb'))
    rpd = pd.DataFrame(mit).transpose()

    # set column values type
    mycols_dtype = {}

    myfloatcols = ['totaltime','imgtime','cubetime','aggtime','fctime',
              'webpredrms','webcontrms','webcontBW','webfreq',
              'webbm','webdirtyDR','webDRcorr','webcontpk','webfreqline',
              'webbmline','webpredrmsline','webdirtyDRline','webDRcorrline',
              'weblinerms','weblinepk','weblineBW','allowedcubesize',
              'allowedcubelimit','predcubesize','mitigatedcubesize','allowedprodsize',
              'initialprodsize','prodsizeaftercube','mitigatedprodsize']
    for col in myfloatcols:
        if col in rpd.columns:
            mycols_dtype[col] = 'float'

    mycols = ['nant','nEB','npt','nscan','nscience','nspw']
    for col in mycols:
        if col in rpd.columns:
            mycols_dtype[col] = 'int'        
          
    mycols = ['mitigated']
    for col in mycols:
        if col in rpd.columns:
            mycols_dtype[col] = 'bool'
    
    rpd = rpd.astype(dtype=mycols_dtype)

    # read into astropy    
    mytab = Table.from_pandas(rpd,index=True)
    mytab.rename_column('index','mous')
    mytab = Table(mytab,masked=True)

    # fix up float columns
    for col in mytab.columns:
        if col in myfloatcols:
            mytab.fill_value = np.nan

    # fix up mous names
    fix_mous_col(mytab)

    return mytab
    


def join_wsu_and_mit_dbs(mous_db,mit_db):
    '''
    Purpose: join wsu and mit data bases removing columns that aren't needed and updating units as needed

    Inputs:
        -- mous_db: assumes per mous, astropy table

        -- mit_db: assumes per mous, astropy table
    
    Date        Programmer      Description of Changes
    ----------------------------------------------------------------------
    1/11/2023   A.A. Kepley     Original Code
    '''

    # calculate calibration time
    mit_db['caltime'] = mit_db['totaltime'] - mit_db['imgtime']
    
    mous_mit_db = join(mous_db,mit_db, join_type='left')

    # convert to masked table
    mous_mit_db = Table(mous_mit_db, masked=True)
    
    # remove columns related to self-cal
    for mykey in ['webpredrms','webcontrms','webcontBW','webfreq',
                 'webbm','webdirtyDR','webDRcorr','webcontpk','webfreqline',
                 'webbmline','webpredrmsline','webdirtyDRline','webDRcorrline',
                 'weblinerms','weblinepk','weblineBW']:
        if mykey in mous_mit_db.keys():
            mous_mit_db.remove_column(mykey)

                                
    # remove redundant columns
    mous_mit_db.remove_columns(['nscience','nspw','project'])
    
    # remove column that is wrong (bug in pipeline code)
    mous_mit_db.remove_column('prodsizeaftercube')
    
    # add units to data from mitigated db
    for mykey in ['totaltime','imgtime','cubetime','aggtime','fctime','caltime']:
        mous_mit_db[mykey].unit = u.hr
        
    for mykey in ['allowedcubesize','allowedcubelimit','predcubesize','mitigatedcubesize',
                  'allowedprodsize','initialprodsize','mitigatedprodsize']:
        mous_mit_db[mykey].unit = u.GB
    
    # fix mitigated column for cases where there was a warning but no mitigation
    idx = (mous_mit_db['mitigatedprodsize'] == mous_mit_db['initialprodsize']) & (mous_mit_db['mitigated'] == True)
    mous_mit_db['mitigated'][idx] = False

    # change column names to clearly indicate that they are pl times
    for mykey in ['totaltime','imgtime','cubetime','aggtime','fctime','caltime']:
        mous_mit_db.rename_column(mykey,'pl_'+mykey)

    return mous_mit_db

def predict_pl_timings(mydb):
    '''
    Purpose: rough predictions of various pipeline timings.

    Method:
    -- wsu_caltime = pl_caltime * (nvis_cal_wsu/nvis_cal_blc)
    -- wsu_imgtime = pl_imgtime * (productsize_wsu_[mit,unmit]/productsize_blc_mit)
    -- wsu_totaltime = wsu_caltime + wsu_imgtime
    
    
    Date        Programmer      Description of Change
    --------------------------------------------------
    1/31/2023   A.A. Kepley     Original Code
    '''
    
    for stage in ['early','later_2x','later_4x']:
        mydb['wsu_pl_caltime_'+stage] =  mydb['pl_caltime'] * (mydb['wsu_nvis_'+stage+'_stepped2_typical_cal']/mydb['blc_nvis_typical_cal'])
        mydb['wsu_pl_imgtime_'+stage] = mydb['pl_imgtime'] * (mydb['wsu_productsize_'+stage+'_stepped2']/mydb['mitigatedprodsize'])
        mydb['wsu_pl_totaltime_'+stage] = mydb['wsu_pl_caltime_'+stage] + mydb['wsu_pl_imgtime_'+stage]

        # if early also calculate mitigated.
        if stage == 'early':
            mydb['wsu_pl_imgtime_'+stage+'_mit'] = mydb['pl_imgtime'] * (mydb['wsu_productsize_'+stage+'_stepped2_mit']/mydb['mitigatedprodsize'])
            mydb['wsu_pl_totaltime_'+stage+'_mit'] = mydb['wsu_pl_caltime_'+stage] + mydb['wsu_pl_imgtime_'+stage+'_mit']
            


def calc_wsu_stats(mydb,stage='early'):
    '''
    Purpose: calculate statistics for WSU Fidicual Properties

    Output: dictionary

    OUT OF DATE AS of 10/30/2023: I have a new calc stats code that does this over all the Band 1 &
    Band 2 realizations.
    
    
    Date        Programmer      Description of Changes
    ---------------------------------------------------
    1/25/2023   A.A. Kepley     Original code
    
    '''


    mystats = {}
    mystats['12m']  = {}
    mystats['7m']  = {}

    
    for array in ['12m','7m']:
        idx = mydb['array'] == array

        # select only the array I want
        mydb_arr = mydb[idx]

        # list of values to calculate
        val_list = ['datarate','nchan_agg','nchan_spw','datavol','cubesize','productsize']

        for myval in val_list:
            if myval == 'datarate':
                wsu_vals = mydb_arr['wsu_datarate_'+stage+'_stepped2_typical']
                blc_vals =  mydb_arr['blc_datarate_typical']
            elif myval == 'nchan_agg':
                wsu_vals = mydb_arr['wsu_nchan_spw_stepped2'] * mydb_arr['wsu_nspw_'+stage]
                blc_vals = mydb_arr['blc_nchan_agg'] / mydb_arr['blc_ntunings']
            elif myval == 'nchan_spw':
                wsu_vals = mydb_arr['wsu_nchan_spw_stepped2']
                blc_vals = mydb_arr['blc_nchan_max']
            elif myval == 'datavol':
                wsu_vals = mydb_arr['wsu_datavol_'+stage+'_stepped2_typical_total']
                blc_vals =  mydb_arr['blc_datavol_typical_total']
            elif myval == 'cubesize':
                wsu_vals = mydb_arr['wsu_cubesize_stepped2']
                blc_vals = mydb_arr['blc_cubesize']
            elif myval == 'productsize':
                wsu_vals = mydb_arr['wsu_productsize_'+stage+'_stepped2']
                blc_vals = mydb_arr['blc_productsize']
            else:
                print("Skipping. Value not recognized: " + myval)
                continue

            myweights = mydb_arr['time_tot'].value / np.sum(mydb_arr['time_tot'].value)            

            # standard values
            mystats[array]['wsu_'+myval+'_min'] = np.min(wsu_vals)
            mystats[array]['wsu_'+myval+'_median'] = np.median(wsu_vals)
            mystats[array]['wsu_'+myval+'_max'] = np.max(wsu_vals)
            mystats[array]['wsu_'+myval+'_75p'] = np.percentile(wsu_vals,0.75)
            mystats[array]['wsu_'+myval+'_avg']  = np.ma.average(wsu_vals)
            mystats[array]['wsu_'+myval+'_wavg']  = np.ma.average(wsu_vals,
                                                                  weights=myweights)
            
            mystats[array]['blc_'+myval+'_min'] = np.min(blc_vals)
            mystats[array]['blc_'+myval+'_median'] = np.median(blc_vals)
            mystats[array]['blc_'+myval+'_max'] = np.max(blc_vals)
            mystats[array]['blc_'+myval+'_75p'] = np.percentile(blc_vals,0.75)
            mystats[array]['blc_'+myval+'_avg']  = np.ma.average(blc_vals)
            mystats[array]['blc_'+myval+'_wavg']  = np.ma.average(blc_vals,
                                                                  weights=myweights)

            # also calculate total if appropriate
            if myval in ['datavol','productsize']:
                mystats[array]['wsu_'+myval+'_tot'] = np.ma.sum(wsu_vals)
                mystats[array]['blc_'+myval+'_tot'] = np.ma.sum(blc_vals)
                mystats[array]['time_tot'] = np.ma.sum(mydb_arr['time_tot'].to('yr'))
        
    return mystats
    

def make_wsu_stats_table(mystats, fileout='test.csv'):
    '''
    Purpose: Make CSV table of WSU stats

    Input: my stats

    Output: cvs table of WSU stats

    ***OUTDATED***


    Date        Programmer      Description of Changes
    --------------------------------------------------
    1/25/2023   A.A. Kepley     Original Code
    '''

    import csv
    
    with open (fileout,'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        writer.writerow(['', '', '', '12m','','','7m','',''])
        writer.writerow(['Value', 'Statistic', 'Unit', 'WSU','BLC', 'WSU/BLC', 'WSU', 'BLC', 'WSU/BLC']) 

        for mycat  in ['datarate','nchan_agg','nchan_spw','datavol','cubesize','productsize']:
            if mycat == 'datarate':
                mylabel = 'Data Rates'
            elif mycat == 'nchan_agg':
                mylabel = 'Aggregate Nchan (per tuning)'
            elif mycat == 'nchan_spw':
                mylabel = 'Nchan per SPW'
            elif mycat == 'datavol':
                mylabel = 'Visibility Data Volume'
            elif mycat == 'cubesize':                
                mylabel = 'Cube size'
            elif mycat == 'productsize':
                mylabel = 'Product size'
                
            if (mycat == 'datavol') | (mycat == 'productsize'):
                #val_list = ['min','median','avg','wavg','75p','max','tot']
                val_list = ['median','wavg','max','tot']
            else:
                #val_list = ['min','median','avg','wavg','75p','max']
                val_list = ['median','wavg','max']
                
            for myval in val_list:

                if myval == 'wavg':
                    outval = 'Time-Weighted Average'
                elif myval == 'median':
                    outval = 'Median'
                elif myval == 'max':
                    outval = 'Maximum'
                elif myval == 'tot':
                    outval = 'Total (1 cycle)'
                    
                    
                if mycat == 'datarate':

                    outunit = u.GB / u.s
                    
                    writer.writerow([mylabel, outval, outunit.to_string(),
                                     '{0:7.3g}'.format(mystats['12m']['wsu_'+mycat+'_'+myval].to(outunit).value),
                                     '{0:7.3g}'.format(mystats['12m']['blc_'+mycat+'_'+myval].to(outunit).value),
                                     '{0:4.1f}'.format(mystats['12m']['wsu_'+mycat+'_'+myval]/mystats['12m']['blc_'+mycat+'_'+myval]),
                                     '{0:7.3g}'.format(mystats['7m']['wsu_'+mycat+'_'+myval].to(outunit).value),
                                     '{0:7.3g}'.format(mystats['7m']['blc_'+mycat+'_'+myval].to(outunit).value),
                                     '{0:4.1f}'.format(mystats['7m']['wsu_'+mycat+'_'+myval]/mystats['7m']['blc_'+mycat+'_'+myval])])

                    mylabel = ''
                    
                elif mycat == 'datavol':

                    if myval == 'tot':
                        outunit = u.PB
                        div_factor = 2 ## divide by two to get one cycle
                    else:
                        outunit = u.TB
                        div_factor = 1
                        
                    writer.writerow([mylabel, outval, outunit.to_string(), 
                                     '{0:7.3g}'.format(mystats['12m']['wsu_'+mycat+'_'+myval].to(outunit).value/div_factor),
                                     '{0:7.3g}'.format(mystats['12m']['blc_'+mycat+'_'+myval].to(outunit).value/div_factor),
                                     '{0:4.1f}'.format(mystats['12m']['wsu_'+mycat+'_'+myval]/mystats['12m']['blc_'+mycat+'_'+myval]),
                                     '{0:7.3g}'.format(mystats['7m']['wsu_'+mycat+'_'+myval].to(outunit).value/div_factor),
                                     '{0:7.3g}'.format(mystats['7m']['blc_'+mycat+'_'+myval].to(outunit).value/div_factor),
                                     '{0:4.1f}'.format(mystats['7m']['wsu_'+mycat+'_'+myval]/mystats['7m']['blc_'+mycat+'_'+myval])])
                    mylabel = ''

                    
                elif mycat == 'cubesize':

                    outunit = u.TB
                        
                    writer.writerow([mylabel, outval, outunit.to_string(), 
                                     '{0:7.3g}'.format(mystats['12m']['wsu_'+mycat+'_'+myval].to(outunit).value),
                                     '{0:7.3g}'.format(mystats['12m']['blc_'+mycat+'_'+myval].to(outunit).value),
                                     '{0:4.1f}'.format(mystats['12m']['wsu_'+mycat+'_'+myval]/mystats['12m']['blc_'+mycat+'_'+myval]),
                                     '{0:7.3g}'.format(mystats['7m']['wsu_'+mycat+'_'+myval].to(outunit).value),
                                     '{0:7.3g}'.format(mystats['7m']['blc_'+mycat+'_'+myval].to(outunit).value),
                                     '{0:4.1f}'.format(mystats['7m']['wsu_'+mycat+'_'+myval]/mystats['7m']['blc_'+mycat+'_'+myval])])
                    mylabel = ''
                    
                elif mycat == 'productsize':
                    if myval == 'tot':
                        outunit = u.PB
                        div_factor = 2 ## divide by two to get one cycle
                    else:
                        outunit = u.TB
                        div_factor = 1
                    
                    writer.writerow([mylabel, outval, outunit.to_string(), 
                                     '{0:7.3g}'.format(mystats['12m']['wsu_'+mycat+'_'+myval].to(outunit).value/div_factor),
                                     '{0:7.3g}'.format(mystats['12m']['blc_'+mycat+'_'+myval].to(outunit).value/div_factor),
                                     '{0:4.1f}'.format(mystats['12m']['wsu_'+mycat+'_'+myval]/mystats['12m']['blc_'+mycat+'_'+myval]),
                                     '{0:7.3g}'.format(mystats['7m']['wsu_'+mycat+'_'+myval].to(outunit).value/div_factor),
                                     '{0:7.3g}'.format(mystats['7m']['blc_'+mycat+'_'+myval].to(outunit).value/div_factor),
                                     '{0:4.1f}'.format(mystats['7m']['wsu_'+mycat+'_'+myval]/mystats['7m']['blc_'+mycat+'_'+myval])])
                    mylabel = ''
                    
                elif mycat == 'nchan_spw':

                    myunit = ''
                    writer.writerow([mylabel, outval,myunit,
                                     '{0:7.0f}'.format(mystats['12m']['wsu_'+mycat+'_'+myval]),
                                     '{0:7.0f}'.format(mystats['12m']['blc_'+mycat+'_'+myval]),
                                     '{0:4.1f}'.format(mystats['12m']['wsu_'+mycat+'_'+myval]/mystats['12m']['blc_'+mycat+'_'+myval]),
                                     '{0:7.0f}'.format(mystats['7m']['wsu_'+mycat+'_'+myval]),
                                     '{0:7.0f}'.format(mystats['7m']['blc_'+mycat+'_'+myval]),
                                     '{0:4.1f}'.format(mystats['7m']['wsu_'+mycat+'_'+myval]/mystats['7m']['blc_'+mycat+'_'+myval])])
                    mylabel = ''
                    
                elif mycat == 'nchan_agg':
                    myunit = ''
                    writer.writerow([mylabel, outval, myunit,                                     
                                     '{0:7.0f}'.format(mystats['12m']['wsu_'+mycat+'_'+myval]),
                                     '{0:7.0f}'.format(mystats['12m']['blc_'+mycat+'_'+myval]),
                                     '{0:4.1f}'.format(mystats['12m']['wsu_'+mycat+'_'+myval]/mystats['12m']['blc_'+mycat+'_'+myval]),
                                     '{0:7.0f}'.format(mystats['7m']['wsu_'+mycat+'_'+myval]),
                                     '{0:7.0f}'.format(mystats['7m']['blc_'+mycat+'_'+myval]),
                                     '{0:4.1f}'.format(mystats['7m']['wsu_'+mycat+'_'+myval]/mystats['7m']['blc_'+mycat+'_'+myval])])
                    

                    mylabel = ''
                    
                else:
                    myunit = ''
                    writer.writerow([mylabel, outval, myunit,
                                     mystats['12m']['wsu_'+mycat+'_'+myval],
                                     mystats['12m']['blc_'+mycat+'_'+myval],
                                     mystats['12m']['wsu_'+mycat+'_'+myval]/mystats['12m']['blc_'+mycat+'_'+myval],
                                     mystats['7m']['wsu_'+mycat+'_'+myval],
                                     mystats['7m']['blc_'+mycat+'_'+myval],
                                     mystats['7m']['wsu_'+mycat+'_'+myval]/mystats['7m']['blc_'+mycat+'_'+myval]])
                


def make_wsu_stats_table_newstats_datarate(mystats,fileout='test.tex'):
    '''
    Purpose: create giant csv table with stats properties

    Input: dictionary with summary statistics.
        stats[quantity][array][value]

    Output: latex table

    Date        Programmer      Description of Changes
    ----------------------------------------------------
    10/31/2023  A.A. Kepley     Original Code. Also Happy Halloween!
    '''

    fout = open(fileout,'w')

    tablehead = '''\definecolor{myband}{RGB}{255,235,205}
\definecolor{my2x}{RGB}{182, 208, 226}
\definecolor{my4x}{RGB}{251, 206, 177}

\\begin{table}
\centering
\caption{Overview of Data Rate Properties for  WSU \label{tab:overview_datarates}}
\\begin{tblr}{colspec={|[2pt]Q[l]|Q[l]|[1pt]c|c|c|c|c|c|[2pt]},
width=\\textwidth,
cells = {font=\scriptsize}}
\\hline[2pt]
\SetCell[c=2,r=2]{c} & & \SetCell[c=3]{c,bg=my2x} Early WSU  & & & \SetCell[c=3]{c,bg=my4x} Later WSU & &  \\\\
& & 12m & 7m & both & 12m & 7m & both \\\\ \hline[1pt]
'''

    fout.write(tablehead)
    
    for myquant in ['datarate','nchan_agg']:
        if myquant == 'datarate':
            myquant_label = 'Data Rate'
            myquant_unit = '(GB/s)'
        elif myquant == 'nchan_agg':
            myquant_label = '{Number of \\\\ of Channels}'
            myquant_unit = ''
            #elif myquant == 'sysperf':
        #    myquant_label = '{System \\\\ Performance}'
        #else:
        #    print("Quantity unknown: " +myquant)


        for myval in ['median','wavg','max']:
            if myval == 'median':
                myval_label = '{{Median {:s}}}'.format(myquant_unit)
                outstr = "{:s} & {:s}".format(myquant_label,myval_label) 
            elif myval == 'wavg':
                myval_label = '{{Time Weighted \\\\ Average {:s}}}'.format(myquant_unit)
                outstr = " & {:s}".format(myval_label) 
            elif myval == 'max':
                myval_label = 'Maximum {:s}'.format(myquant_unit)
                outstr = " & {:s}".format(myval_label) 
            else:
                print("Value unknown: "+myval)



            for mystage in ['early','later_4x']:
                
                for myarray in ['12m','7m','both']:
                    if myquant == 'datarate':
                        myfullquant = 'wsu_datarate_'+mystage+'_stepped2_typical'
                        myquantunit = mystats[myfullquant]['unit']
                        outstr = outstr + "& ${:5.3f}$".format(mystats[myfullquant][myarray][myval+"_mean"])
                        
                    elif myquant == 'nchan_agg':
                        myfullquant = 'wsu_nchan_agg_stepped2_'+mystage
                        myquantunit = ''
                        outstr = outstr + "& {:6,d}".format(round(mystats[myfullquant][myarray][myval+"_mean"]))
                        #outstr = outstr + "& $ {{ {:7.3e} \\\\ \\pm {:7.3e} }} $".format(round(mystats[myfullquant][myarray][myval+"_mean"]),mystats[myfullquant][myarray][myval+"_std"])

                    #elif myquant == 'sysperf':
                    #    myfullquant = 'wsu_sysperf_'+mystage+'_stepped2_typical_aprojonly'
                    #    myquantunit = 'PFLOP/s'
                    #    outstr = outstr + "& $  {:5.3f}\\pm {:5.3f} $".format(mystats[myfullquant][myarray][myval+"_mean"],mystats[myfullquant][myarray][myval+"_std"])
                    else:
                        print('Quantity unknown: '+myquant)

            outstr = outstr + "\\\\ \n"
            fout.write(outstr)
        if myquant != 'nchan_agg': 
            outstr = "\hline \n"
            fout.write(outstr)

    tablefoot = '''
\hline[2pt]
\end{tblr}
\end{table}   
    '''

    fout.write(tablefoot)
    fout.close()

    
def make_wsu_stats_table_newstats_datavol(mystats,fileout='test2.tex'):
    '''
    Purpose: create giant csv table with stats properties

    Input: dictionary with summary statistics.
        stats[quantity][array][value]

    Output: latex table

    Date        Programmer      Description of Changes
    ----------------------------------------------------
    10/31/2023  A.A. Kepley     Original Code. Also Happy Halloween!
    '''
    fout = open(fileout,'w')

    tablehead = '''\definecolor{myband}{RGB}{255,235,205}
\definecolor{my2x}{RGB}{182, 208, 226}
\definecolor{my4x}{RGB}{251, 206, 177}

\\begin{table}
\centering
\caption{Overview of Data Volume Properties for WSU \label{tab:overview_datavol}}
\\begin{tblr}{colspec={|[2pt]Q[l]|Q[l]|[1pt]c|c|c|c|c|c|[2pt]},
width=\\textwidth,
cells = {font=\scriptsize}}
\\hline[2pt]
\SetCell[c=2,r=2]{c} & & \SetCell[c=3]{c,bg=my2x} Early WSU  & & & \SetCell[c=3]{c,bg=my4x} Later WSU & &  \\\\
& & 12m & 7m & both & 12m & 7m & both \\\\ \hline[1pt]
'''

    fout.write(tablehead)
    
    for myquant in ['datavol_total','datavol_target_tot','productsize']:
        if myquant == 'datavol_total':
            myquant_label = '{Visibility Data \\\\ Volume (Total) }'
        elif myquant == 'datavol_target_tot':
            myquant_label = '{Visibility Data \\\\ Volume (Science)}'
        elif myquant == 'productsize':
            myquant_label = '{Product Size \\\\ (Total)}'
        else:
            print("Quantity unknown: " +myquant)


        for myval in ['median','wavg','max','total']:
            if myval == 'median':
                myval_label = '{Median (TB)}'
                outstr = "{:s} & {:s}".format(myquant_label,myval_label) 
                scale = 1.0
            elif myval == 'wavg':
                myval_label = '{Time Weighted \\\\ Average (TB)} '
                outstr = " & {:s}".format(myval_label) 
                scale = 1.0
            elif myval == 'max':
                myval_label = 'Maximum (TB)'
                outstr = " & {:s}".format(myval_label) 
                scale = 1.0
            elif myval == 'total':
                myval_label = '{{ {\\bf Total per cycle (PB)}}}'
                outstr = "  & {:s}".format(myval_label)
                scale = 2.0 * 1000.0 # divide by 2 cycles and 1000 to get 1 cycle in PB
            else:
                print("Value unknown: "+myval)

            for mystage in ['early','later_4x']:
                
                for myarray in ['12m','7m','both']:
                    if myquant == 'datavol_total':
                        myfullquant = 'wsu_datavol_'+mystage+'_stepped2_typical_total'
                        #myquantunit = mystats[myfullquant]['unit']
                        outstr = outstr + "& ${:7.3f}$".format(mystats[myfullquant][myarray][myval+"_mean"]/scale)                
                    elif myquant == 'datavol_target_tot':
                        myfullquant = 'wsu_datavol_'+mystage+'_stepped2_typical_target_tot'
                        #myquantunit = mystats[myfullquant]['unit']
                        outstr = outstr + "& ${:7.3f}$".format(mystats[myfullquant][myarray][myval+"_mean"]/scale)
                    elif myquant == 'productsize':
                        myfullquant = 'wsu_productsize_'+mystage+'_stepped2'
                        #myquantunit = mystats[myfullquant]['unit']
                        outstr = outstr + "& ${:7.3f}$".falcormat(mystats[myfullquant][myarray][myval+"_mean"]/scale)                
                    else:
                        print('Quantity unknown: '+myquant)

            outstr = outstr + "\\\\ \n"
            fout.write(outstr)

            if myval == 'max':
                fout.write("\\cline{2-8} \n")
            
        if myquant != 'productsize': 
            outstr = "\hline \n"
            fout.write(outstr)

    tablefoot = '''
\hline[2pt]
\end{tblr}
\end{table}   
    '''

    fout.write(tablefoot)
    fout.close()


def make_wsu_stats_table_newstats_sysperf(mystats, fileout='test.tex'):
    '''
    Purpose: create a giant csv table with sysperf related stats properties

    Input: dictionary with summary statistics
         stats[quantity][array][value]

    Output: latex table

    Date        Programmer      Description of Changes
    ----------------------------------------------------
    12/10/2023  A.A. Kepley     original code based on similar functions
    '''
    
    fout = open(fileout,'w')

    tablehead = '''\definecolor{myband}{RGB}{255,235,205}
\definecolor{my2x}{RGB}{182, 208, 226}
\definecolor{my4x}{RGB}{251, 206, 177}

\\begin{table}
\centering
\caption{Overview of System Performance Related Quantities for  WSU \label{tab:overview_sysperf}}
\\begin{tblr}{colspec={|[2pt]Q[l]|Q[l]|[1pt]c|c|c|c|c|c|[2pt]},
width=\\textwidth,
cells = {font=\scriptsize}}
\\hline[2pt]
\SetCell[c=2,r=2]{c} & & \SetCell[c=3]{c,bg=my2x} Early WSU  & & & \SetCell[c=3]{c,bg=my4x} Later WSU & &  \\\\
& & 12m & 7m & both & 12m & 7m & both \\\\ \hline[1pt]
'''
    fout.write(tablehead)
    
    for myquant in ['datarate','visrate','sysperf']:
        if myquant == 'datarate':
            myquant_label = 'Data Rate'
            myquant_unit = '(GB/s)'
        elif myquant == 'visrate':
            myquant_label = 'Visibility Rate'
            myquant_unit = '(Gvis/hr)'
        elif myquant == 'sysperf':
            myquant_label = '{System \\\\ Performance}'
            myquant_unit = '(PFLOP/s)'
        else:
            print("Quantity unknown: " +myquant)
            
        for myval in ['median','wavg','max']:
            if myval == 'median':
                myval_label = '{{Median {:s}}}'.format(myquant_unit)
                outstr = "{:s} & {:s}".format(myquant_label,myval_label) 
            elif myval == 'wavg':
                myval_label = '{{Time Weighted \\\\ Average {:s}}}'.format(myquant_unit)
                outstr = " & {:s}".format(myval_label) 
            elif myval == 'max':
                myval_label = 'Maximum {:s}'.format(myquant_unit)
                outstr = " & {:s}".format(myval_label) 
            else:
                print("Value unknown: "+myval)

            
            for mystage in ['early','later_4x']:            
                for myarray in ['12m','7m','both']:
                    if myquant == 'datarate':
                        myfullquant = 'wsu_datarate_'+mystage+'_stepped2_typical'
                        outstr = outstr + "& ${:5.3f}$".format(mystats[myfullquant][myarray][myval+"_mean"])
                    elif myquant == 'visrate':
                        myfullquant = 'wsu_visrate_'+mystage+'_stepped2_typical'
                        outstr = outstr + "& ${:5.1f}$".format(mystats[myfullquant][myarray][myval+"_mean"])
                    elif myquant == 'sysperf':
                        myfullquant = 'wsu_sysperf_'+mystage+'_stepped2_typical_aprojonly'
                        outstr = outstr + "& $  {:5.3f} $".format(mystats[myfullquant][myarray][myval+"_mean"])
                    else:
                        print('Quantity unknown: '+myquant)

            outstr = outstr + "\\\\ \n"
            fout.write(outstr)

        if myquant != 'sysperf': 
            outstr = "\hline \n"
            fout.write(outstr)

    tablefoot = '''
\hline[2pt]
\end{tblr}
\end{table}   
    '''

    fout.write(tablefoot)
    fout.close()

    
    
    
def make_blc_stats_table_newstats_datarate(mystats,fileout='test.tex'):
    '''
    Purpose: create giant csv table with stats properties

    Input: dictionary with summary statistics.
        stats[quantity][array][value]

    Output: latex table

    Date        Programmer      Description of Changes
    ----------------------------------------------------
    10/31/2023  A.A. Kepley     Original Code. Also Happy Halloween!
    11/3/2023   A.A. Kepley     Updated for BLC
    '''

    fout = open(fileout,'w')

    tablehead = '''\definecolor{myband}{RGB}{255,235,205}
\definecolor{my2x}{RGB}{182, 208, 226}
\definecolor{my4x}{RGB}{251, 206, 177}

\\begin{table}
\centering
\caption{Overview of Data Rate Properties for  BLC/ACA \label{tab:overview_datarates_blc}}
\\begin{tblr}{colspec={|[2pt]Q[l]|Q[l]|[1pt]c|c|c|[2pt]},
width=\\textwidth,
cells = {font=\scriptsize}}
\\hline[2pt]
\SetCell[c=2,r=2]{c} & & \SetCell[c=3]{c,bg=myband} BLC/ACA  & &    \\\\
& & 12m & 7m & both \\\\ \hline[1pt]
'''

    fout.write(tablehead)
    
    for myquant in ['datarate','nchan_agg']:
        if myquant == 'datarate':
            myquant_label = 'Data Rate'
            myquant_unit = '(MB/s)'
            scale= 0.001
        elif myquant == 'nchan_agg':
            myquant_label = '{Number of \\\\ of Channels}'
            myquant_unit = ''
            scale=1.0
        #elif myquant == 'sysperf':
        #    myquant_label = '{System \\\\ Performance}'
        #    myquantunit = 'TFLOP/s'
        #    scale=0.001
        else:
            print("Quantity unknown: " +myquant)


        for myval in ['median','wavg','max']:
            if myval == 'median':
                myval_label = '{{Median {:s}}}'.format(myquant_unit)
                outstr = "{:s} & {:s}".format(myquant_label,myval_label) 
            elif myval == 'wavg':
                myval_label = '{{Time Weighted \\\\ Average {:s}}}'.format(myquant_unit)
                outstr = " & {:s}".format(myval_label) 
            elif myval == 'max':
                myval_label = 'Maximum {:s}'.format(myquant_unit)
                outstr = " & {:s}".format(myval_label)                 
            else:
                print("Value unknown: "+myval)

                
            for myarray in ['12m','7m','both']:
                if myquant == 'datarate':
                    myfullquant = 'blc_datarate_typical'
                    outstr = outstr + "& ${:5.2f} $".format(mystats[myfullquant][myarray][myval+"_mean"]/scale)
                elif myquant == 'nchan_agg':
                    myfullquant = 'blc_nchan_agg'
                    outstr = outstr + "& {:6,d}".format(round(mystats[myfullquant][myarray][myval+"_mean"]/scale))
                    #outstr = outstr + "& ${{ {:7.3e} \pm {:7.3e} }}$".format(round(mystats[myfullquant][myarray][myval+"_mean"]),mystats[myfullquant][myarray][myval+"_std"])

                #elif myquant == 'sysperf':
                #    myfullquant = 'blc_sysperf_typical_aprojonly'
                #    outstr = outstr + "& ${:5.2f}$".format(mystats[myfullquant][myarray][myval+"_mean"]/scale)
                else:
                    print('Quantity unknown: '+myquant)

            outstr = outstr + "\\\\ \n"
            fout.write(outstr)

        if myquant != 'nchan_agg': 
            outstr = "\hline \n"
            fout.write(outstr)

    tablefoot = '''
\hline[2pt]
\end{tblr}
\end{table}   
    '''

    fout.write(tablefoot)
    fout.close()

     
def make_blc_stats_table_newstats_datavol(mystats,fileout='test2.tex'):
    '''
    Purpose: create giant csv table with stats properties

    Input: dictionary with summary statistics.
        stats[quantity][array][value]

    Output: latex table

    Date        Programmer      Description of Changes
    ----------------------------------------------------
    10/31/2023  A.A. Kepley     Original Code. Also Happy Halloween!
    '''
    
    fout = open(fileout,'w')
    
    tablehead = '''\definecolor{myband}{RGB}{255,235,205}
\definecolor{my2x}{RGB}{182, 208, 226}
 \definecolor{my4x}{RGB}{251, 206, 177}

\\begin{table}
\centering
\caption{Overview of Data Volume Properties for BLC/ACA \label{tab:overview_datavol_blc}}
\\begin{tblr}{colspec={|[2pt]Q[l]|Q[l]|[1pt]c|c|c|[2pt]},
width=\\textwidth,
cells = {font=\scriptsize}}
\\hline[2pt]
\SetCell[c=2,r=2]{c} & & \SetCell[c=3]{c,bg=myband} BLC/ACA & &    \\\\
& & 12m & 7m & both \\\\ \hline[1pt]
'''

    fout.write(tablehead)
    
    for myquant in ['datavol_total','datavol_target_tot','productsize']:
        if myquant == 'datavol_total':
            myquant_label = '{Visibility Data \\\\ Volume (Total) }'
        elif myquant == 'datavol_target_tot':
            myquant_label = '{Visibility Data \\\\ Volume (Science)}'
        elif myquant == 'productsize':
            myquant_label = '{Product Size \\\\ (Total)}'
        else:
            print("Quantity unknown: " +myquant)


        for myval in ['median','wavg','max','total']:
            if myval == 'median':
                myval_label = '{Median (GB)}'
                outstr = "{:s} & {:s}".format(myquant_label,myval_label) 
                scale = 0.001
            elif myval == 'wavg':
                myval_label = '{Time Weighted \\\\ Average (GB)} '
                outstr = " & {:s}".format(myval_label) 
                scale = 0.001
            elif myval == 'max':
                myval_label = '{Maximum  (GB)}'
                outstr = " & {:s}".format(myval_label) 
                scale=0.001
                #scale=1
            elif myval == 'total':
                myval_label = '{{\\bf Total per cycle (TB)}}'
                outstr = "  & {:s}".format(myval_label)            
                #scale=0.002 # divide by 2 cycles
                scale=2 # divide by 2 cycles
            else:
                print("Value unknown: "+myval)
                
            for myarray in ['12m','7m','both']:
                if myquant == 'datavol_total':
                    myfullquant = 'blc_datavol_typical_total'
                    #myquantunit = mystats[myfullquant]['unit']
                    outstr = outstr + "& ${:5.2f} $".format(mystats[myfullquant][myarray][myval+"_mean"]/scale)
                elif myquant == 'datavol_target_tot':
                    myfullquant = 'blc_datavol_typical_target_tot'
                    #myquantunit = mystats[myfullquant]['unit']
                    outstr = outstr + "& ${:5.2f} $".format(mystats[myfullquant][myarray][myval+"_mean"]/scale)
                elif myquant == 'productsize':
                    myfullquant = 'blc_productsize'
                    #myquantunit = mystats[myfullquant]['unit']
                    outstr = outstr + "& ${:5.2f} $".format(mystats[myfullquant][myarray][myval+"_mean"]/scale)      
                else:
                    print('Quantity unknown: '+myquant)

            outstr = outstr + "\\\\ \n"
            fout.write(outstr)

            if myval == 'max':
                fout.write("\\cline{2-8} \n")
            
        if myquant != 'productsize': 
            outstr = "\hline \n"
            fout.write(outstr)

    tablefoot = '''
\hline[2pt]
\end{tblr}
\end{table}   
    '''

    fout.write(tablefoot)
    fout.close()



def make_blc_stats_table_newstats_sysperf(mystats,fileout='test2.tex'):
    '''
    Purpose: create giant csv table with stats properties

    Input: dictionary with summary statistics.
        stats[quantity][array][value]

    Output: latex table

    Date        Programmer      Description of Changes
    ----------------------------------------------------
    10/31/2023  A.A. Kepley     Original Code. Also Happy Halloween!
    11/3/2023   A.A. Kepley     Updated for BLC
    12/10/2023  A.A. Kepley     Modified for system performance calcs
    '''
    
    fout = open(fileout,'w')

    tablehead = '''\definecolor{myband}{RGB}{255,235,205}
\definecolor{my2x}{RGB}{182, 208, 226}
\definecolor{my4x}{RGB}{251, 206, 177}

\\begin{table}
\centering
\caption{Overview of System Performance Related Quantities for  BLC/ACA \label{tab:overview_sysperf_blc}}
\\begin{tblr}{colspec={|[2pt]Q[l]|Q[l]|[1pt]c|c|c|[2pt]},
width=\\textwidth,
cells = {font=\scriptsize}}
\\hline[2pt]
\SetCell[c=2,r=2]{c} & & \SetCell[c=3]{c,bg=myband} BLC/ACA  & &    \\\\
& & 12m & 7m & both \\\\ \hline[1pt]
'''

    fout.write(tablehead)
    
    for myquant in ['datarate','visrate','sysperf']:
        if myquant == 'datarate':
            myquant_label = 'Data Rate'
            myquant_unit = '(MB/s)'
            scale= 0.001
        elif myquant == 'visrate':
            myquant_label = 'Visibility Rate'
            myquant_unit = '(Tvis/hr)'
            scale=0.001
            #scale=1
        elif myquant == 'sysperf':            
            myquant_label = '{System \\\\ Performance}'
            myquant_unit = '(TFLOP/s)'
            scale=0.001
        else:
            print("Quantity unknown: " +myquant)


        for myval in ['median','wavg','max']:
            if myval == 'median':
                myval_label = '{{Median {:s}}}'.format(myquant_unit)
                outstr = "{:s} & {:s}".format(myquant_label,myval_label) 
            elif myval == 'wavg':
                myval_label = '{{Time Weighted \\\\ Average {:s}}}'.format(myquant_unit)
                outstr = " & {:s}".format(myval_label) 
            elif myval == 'max':
                myval_label = 'Maximum {:s}'.format(myquant_unit)
                outstr = " & {:s}".format(myval_label)                 
            else:
                print("Value unknown: "+myval)

                
            for myarray in ['12m','7m','both']:
                if myquant == 'datarate':
                    myfullquant = 'blc_datarate_typical'
                    outstr = outstr + "& ${:5.2f} $".format(mystats[myfullquant][myarray][myval+"_mean"]/scale)
                elif myquant == 'visrate':
                    myfullquant = 'blc_visrate_typical'
                    outstr = outstr + "& ${:5.1f}$".format(round(mystats[myfullquant][myarray][myval+"_mean"]/scale))
                elif myquant == 'sysperf':
                    myfullquant = 'blc_sysperf_typical_aprojonly'
                    outstr = outstr + "& ${:5.2f}$".format(mystats[myfullquant][myarray][myval+"_mean"]/scale)
                else:
                    print('Quantity unknown: '+myquant)

            outstr = outstr + "\\\\ \n"
            fout.write(outstr)

        if myquant != 'sysperf': 
            outstr = "\hline \n"
            fout.write(outstr)

    tablefoot = '''
\hline[2pt]
\end{tblr}
\end{table}   
    '''

    fout.write(tablefoot)
    fout.close()

    
    
                    
def make_mitigation_stats_table(mydb, maxcubesize=40*u.GB,
                                maxcubelimit=60*u.GB,
                                maxproductsize=500*u.GB):
    '''
    Purpose: calculate mitigation statistics for WSU

    Date        Programmer      Description of Changes
    -----------------------------------------------
    1/26/2023   A.A. Kepley     Original Code
    '''

    print("maxcubesize: " + maxcubesize.to_string())
    print("maxcubelimit: " + maxcubelimit.to_string())
    print("maxproductsize: " + maxproductsize.to_string())

    print("\n")

    nmous = float(len(mydb))

    print("For all stages:")

    myval = np.sum(mydb['wsu_cubesize_stepped2_mit'] > maxcubelimit)
    print("Percent of MOUSes that will fail on cubesize, assuming mitigation: {:5.2f}".format(100*myval/nmous))
    
    myval = np.sum(mydb['wsu_cubesize_stepped2'] > maxcubelimit)
    print("Percent of MOUSes that will fail on cubesize, assuming no mitigation: {:5.2f}".format(100*myval/nmous))

    myval = np.sum(mydb['wsu_cubesize_stepped2_mit'] > maxproductsize)
    print("Percent of MOUSes with single cube size greater than productsize, assuming mitigation: {:5.2f}".format(100*myval/nmous))
    
    myval = np.sum(mydb['wsu_cubesize_stepped2'] > maxproductsize)
    print("Percent of MOUSes with single cube size greater than productsize, assuming no mitigation: {:5.2f}".format(100*myval/nmous))

    
    
    print("\n")
    print("early + mitigation:")
    
    myval1 = np.sum((mydb['wsu_productsize_early_stepped2_mit'] > maxproductsize) & 
                   (mydb['wsu_cubesize_stepped2_mit'] < maxcubelimit))
    print("Percent of MOUSes that will fail productsize (only) assuming mitigation: {:5.2f}".format(100*myval1 / nmous))

    myval2 = np.sum((mydb['wsu_productsize_early_stepped2_mit'] < maxproductsize) & 
                                    (mydb['wsu_cubesize_stepped2_mit'] > maxcubelimit))
    print("Percent of MOUSes that will fail cubesize (only) assuming mitigation: {:5.2f}".format(100*myval2/nmous))

    myval3 = np.sum((mydb['wsu_productsize_early_stepped2_mit'] > maxproductsize ) & 
                   (mydb['wsu_cubesize_stepped2_mit'] > maxcubelimit))
    print("Percent of MOUSes that will fail on cube and productsize assuming mitigation: {:5.2f}".format(100*myval3/nmous))

    myval4 = myval1 + myval2+ myval3
    print("Total Percentage of MOUSes failing mitigation: {:5.2f}".format(100*myval4/nmous))
    
    for stage in ['early','later_2x', 'later_4x']:
        print("\n")
        print(stage+':')
        myval1 = np.sum((mydb['wsu_productsize_'+stage+'_stepped2'] > maxproductsize) & 
                       (mydb['wsu_cubesize_stepped2'] < maxcubelimit))
        print("Percent of MOUSes that will fail productsize (only) assuming NO mitigation: {:5.2f}".format(100*myval1 / nmous))
        
        myval2 = np.sum((mydb['wsu_productsize_'+stage+'_stepped2'] < maxproductsize) & 
                       (mydb['wsu_cubesize_stepped2'] > maxcubesize))
        print("Percent of MOUSes that will fail cubesize (only) assuming NO mitigation: {:5.2f}".format(100*myval2/nmous))
        
        myval3 = np.sum((mydb['wsu_productsize_'+stage+'_stepped2'] > maxproductsize) & 
                       (mydb['wsu_cubesize_stepped2'] > maxcubesize))
        print("Percent of MOUSes that will fail on cube and productsize assuming NO mitigation: {:5.2f}".format(100*myval3/nmous))
          
        myval4 = myval1 + myval2+ myval3
        print("Total Percentage of MOUSes failing mitigation: {:5.2f}".format(100*myval4/nmous))
    

                


def calc_wsu_sysperf_stats(mydb, label='allgrid', stages = ['early','later_2x','later_4x']):
    '''
    Purpose: calculate statistics for WSU Fidicual Properties

    Output: dictionary
    
    
    Date        Programmer      Description of Changes
    ---------------------------------------------------
    1/25/2023   A.A. Kepley     Original code
    
    '''


    mystats = {}
    mystats['12m']  = {}
    mystats['7m']  = {}

    allresults = []

    results = {}
    results['stage'] = 'blc'
    
    results['median'] =  np.median(mydb['blc_sysperf_typical'+'_'+label])
    results['wavg']  = np.average(mydb['blc_sysperf_typical'+'_'+label], weights=mydb['weights_all'])
    results['max']= np.max(mydb['blc_sysperf_typical'+'_'+label])

    allresults.append(results)
    
    for mystage in stages:
        results = {}
        results['stage'] = mystage
        results['median'] = np.median(mydb['wsu_sysperf_'+mystage+'_stepped2_typical'+'_'+label])
        results['wavg'] = np.average(mydb['wsu_sysperf_'+mystage+'_stepped2_typical'+'_'+label],weights=mydb['weights_all'])
        results['max'] = np.max(mydb['wsu_sysperf_'+mystage+'_stepped2_typical'+'_'+label])

        allresults.append(results)

    return allresults

   
    
def create_db_for_sanjay(mydb,filename='data/test.csv'):
    '''
    Purpose:

    Inputs: mydb

    Output: csv file with values that Sanjay has requested.
        * fraction X
        * array X 
        * dump time X 
        * channel size X
        * number of channels X
        * number of spws X
        * image linear size X
        * number of baselines X
        * visibility rate (vis/hr) X
        * fractional bandwidth X 
        * multi-term ??
        * data rate X
        * required system performance X ???

    Method: Take data base and output a sub-set of relevant columns to csv.
   
    Date        Programmer      Description of Changes
    ----------------------------------------------------------------------
    2/27/2023   A.A. Kepley     Original Code
    '''

    mycols = ['mous','proposal_id','schedblock_name', ## proposal info
              'weights_all', # fraction of total cycle 7 and 8 time spend observing
              'array','nant_typical','nbase_typical', 'L80',## array information
              'imsize','cell','mosaic', ## image information
              'wsu_freq','wsu_npol','wsu_tint', ## basic correlator info
              'wsu_bandwidth_spw', 'wsu_specwidth_stepped2','wsu_nchan_spw_stepped2',  'wsu_frac_bw_spw', # SPW properties
              'wsu_nspw_early','wsu_nspw_later_2x','wsu_nspw_later_4x', # nspws
              'wsu_bandwidth_early','wsu_bandwidth_later_2x','wsu_bandwidth_later_4x', #bandwidth
              'wsu_frac_bw_early','wsu_frac_bw_later_2x','wsu_frac_bw_later_4x', #fractional BW
              'wsu_datarate_early_stepped2_typical','wsu_datarate_later_2x_stepped2_typical','wsu_datarate_later_4x_stepped2_typical',
              'wsu_visrate_early_stepped2_typical','wsu_visrate_later_2x_stepped2_typical','wsu_visrate_later_4x_stepped2_typical',
              'wsu_sysperf_early','wsu_sysperf_later_2x','wsu_sysperf_later_4x']

    
    mydb[mycols].write(filename)
         
    
    pass


def write_out_db_info(mydb,filename='data/col_info.tex'):
    '''
    Purpose: write out data base column information so I can put into latex easily.

    Inputs:  mydb

    Outputs: file containing list of all columns and units in db 

    Date        Programmer      Description of Changes
    -----------------------------------------------------------------
    8/21/2023   A.A. Kepley     Original Code
    '''

    f = open(filename,'w+')
    
    col_list = mydb.columns

    for col in col_list:
        if bool(mydb[col].unit):
            f.write(" {0} & {1} & \\\\ \n".format(col, mydb[col].unit.to_string('latex')))
        else:
            f.write(" {0} & \\nodata & \\\\ \n".format(col))

    f.close()
   



def fix_scientific_categories(mydb):
    '''
    Purpose: fix scientific categories given by archive to match the proposal scientific categories

    Inputs: mydb

    Outputs: mydb plus column with new categories

    Date        Programmer      Description of Changes
    ------------------------------------------------------------
    8/28/2023   A.A. Kepley     Original Code
    '''

    category_dict =  {'Cosmology and the high redshift Universe': ["Lyman Alpha Emitters/Blobs (LAE/LAB)",
                                                                   "Lyman Break Galaxies (LBG)","Starburst galaxies",
                                                                   "Sub-mm Galaxies (SMG)","High-z Active Galactic Nuclei (AGN)",
                                                                   "Gravitational lenses",
                                                                   "Damped Lyman Alpha (DLA) systems",
                                                                   "Cosmic Microwave Background (CMB)/Sunyaev-Zel'dovich Effect (SZE)",
                                                                   "Galaxy structure & evolution","Gamma Ray Bursts (GRB)",
                                                                   "Galaxy Clusters"],
                      'Galaxies and galactic nuclei': ["Starbursts", "star formation","Active Galactic Nuclei (AGN)/Quasars (QSO)",

                                                       "Spiral galaxies","Merging and interacting galaxies","Surveys of galaxies",
                                                       "Outflows","jets", "feedback","Early-type galaxies",
                                                       "Galaxy groups and clusters","Galaxy chemistry","Galactic centres/nuclei",
                                                       "Dwarf/metal-poor galaxies",
                                                       "Luminous and Ultra-Luminous Infra-Red Galaxies (LIRG & ULIRG)",
                                                       "Giant Molecular Clouds (GMC) properties"],
                      'ISM, star formation and astrochemistry': ['Outflows', 'jets and ionized winds','High-mass star formation',
                                                                 'Intermediate-mass star formation','Low-mass star formation',
                                                                 'Pre-stellar cores', 'Infra-Red Dark Clouds (IRDC)',
                                                                 'Astrochemistry','Inter-Stellar Medium (ISM)/Molecular clouds',
                                                                 'Photon-Dominated Regions (PDR)/X-Ray Dominated Regions (XDR)',
                                                                 'HII regions',
                                                                 'Magellanic Clouds'],
                      'Circumstellar disks, exoplanets and the solar system': ['Debris disks','Disks around low-mass stars',
                                                                               'Disks around high-mass stars','Exo-planets',
                                                                               'Solar system: Comets',
                                                                               'Solar system: Planetary atmospheres',
                                                                               'Solar system: Planetary surfaces',
                                                                               'Solar system: Trans-Neptunian Objects (TNOs)',
                                                                               'Solar system: Asteroids'],
                      'Stellar evolution and the Sun': ['The Sun','Main sequence stars','Asymptotic Giant Branch (AGB) stars',
                                                        'Post-AGB stars','Hypergiants','Evolved stars - Shaping/physical structure',
                                                        'Evolved stars - Chemistry','Cataclysmic stars',
                                                        'Luminous Blue Variables (LBV)','White dwarfs',
                                                        'Brown dwarfs','Supernovae (SN) ejecta',
                                                        'Pulsars and neutron stars','Black holes','Transients']}


    newcat_array = []
    
    for row in mydb:
        newcat = []
        keylist = row['science_keyword'].split(', ')
        
        for mykey in keylist:
            for mycat in category_dict.keys():
                if mykey in category_dict[mycat]:
                    newcat.append(mycat)
                    
        if len(np.unique(newcat)) > 1:

            if row['scientific_category'] == 'Active galaxies':
                newcat = ['Galaxies and galactic nuclei']
            elif row['scientific_category'] == 'ISM and star formation':
                newcat  = ['ISM, star formation and astrochemistry']
            else:
                print('Multiple categories found: '+ row['mous'])
                print(row['scientific_category'])
                print(np.unique(newcat))
                print('--')
                

                    
        if len(newcat) == 0:
            print('No new category found: '+row['mous'])
            print(keylist)
            print(row['scientific_category'])
            print(np.unique(newcat))
            print('--')

            

        newcat_array.append(np.unique(newcat)[0])

    if 'scientific_category_proposal' in mydb.columns:
        mydb.replace_column('scientific_category_proposal',newcat_array)
    else:
        mydb.add_column(newcat_array,6,'scientific_category_proposal')
        
                           

def remove_projects(mydb, array='12m', time_frac=0.05):
    '''
    Purpose: remove some projects from the total and return database with projects removed

    Input: mydb

    Output: new data base with projects removed.

    Method: Following method outlined by IST/John Carpenter.
    I did parameterize so can make it easy.

    TODOS:
    -- How to implement varying band probability?
    
    Date        Programmer      Description of Changes
    -----------------------------------------------------
    9/19/2023   A.A. Kepley     Original Code
    
    '''

    # copy old data base to new data base
    newdb = mydb.copy()

    # calculate number of rows in original database
    orig_nrows = len(mydb)
    
    # figure out how much time I should remove.
    total_time = np.nansum(mydb['time_tot'][mydb['array'] == array])
    replace_time = total_time * time_frac
    
    # should probably do something smarter with band selection here
    idx = (mydb['array'] == array) & ((mydb['band'] == 3.0) | (mydb['band'] == 6.0) | (mydb['band'] == 7.0))
    nrows = len(mydb[idx])

    # initialize random number generators
    rng = np.random.default_rng()
    
    ## set up counters
    time_accum = 0.0 * u.s
    myrows = []

    # go through and remove projects
    while time_accum <= replace_time:

        # high = 1+the max value generated so nrows is right here.
        myrow_tmp = rng.integers(low=0,high=nrows,size=1)

        # if we haven't already picked the row
        if myrow_tmp not in myrows:

            #add the row to the list of rows removed
            myrows.extend(myrow_tmp)
            
            # add to the accumulate time
            time_accum = time_accum + mydb[idx][myrow_tmp]['time_tot'][0]
            
            # match the mous for the given row to the main data base
            idx_match = newdb['mous'] == mydb[idx][myrow_tmp]['mous'][0]
            
            # remove the row from the original data base.
            # A bit of hack that relies on np interpreting a boolean True as 1
            newdb.remove_rows(np.argmax(idx_match))

    # get the number of new rows in the data base
    new_nrows = len(newdb)

    # difference in number of rows in database.
    diff = orig_nrows - new_nrows
    diff_time = time_accum ## or do I want to calculate?
    
    # print out some diagnostics
    print('---')
    print("Total number of MOUSes:", orig_nrows)
    print("Total number of MOUSes meeting the criteria:", nrows)
    print("Number of MOUSes removed:", diff)
    print("New number of MOUSes:", new_nrows)
    print("\n")
    print("Total time:", total_time.to('hr'))
    print("Time to be replaced:", replace_time.to('hr'))
    print("Time replaced:", time_accum.to('hr'))
    print('---')
    
    return newdb, time_accum 


def add_bands(mydb, array='12m', band=1.0, total_time=260*u.hr):
    '''
    Purpose: Add band1 and band2 information into data base by
    scaling from band 3.

    Input: database with replacement projects removed

    Output: updated database with estimates for new bands. Do I want to output everything
    or just the new projects?

    Method:

    Date        Programmer      Description of Changes
    --------------------------------------------------------
    9/20/2023  A.A. Kepley     Original Code
    '''

    # fiducial band 1 and 2 frequencies
    band1_freq = 39.0 * u.GHz
    band2_freq = 75.0 * u.GHz

    # copy old data base to new database
    newdb = mydb.copy()

    # calculate number of rows in original data base.
    orig_nrows = len(mydb)

    # select relevant portions of the data base
    idx = (mydb['array'] == array) & (mydb['band'] == 3.0)
    nrows = len(mydb[idx])

    # initialize random number generator
    rng = np.random.default_rng()

    ## set up counters
    time_accum = 0.0 * u.s
    myrows = []
    db_update = None
    
    ## need to convert total_time to seconds
    while time_accum <= total_time.to('s'):
        myrow_tmp = rng.integers(low=0,high=nrows,size=1)

        if myrow_tmp not in myrows:
            myrows.extend(myrow_tmp)
            time_accum = time_accum + mydb[idx][myrow_tmp]['time_tot'][0]

            ## create new info with band 1 or 2 info
            old_info = mydb[idx][myrow_tmp]
            new_info = old_info
            
            new_info['band'] = band
            new_info['cycle'] = 'estimate'

            # setup the band 1 and 2 properties.
            if band == 1.0:
                new_info['wsu_freq'] = band1_freq
                new_info['wsu_bandwidth_early'] = 8.0*u.GHz ## originally had this as 16.0GHz, but based on data rate ramp up, needs to be 8GHz for early.
                new_info['wsu_bandwidth_later_2x'] = 16.0*u.GHz
                new_info['wsu_bandwidth_later_4x'] = 16.0*u.GHz ## Can't be upgraded beyond 16 GHz.
                
            if band == 2.0:
                new_info['wsu_freq'] = band2_freq
                new_info['wsu_bandwidth_early'] = 16.0*u.GHz
                new_info['wsu_bandwidth_later_2x'] = 16.0*u.GHz
                new_info['wsu_bandwidth_later_4x'] = 32.0*u.GHz ## Can get 32GHz.

            # scale frequency dependent quantities.
            scale_factor = old_info['wsu_freq'] / new_info['wsu_freq']
            
            new_info['s_fov'] = old_info['s_fov'] * scale_factor
            new_info['s_resolution'] = old_info['s_resolution'] * scale_factor
            new_info['pb'] = old_info['pb'] * scale_factor
            new_info['cell'] = old_info['cell'] * scale_factor

            # if this is the first new entry, just create a new table.
            # otherwise add to existing table.
            if db_update is None:                
                db_update = QTable(new_info)
            else:
                db_update = vstack([db_update,new_info])


    # removing irrelevant BLC columns. This works better than blanking.
    db_update.remove_columns(['blc_npol','blc_nspw','blc_specwidth','blc_freq',
                              'blc_nchan_agg','blc_nchan_max',
                              'blc_bandwidth_max','blc_bandwidth_agg'])

    # calculating the number of spectral windows this way allows me to easily propagate any spw related changes.    
    db_update['wsu_nspw_early'] = np.round(db_update['wsu_bandwidth_early']/db_update['wsu_bandwidth_spw']).value
    db_update['wsu_nspw_later_2x'] = np.round(db_update['wsu_bandwidth_later_2x']/db_update['wsu_bandwidth_spw']).value
    db_update['wsu_nspw_later_4x'] = np.round(db_update['wsu_bandwidth_later_4x']/db_update['wsu_bandwidth_spw']).value


    #calculating new specwidth, velres, and chanavg. 
    for veltype in ['finest','stepped','stepped2']:
        vel_res_tmp = np.round(db_update['wsu_velres_'+veltype],1) ## need the round to get the steps ## has units attachd.
        specwidth_tmp = ((vel_res_tmp / const.c.to('km/s')) * db_update['wsu_freq'].to('Hz')).to('kHz')
        specwidth_wsu, chanavg_wsu = calc_talon_specwidth(specwidth_tmp.value,band,vel_res_tmp.value) ## kHz, nchan
        wsu_velres = ((specwidth_wsu * 1e3) /(db_update['wsu_freq'].to('Hz').value)) * const.c.to('km/s').value ## km/z

        # save the nbin, specwidth, and velrest
        db_update['wsu_chanavg_'+veltype] = chanavg_wsu 
        db_update['wsu_specwidth_'+veltype] = specwidth_wsu *u.kHz
        db_update['wsu_velres_'+veltype] = wsu_velres * u.km/u.s
        
        # cap the amount of averaging for bands 1 and 2 to match the peak data rate memo
        # Not needed any more since I enforced a cap on the minimum nbin
        #idx =  (db_update['wsu_chanavg_'+veltype] < 2.0) & (db_update['band'] == 2.0)
        #if np.any(idx):
        #    print("fixing up band2 to have minimum nbin=2")
        #    print("number of lines fixed:", np.sum(idx))
        #    db_update['wsu_chanavg_'+veltype][idx] = np.full(len(db_update[idx]), 2.0)
        #    db_update['wsu_specwidth_'+veltype][idx] = db_update['wsu_specwidth_'+veltype][idx] * 2.0
        #    db_update['wsu_velres_'+veltype][idx] = db_update['wsu_velres_'+veltype][idx] * 2.0
            
        
        db_update['wsu_nchan_spw_'+veltype] = np.floor((db_update['wsu_bandwidth_spw']/db_update['wsu_specwidth_'+veltype]).decompose())

        db_update['wsu_nchan_agg_'+veltype+'_early'] = db_update['wsu_nchan_spw_'+veltype] * db_update['wsu_nspw_early']
        db_update['wsu_nchan_agg_'+veltype+'_later_2x'] = db_update['wsu_nchan_spw_'+veltype] * db_update['wsu_nspw_later_2x']
        db_update['wsu_nchan_agg_'+veltype+'_later_4x'] = db_update['wsu_nchan_spw_'+veltype] * db_update['wsu_nspw_later_4x']
        

    # calculating fractional banddwidth
      
    db_update['wsu_frac_bw_early'] = db_update['wsu_bandwidth_early']/db_update['wsu_freq']
    db_update['wsu_frac_bw_later_2x'] = db_update['wsu_bandwidth_later_2x']/db_update['wsu_freq']
    db_update['wsu_frac_bw_later_4x'] = db_update['wsu_bandwidth_later_4x']/db_update['wsu_freq']
    db_update['wsu_frac_bw_spw'] = db_update['wsu_bandwidth_spw']/db_update['wsu_freq']
    
        
    ## calculate data rates, data volumes, and cube sizes
    add_rates_to_db(db_update, wsu_only=True, permous=True)

    # calculate required system performance
    calc_sysperf(db_update,
                 label='allgrid',
                 mosaic_aproject=True,
                 wproject=True)

    calc_sysperf(db_update,
                 label='aprojonly',
                 mosaic_aproject=True,
                 wproject=False)


    
    # get rid of the full pol cases requesting the highest resolution.
    ## these are rare, but possible.
    bad_idx = ( ((db_update['band'] == 1.0) & (db_update['wsu_chanavg_stepped2'] == 1.0) & (db_update['wsu_npol'] == 4)) |
                ((db_update['band'] == 2.0) & (db_update['wsu_chanavg_stepped2'] == 2.0) & (db_update['wsu_npol'] == 4)) )
    if np.any(bad_idx):
        #ipdb.set_trace()
        print("****removing too high a data rate****")
        print("number of idx:", np.sum(bad_idx))
        idx = np.invert(bad_idx)
        db_update = db_update[idx]

    return db_update
    

def generate_db_realizations(mydb, outDir='data/sample_band1_band2',filename='test',
                             frac_12m=0.1, frac_7m=0.06, n=3):
    '''
    generate realizations of data base so I can calculate statistics

    frac_12m: fraction of 12m data to remove (band 1 and 2 combined)

    frac_7m: fraction of 7m data to remove (band1 and 2 combined)
    
    n: number of iterations

    filename: name of resulting data base

    outDir: directory to put data bases
    
    ## TODO:
    might want to explicitly remove variables at the end of each loop to avoid issues.
    
    
    Date        Programmer       Description of Changes
    ---------------------------------------------------
    10/2/2023   A.A. Kepley     Original Code
    '''

    import os

    if not os.path.exists(outDir):
        os.mkdir(outDir)
    
    for i in range(n):
        
        outfilename = "{0:s}_{1:03d}.ecsv".format(filename,i)
        outfile = os.path.join(outDir,outfilename)
        print('-------------')
        print(outfile)
        
        (new_db_12m, time_12m) = remove_projects(mydb,array='12m',time_frac=0.1)
        (new_db_12m_7m,time_7m) = remove_projects(new_db_12m,array='7m',time_frac=0.06)

        
        db_update_band1_12m = add_bands(new_db_12m_7m, array='12m', band=1.0,total_time=time_12m.to('hr')/2.0)        
        db_update_band2_12m = add_bands(new_db_12m_7m, array='12m', band=2.0,total_time=time_12m.to('hr')/2.0)

        db_update_band1_7m = add_bands(new_db_12m_7m, array='7m', band=1.0,total_time=time_7m.to('hr')/2.0)
        db_update_band2_7m = add_bands(new_db_12m_7m, array='7m', band=2.0,total_time=time_7m.to('hr')/2.0)

        wsu_all_band1_band2 = vstack([new_db_12m_7m, db_update_band1_12m, db_update_band2_12m, db_update_band1_7m, db_update_band2_7m])

        # update the data weights.
        wsu_all_band1_band2['weights_all'] = wsu_all_band1_band2['time_tot']/np.nansum(wsu_all_band1_band2['time_tot'])

        # write out the new file
        wsu_all_band1_band2.write(outfile,overwrite=True)
        
        
        
def calculate_dist(outDir='data/sample_band1_band2',
                   filename='wsu_datarates_mit_per_mous_band12_20231002',             
                   nbins=500):
    
    '''
    calculate the max, median, min of the cumulative distribution functions.

    outDir: directory with samples

    filename: baseline for samples

    quantity: quantity to calculate the cumulative distribution for/

    bins: bins to calculate the cumulative distribution for. Otherwise, just use
    the first data set to generate the bins.

    TODO: modify to calculate multiple quantities and save as astropy table?

    Date        Programmer      Description of Changes
    --------------------------------------------------
    10/2/2023   A.A. Kepley     Original code
    '''

    import glob
    import os
    import re
    
    sample_list = glob.glob(os.path.join(outDir,filename+"_*.ecsv"))

    quantity_list = ['wsu_cubesize_stepped2',
                     'wsu_productsize_early_stepped2',                                  
                     'wsu_datarate_early_stepped2_typical', # typical number of antennas
                     'wsu_visrate_early_stepped2_typical',                                  
                     'wsu_datavol_early_stepped2_typical_target_tot',
                     'wsu_datavol_early_stepped2_typical_cal',
                     'wsu_datavol_early_stepped2_typical_total',
                     #'wsu_nvis_early_stepped2_typical_target_tot',
                     #'wsu_nvis_early_stepped2_typical_cal',
                     #'wsu_nvis_early_stepped2_typical_total',
                     #'wsu_productsize_later_2x_stepped2',                                  
                     #'wsu_datarate_later_2x_stepped2_typical',
                     #'wsu_visrate_later_2x_stepped2_typical',                                  
                     #'wsu_datavol_later_2x_stepped2_typical_target_tot',
                     #'wsu_datavol_later_2x_stepped2_typical_cal',
                     #'wsu_datavol_later_2x_stepped2_typical_total',
                     #'wsu_nvis_later_2x_stepped2_typical_target_tot',
                     #'wsu_nvis_later_2x_stepped2_typical_cal',
                     #'wsu_nvis_later_2x_stepped2_typical_total',
                     'wsu_productsize_later_4x_stepped2',                                  
                     'wsu_datarate_later_4x_stepped2_typical',
                     'wsu_visrate_later_4x_stepped2_typical',                                  
                     'wsu_datavol_later_4x_stepped2_typical_target_tot',
                     'wsu_datavol_later_4x_stepped2_typical_cal',
                     'wsu_datavol_later_4x_stepped2_typical_total',
                     #'wsu_nvis_later_4x_stepped2_typical_target_tot',
                     #'wsu_nvis_later_4x_stepped2_typical_cal',
                     #'wsu_nvis_later_4x_stepped2_typical_total',                     
                     #'wsu_datarate_early_stepped2_array', ## peak number of antenna (total array)
                     #'wsu_visrate_early_stepped2_array',                                  
                     #'wsu_datavol_early_stepped2_array_target_tot',
                     #'wsu_datavol_early_stepped2_array_cal',
                     #'wsu_datavol_early_stepped2_array_total',
                     #'wsu_nvis_early_stepped2_array_target_tot',
                     #'wsu_nvis_early_stepped2_array_cal',
                     #'wsu_nvis_early_stepped2_array_total',
                     #'wsu_productsize_later_2x_stepped2',                                  
                     #'wsu_datarate_later_2x_stepped2_array',
                     #'wsu_visrate_later_2x_stepped2_array',                                  
                     #'wsu_datavol_later_2x_stepped2_array_target_tot',
                     #'wsu_datavol_later_2x_stepped2_array_cal',
                     #'wsu_datavol_later_2x_stepped2_array_total',
                     #'wsu_nvis_later_2x_stepped2_array_target_tot',
                     #'wsu_nvis_later_2x_stepped2_array_cal',
                     #'wsu_nvis_later_2x_stepped2_array_total',
                     'wsu_productsize_later_4x_stepped2',                                  
                     #'wsu_datarate_later_4x_stepped2_array',
                     #'wsu_visrate_later_4x_stepped2_array',                                  
                     #'wsu_datavol_later_4x_stepped2_array_target_tot',
                     #'wsu_datavol_later_4x_stepped2_array_cal',
                     #'wsu_datavol_later_4x_stepped2_array_total',
                     #'wsu_nvis_later_4x_stepped2_array_target_tot',
                     #'wsu_nvis_later_4x_stepped2_array_cal',
                     #'wsu_nvis_later_4x_stepped2_array_total',
                     'blc_sysperf_typical_allgrid',
                     #'flops_per_vis_allgrid',
                     'wsu_sysperf_early_stepped2_typical_allgrid',
                     'wsu_sysperf_later_2x_stepped2_typical_allgrid',
                     'wsu_sysperf_later_4x_stepped2_typical_allgrid',
                     'blc_sysperf_typical_aprojonly',
                     #'flops_per_vis_aprojonly',
                     'wsu_sysperf_early_stepped2_typical_aprojonly',
                     'wsu_sysperf_later_2x_stepped2_typical_aprojonly',
                     'wsu_sysperf_later_4x_stepped2_typical_aprojonly']



    myresults = {}
    myresults_hist = {}
    myresults_hist_log = {}
    myresults_all = {}
    
    for i in np.arange(len(sample_list)):
        mydb = Table.read(sample_list[i])        

        # get column information
        if i == 0:        
            mycolumns = mydb.columns        
            
        for quantity in mycolumns:
            if quantity in quantity_list:
                                        
                ## Getting complementary cumulative histogram information
                ## -----------------------------------------------------
                # add quantity to results keys
                if quantity not in myresults.keys():
                    myresults[quantity] = {}
                    #print("adding "+quantity)
                    
                    
                # do unit conversion on desired quantities.
                if (re.search('productsize',quantity) or re.search('datavol',quantity)):
                    myvals_full = mydb[quantity].to('TB')
                    
                else:                    
                    myvals_full = mydb[quantity]

                    
                myvals = myvals_full.value
                if myvals_full.unit is not None:
                    myresults[quantity]['unit'] = myvals_full.unit.to_string()

                # if this is the first file. set up bins and dist counts arrays
                if i == 0:
                    counts, bins, patches = plt.hist(np.log10(myvals),bins=nbins,cumulative=-1,density=True,log=True,
                                                     weights=mydb['weights_all'])
                        
                    myresults[quantity]['bins'] = bins
                    myresults[quantity]['dist_counts'] = np.empty((len(sample_list),len(counts)))
                    myresults[quantity]['dist_counts'][i,:] = counts
                    
                # otherwise, just make histogram
                else:
                    counts, bins, patches = plt.hist(np.log10(myvals),bins=myresults[quantity]['bins'],cumulative=-1,density=True,log=True,
                                                     weights=mydb['weights_all'])
                    myresults[quantity]['dist_counts'][i,:] = counts


                ## Getting the histogram information
                ##----------------------------------------------
                if quantity in ['wsu_datarate_early_stepped2_typical',
                                'wsu_datarate_later_4x_stepped2_typical']:
                    
                    # regular histogram
                    if quantity not in myresults_hist.keys():
                        myresults_hist[quantity] = {}
                        myresults_hist[quantity]['unit'] = myresults[quantity]['unit'] # copy unit over from other database

                    if i == 0:
                        counts_hist, bins_hist, patches_hist = plt.hist(myvals,bins=nbins,log=False,weights=mydb['weights_all'])
                        
                        myresults_hist[quantity]['bins'] = bins_hist
                        myresults_hist[quantity]['hist_counts'] = np.empty((len(sample_list),len(counts_hist)))
                        myresults_hist[quantity]['hist_counts'][i,:] = counts_hist

                    else:
                        counts_hist, bins_hist, patches_hist = plt.hist(myvals,bins=myresults_hist[quantity]['bins'],
                                                                        log=False,weights=mydb['weights_all'])
                        myresults_hist[quantity]['hist_counts'][i,:] = counts_hist

                    #log histogram                        
                    if quantity not in myresults_hist_log.keys():
                        myresults_hist_log[quantity] = {}
                        myresults_hist_log[quantity]['unit'] = myresults[quantity]['unit'] # copy unit over from other database

                    if i == 0:
                        counts_hist, bins_hist, patches_hist = plt.hist(myvals,bins=nbins,log=True,weights=mydb['weights_all'])
                        
                        myresults_hist_log[quantity]['bins'] = bins_hist
                        myresults_hist_log[quantity]['hist_counts'] = np.empty((len(sample_list),len(counts_hist)))
                        myresults_hist_log[quantity]['hist_counts'][i,:] = counts_hist

                    else:
                        counts_hist, bins_hist, patches_hist = plt.hist(myvals,bins=myresults_hist_log[quantity]['bins'],
                                                                        log=True,weights=mydb['weights_all'])
                        myresults_hist_log[quantity]['hist_counts'][i,:] = counts_hist



    for quantity in myresults.keys():
        # save cumulative histogram results to dictionary
        median_val = np.median(myresults[quantity]['dist_counts'],axis=0)
        max_val = np.max(myresults[quantity]['dist_counts'],axis=0)
        min_val = np.min(myresults[quantity]['dist_counts'],axis=0)
        
        myresults[quantity]['median'] = median_val
        myresults[quantity]['min'] = min_val
        myresults[quantity]['max'] = max_val
    

    for quantity in myresults_hist.keys():
        # save regular histogram results to dictionary
        median_val_hist = np.median(myresults_hist[quantity]['hist_counts'],axis=0)
        max_val_hist = np.max(myresults_hist[quantity]['hist_counts'],axis=0)
        min_val_hist = np.min(myresults_hist[quantity]['hist_counts'],axis=0)
    
        myresults_hist[quantity]['median'] = median_val_hist
        myresults_hist[quantity]['min'] = min_val_hist
        myresults_hist[quantity]['max'] = max_val_hist

        
    for quantity in myresults_hist_log.keys():
        # save log histogram results to dictionary
        median_val_hist_log = np.median(myresults_hist_log[quantity]['hist_counts'],axis=0)
        max_val_hist_log = np.max(myresults_hist_log[quantity]['hist_counts'],axis=0)
        min_val_hist_log = np.min(myresults_hist_log[quantity]['hist_counts'],axis=0)
    
        myresults_hist_log[quantity]['median'] = median_val_hist_log
        myresults_hist_log[quantity]['min'] = min_val_hist_log
        myresults_hist_log[quantity]['max'] = max_val_hist_log

    
    # combine individual dictionaries into one results object.
    myresults_all['hist_cumulative'] = myresults
    myresults_all['hist'] = myresults_hist
    myresults_all['hist_log'] = myresults_hist_log

    return myresults_all



def calc_wsu_stats_allsamples(outDir='data/sample_band1_band2',
                              filename='wsu_datarates_mit_per_mous_band12_20231002'):
    '''
    Purpose: calcuate the statistics over all the realizations of the WSU
    distribution with Bands 1 and 2. 

    Input: mydb

    Outputs: dictionary with summary statistics

    Method: I'm separating out from above where I calculate the distribution
    to get things to run faster.
    
    
    Date        Programmer      Description of Changes
    ----------------------------------------------------
    10/30/2023  A.A. Kepley     Original Code
    
    '''

    import glob
    import os
    import re
    
    sample_list = glob.glob(os.path.join(outDir,filename+"_*.ecsv"))

    quantity_list = ['blc_nchan_agg',
                     'blc_nspw',
                     'blc_cubesize',
                     'blc_productsize',
                     'blc_datarate_typical',
                     'blc_visrate_typical',
                     'blc_datavol_typical_target_tot',
                     'blc_datavol_typical_cal',
                     'blc_datavol_typical_total',                    
                     'wsu_nchan_agg_stepped2_early',
                     'wsu_nchan_agg_stepped2_later_4x',
                     'wsu_nspw_early','wsu_nspw_later_4x',
                     'wsu_cubesize_stepped2',
                     'wsu_productsize_early_stepped2',                                  
                     'wsu_datarate_early_stepped2_typical', # typical number of antennas
                     'wsu_visrate_early_stepped2_typical',                                  
                     'wsu_datavol_early_stepped2_typical_target_tot',
                     'wsu_datavol_early_stepped2_typical_cal',
                     'wsu_datavol_early_stepped2_typical_total',
                     #'wsu_nvis_early_stepped2_typical_target_tot',
                     #'wsu_nvis_early_stepped2_typical_cal',
                     #'wsu_nvis_early_stepped2_typical_total',
                     #'wsu_productsize_later_2x_stepped2',                                  
                     #'wsu_datarate_later_2x_stepped2_typical',
                     #'wsu_visrate_later_2x_stepped2_typical',                                  
                     #'wsu_datavol_later_2x_stepped2_typical_target_tot',
                     #'wsu_datavol_later_2x_stepped2_typical_cal',
                     #'wsu_datavol_later_2x_stepped2_typical_total',
                     #'wsu_nvis_later_2x_stepped2_typical_target_tot',
                     #'wsu_nvis_later_2x_stepped2_typical_cal',
                     #'wsu_nvis_later_2x_stepped2_typical_total',
                     'wsu_productsize_later_4x_stepped2',                                  
                     'wsu_datarate_later_4x_stepped2_typical',
                     'wsu_visrate_later_4x_stepped2_typical',
                     'wsu_datavol_later_4x_stepped2_typical_target_tot',
                     'wsu_datavol_later_4x_stepped2_typical_cal',
                     'wsu_datavol_later_4x_stepped2_typical_total',
                     #'wsu_nvis_later_4x_stepped2_typical_target_tot',
                     #'wsu_nvis_later_4x_stepped2_typical_cal',
                     #'wsu_nvis_later_4x_stepped2_typical_total',                     
                     #'wsu_datarate_early_stepped2_array', ## peak number of antenna (total array)
                     #'wsu_visrate_early_stepped2_array',                                  
                     #'wsu_datavol_early_stepped2_array_target_tot',
                     #'wsu_datavol_early_stepped2_array_cal',
                     #'wsu_datavol_early_stepped2_array_total',
                     #'wsu_nvis_early_stepped2_array_target_tot',
                     #'wsu_nvis_early_stepped2_array_cal',
                     #'wsu_nvis_early_stepped2_array_total',
                     #'wsu_productsize_later_2x_stepped2',                                  
                     #'wsu_datarate_later_2x_stepped2_array',
                     #'wsu_visrate_later_2x_stepped2_array',                                  
                     #'wsu_datavol_later_2x_stepped2_array_target_tot',
                     #'wsu_datavol_later_2x_stepped2_array_cal',
                     #'wsu_datavol_later_2x_stepped2_array_total',
                     #'wsu_nvis_later_2x_stepped2_array_target_tot',
                     #'wsu_nvis_later_2x_stepped2_array_cal',
                     #'wsu_nvis_later_2x_stepped2_array_total',
                     'wsu_productsize_later_4x_stepped2',                                  
                     #'wsu_datarate_later_4x_stepped2_array',
                     #'wsu_visrate_later_4x_stepped2_array',                                  
                     #'wsu_datavol_later_4x_stepped2_array_target_tot',
                     #'wsu_datavol_later_4x_stepped2_array_cal',
                     #'wsu_datavol_later_4x_stepped2_array_total',
                     #'wsu_nvis_later_4x_stepped2_array_target_tot',
                     #'wsu_nvis_later_4x_stepped2_array_cal',
                     #'wsu_nvis_later_4x_stepped2_array_total',
                     #'blc_sysperf_typical_allgrid',
                     #'flops_per_vis_allgrid',
                     #'wsu_sysperf_early_stepped2_typical_allgrid',
                     #'wsu_sysperf_later_2x_stepped2_typical_allgrid',
                     #'wsu_sysperf_later_4x_stepped2_typical_allgrid',
                     'blc_sysperf_typical_aprojonly',
                     #'flops_per_vis_aprojonly',
                     'wsu_sysperf_early_stepped2_typical_aprojonly',
                     'wsu_sysperf_later_2x_stepped2_typical_aprojonly',
                     'wsu_sysperf_later_4x_stepped2_typical_aprojonly']


    myresults_stats = {}

    for i in np.arange(len(sample_list)):
        mydb = Table.read(sample_list[i])        

        # get column information
        if i == 0:        
            mycolumns = mydb.columns        
            
        for quantity in mycolumns:
            if quantity in quantity_list:
            
                ## Getting stats
                ### -------------

                # create dictionary
                if quantity not in myresults_stats.keys():
                    myresults_stats[quantity] = {}

                # initialize dictionary
                if i == 0:

                    if (re.search("productsize",quantity) or re.search("datavol",quantity)):
                        myresults_stats[quantity]['unit'] = 'TB'
                    elif mydb[quantity].unit is not None:
                        myresults_stats[quantity]['unit'] = mydb[quantity].unit.to_string()

                    for myarray in ['12m','7m','both']:
                        myresults_stats[quantity][myarray] = {}
                
                        for mykey in ['mean_arr','wavg_arr','median_arr','max_arr']:
                            myresults_stats[quantity][myarray][mykey] = np.empty(len(sample_list))

                        # also get totals for productsize and data vol
                        if (re.search("productsize",quantity) or re.search("datavol",quantity)):
                            myresults_stats[quantity][myarray]['total_arr'] = np.empty(len(sample_list))

                # fill dictionary
                for myarray in ['both','12m','7m']:
                    if myarray == 'both':
                        idx = (mydb['array'] == '12m') | (mydb['array'] == '7m')
                    else:
                        idx = mydb['array'] == myarray


                    if (re.search("productsize",quantity) or re.search("datavol",quantity)):
                        
                        myresults_stats[quantity][myarray]['mean_arr'][i] = np.nanmean(mydb[quantity][idx].to('TB').value)
                        myresults_stats[quantity][myarray]['wavg_arr'][i] = np.average(mydb[quantity][idx].to('TB').value,weights=mydb[idx]['weights_all'])
                        myresults_stats[quantity][myarray]['median_arr'][i] = np.nanmedian(mydb[quantity][idx].to('TB').value)
                        myresults_stats[quantity][myarray]['max_arr'][i] = np.nanmax(mydb[quantity][idx].to('TB').value)

                        if 'total_arr' in myresults_stats[quantity][myarray].keys():
                            myresults_stats[quantity][myarray]['total_arr'][i] = np.nansum(mydb[quantity][idx].to('TB').value)

                    else:
                        myresults_stats[quantity][myarray]['mean_arr'][i] = np.nanmean(mydb[quantity][idx])
                        myresults_stats[quantity][myarray]['wavg_arr'][i] = np.average(mydb[quantity][idx],weights=mydb[idx]['weights_all'])
                        myresults_stats[quantity][myarray]['median_arr'][i] = np.nanmedian(mydb[quantity][idx])
                        myresults_stats[quantity][myarray]['max_arr'][i] = np.nanmax(mydb[quantity][idx])

                        if 'total_arr' in myresults_stats[quantity][myarray].keys():
                            myresults_stats[quantity][myarray]['total_arr'][i] = np.nansum(mydb[quantity][idx])

    # calculate the overall instances
    for quantity in myresults_stats.keys():

        for myarray in ['12m','7m','both']:
            

            myresults_stats[quantity][myarray]['mean_mean'] = np.mean(myresults_stats[quantity][myarray]['mean_arr'])
            myresults_stats[quantity][myarray]['mean_std'] = np.std(myresults_stats[quantity][myarray]['mean_arr'])
            
            myresults_stats[quantity][myarray]['wavg_mean'] = np.mean(myresults_stats[quantity][myarray]['wavg_arr'])
            myresults_stats[quantity][myarray]['wavg_std'] = np.std(myresults_stats[quantity][myarray]['wavg_arr'])
    
            myresults_stats[quantity][myarray]['median_mean'] = np.mean(myresults_stats[quantity][myarray]['median_arr'])
            myresults_stats[quantity][myarray]['median_std'] = np.std(myresults_stats[quantity][myarray]['median_arr'])
            
            myresults_stats[quantity][myarray]['max_mean'] = np.mean(myresults_stats[quantity][myarray]['max_arr'])
            myresults_stats[quantity][myarray]['max_std'] = np.std(myresults_stats[quantity][myarray]['max_arr'])
            
            if 'total_arr' in myresults_stats[quantity][myarray].keys():
                myresults_stats[quantity][myarray]['total_mean'] = np.mean(myresults_stats[quantity][myarray]['total_arr'])
                myresults_stats[quantity][myarray]['total_std'] = np.std(myresults_stats[quantity][myarray]['total_arr'])
            

    return myresults_stats
                

def calc_sysperf(mydb,
                 mosaic_aproject=True,
                 wproject=True,
                 label='allgrid',
                 multiscale_factor = 1.2, # default for ngVLA SoC
                 core_efficiency = 0.05, # core_efficiency; value from ngVLA SoC -- measured number
                 parallelization_efficiency = 0.8, # parallelization_efficiency; value from ngVLA SoC -- assumption
                 k = 20 # total number of major cycles over all re-runs; my standard value
                 ):
    '''
    Purpose: calculate the fraction of time spent for each MOUS and the required system performance

    Inputs: mydb

    Output: mydb with fractions and estimates of the required system performance

    Date        Programmer      Description of Changes
    ----------------------------------------------------------------------
    2/27/2023   A.A. Kepley     Original Code
    3/21/2023   A.A. Kepley     updated to do actual calculation rather than scale
    10/25/2023  A.A. Kepley     Updated to treat mosaics as a-project
    '''


    # flops/vis for standard convolution kernel 
    flops_per_vis_std = 1280.8  # from ngVLA SoC
    flops_per_vis_aproj = 7472.8  # from ngVLA SoC
    flops_per_vis_wproj = 21768.4 # from ngVLA SoC
    flops_per_vis_awproj = 39704.8 # from ngVLA SoC
    
    flops_per_vis_arr = np.full(len(mydb), flops_per_vis_std)    

    if mosaic_aproject and wproject:

        # if mosaic, use aproject value - most values
        idx = (mydb['mosaic'] == 'T')
        flops_per_vis_arr[idx] = np.full(len(mydb[idx]),flops_per_vis_aproj)

        # if band 1 and long baseline use wproject ## 2nd most values
        idx = (mydb['band'] == 1 ) & (mydb['L80'].value > 6200.0)
        flops_per_vis_arr[idx] = np.full(len(mydb[idx]),flops_per_vis_wproj)
        
        # if mosaic and in band 1 at configurations greater than C43-9 (L80 ~ 6400), use aw- project -- 3 most values
        idx = (mydb['mosaic'] == 'T') & (mydb['band'] == 1 ) & (mydb['L80'].value > 6200.0) # L80 in m
        flops_per_vis_arr[idx] = np.full(len(mydb[idx]),flops_per_vis_awproj)


    elif mosaic_aproject and not wproject:
        
        # if mosaic, use aproject value
        idx = (mydb['mosaic'] == 'T')
        flops_per_vis_arr[idx] = np.full(len(mydb[idx]),flops_per_vis_aproj)


    elif not mosaic_aproject and wproject:

        # if band 1 and long baseline use wproject
        idx = (mydb['band'] == 1 ) & (mydb['L80'].value > 6200.0) # L80 in m
        flops_per_vis_arr[idx] = np.full(len(mydb[idx]),flops_per_vis_wproj)

    else:
        print("not applying aproject or wproject")
        
    # multi-scale factor
    multiscale_factor_arr = np.full(len(mydb), multiscale_factor) # ngVLA SoC uses 1.2
    
    # nterms factor = 1 because we are doing cubes
    nterms_factor_arr = np.ones(len(mydb))                
    
    # calculate system performance
    visrate_list = ['blc_visrate_typical',
                    'wsu_visrate_early_stepped2_typical',
                    'wsu_visrate_later_2x_stepped2_typical',
                    'wsu_visrate_later_4x_stepped2_typical']
    
    for visrate in visrate_list :

        sysperf = (mydb[visrate].value * 1e9 / 3600.0 ) *  \
            flops_per_vis_arr * k * \
            multiscale_factor_arr * nterms_factor_arr / \
            (core_efficiency * parallelization_efficiency)
        
        outname = visrate.replace('visrate','sysperf') + '_'+label
        mydb[outname]  = sysperf / 1e15 # convert to PFLOPS
        mydb['flops_per_vis_'+label] = flops_per_vis_arr # save the flops per vis
