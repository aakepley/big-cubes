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
    if_mous_list = []
    proposal_id_list = []
    array_list = []
    ntarget_list = []
    target_name_list = []
    s_fov_list = []
    s_resolution_list = []
    mosaic_list = []
    npol_list = []
    wsu_bandwidth_final = []
    wsu_bandwidth_initial = []
    wsu_bandwidth_spw = []
    band_list_array = []
    tint_list = [] 
    wsu_freq_list = []
    wsu_specwidth_finest = []
    wsu_chanavg_finest = []
    wsu_specwidth_stepped = []
    wsu_chanavg_stepped = []
    wsu_specwidth_stepped2 = []
    wsu_chanavg_stepped2 = []
    nant_typical_list = []
    nant_array_list = []
    nant_all_list = []
    imsize_list = []
    pb_list = []
    cell_list = []
    vel_res_list = []

    ## TODO: MAY WANT TO .EXTEND RATHER THAN NP.APPEND BELOW. MUCH FASTER PERFORMANCE.
    ## NEED TO BE CAREFUL WITH STRINGS.

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

        mytargets = np.unique(cycle7tab[idx_mous]['target_name'])
        ntarget = np.unique(cycle7tab[idx_mous]['ntarget'])
        
        for target_name in mytargets:
            idx = (cycle7tab['member_ous_uid'] == mymous) & (cycle7tab['target_name'] == target_name)

            # MOUS
            if_mous_list = np.append(if_mous_list,mymous)

            # targetname
            target_name_list = np.append(target_name_list,target_name)

            # n targets
            ntarget_list = np.append(ntarget_list,ntarget)
            
            # proposal id 
            proposal_id = np.unique(cycle7tab[idx]['proposal_id'])
            proposal_id_list = np.append(proposal_id_list,proposal_id)

            # array info
            array_list = np.append(array_list,array)

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
            nant_typical_list = np.append(nant_typical_list,nant_typical)
            nant_array_list = np.append(nant_array_list,nant_array)
            nant_all_list = np.append(nant_all_list, nant_all)
            tint_list = np.append(tint_list,tint)
        
            # FOV
            s_fov = np.mean(cycle7tab[idx]['s_fov']) 
            s_fov_list = np.append(s_fov_list,s_fov)
            
            # Resolution
            s_resolution = np.mean(cycle7tab[idx]['s_resolution'])
            s_resolution_list = np.append(s_resolution_list, s_resolution)
    
            # mosaic
            mosaic = np.unique(cycle7tab[idx]['is_mosaic'])
            if len(mosaic) > 1:
                print("mosaic and single pointings in same MOUS " + mymous + ". Setting mosaic to True")
                mosaic = 'T'
            mosaic_list = np.append(mosaic_list,mosaic)
        
            # imsize
            imsize = np.mean(cycle7tab[idx]['imsize'])
            imsize_list = np.append(imsize_list,imsize)
            
            # pb
            pb = np.mean(cycle7tab[idx]['pb'])
            pb_list = np.append(pb_list,pb)

            # cell
            cell = np.mean(cycle7tab[idx]['cell'])
            cell_list = np.append(cell_list,cell)
        
            # polarization states
            pol_states = np.unique(cycle7tab[idx]['pol_states'])
            if len(pol_states) > 1:
                print("print multiple polarization setups in same MOUS " + mymous)
            npol = len(pol_states.data[0].split('/')[1:-1])
            npol_list = np.append(npol_list,npol)
    
            # WSU BW
            # ------------
            # everything will have 16GHz eventually
            wsu_bandwidth_final = np.append(wsu_bandwidth_final, 16.0) 

            # but at beginning only band 6 and band 2 will be upgraded. Band 2 is under dev now, so no band 2 in cycle 7.
            band_list = np.unique(cycle7tab[idx]['band_list'])
            if len(band_list) > 1:
                print("multiple bands in same MOUS " + mymous)
            band_list_array = np.append(band_list_array, band_list)
    
            if band_list == 6:
                wsu_bandwidth_initial = np.append(wsu_bandwidth_initial, 16.0)
            elif (band_list >= 3) & (band_list <= 8) & (band_list != 6):
                wsu_bandwidth_initial = np.append(wsu_bandwidth_initial, 8.0)
            elif (band_list >= 9 & band_list <= 10):
                wsu_bandwidth_initial = np.append(wsu_bandwidth_initial, 16.0)
            else:
                print('Band not recognized for MOUS: ' + mymous)

            # each spw likely to have 1.6GHz BW
            wsu_bandwidth_spw = np.append(wsu_bandwidth_spw,1.6) 
        
            
            # WSU spectral resolution
            # -----------------------------
            
            # I believe that spec_width is what i want because that is the spectral 
            # resolution which is greater than the channel spacing for cases where 
            # averaging isn't happening for the channels

            ## finest
            specwidth_finest = min(cycle7tab[idx]['spw_specwidth'])
            (specwidth_finest_talon, chanavg_finest_talon) = calc_talon_specwidth(specwidth_finest)
            wsu_specwidth_finest = np.append(wsu_specwidth_finest, specwidth_finest_talon )
            wsu_chanavg_finest = np.append(wsu_chanavg_finest, chanavg_finest_talon)

            ## stepped -- 4 steps
            vel_res =  min(((cycle7tab[idx]['spw_specwidth']*1e3) / (cycle7tab[idx]['spw_freq']*1e9)) * const.c.to('km/s')).value
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
            wsu_specwidth_stepped = np.append(wsu_specwidth_stepped,specwidth_stepped_talon)
            wsu_chanavg_stepped = np.append(wsu_chanavg_stepped, chanavg_stepped_talon)

            ## stepped -- 5 steps
            # finer coverage around 1km/s. At band 6 projects often are slightly over 1 km/s to get full bandwidth.
            vel_res =  min(((cycle7tab[idx]['spw_specwidth']*1e3) / (cycle7tab[idx]['spw_freq']*1e9)) * const.c.to('km/s')).value
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
            wsu_specwidth_stepped2 = np.append(wsu_specwidth_stepped2,specwidth_stepped2_talon)
            wsu_chanavg_stepped2 = np.append(wsu_chanavg_stepped2, chanavg_stepped2_talon) 
            
            vel_res_list = np.append(vel_res_list,vel_res)
        
            # frequency -- only approximate
            wsu_freq_list = np.append(wsu_freq_list,np.mean(cycle7tab[idx]['spw_freq']))
    
 
    # put appropriate units on quantities.
    s_fov_list = s_fov_list * u.deg
    s_resolution_list = s_resolution_list * u.arcsec
    wsu_bandwidth_initial = wsu_bandwidth_initial * u.GHz
    wsu_bandwidth_final = wsu_bandwidth_final * u.GHz
    wsu_bandwidth_spw = wsu_bandwidth_spw * u.GHz
    wsu_specwidth_finest = wsu_specwidth_finest * u.kHz
    wsu_specwidth_stepped = wsu_specwidth_stepped * u.kHz
    wsu_specwidth_stepped2 = wsu_specwidth_stepped2 * u.kHz
    vel_res_list = vel_res_list * u.km / u.s
    
    wsu_freq_list = wsu_freq_list * u.GHz
    tint_list = tint_list * u.s
    
    # put together table
    if_mous_tab = QTable([if_mous_list, proposal_id_list, array_list, 
                          nant_typical_list, nant_array_list, nant_all_list, 
                          band_list_array, ntarget_list, target_name_list,
                          s_fov_list, s_resolution_list, mosaic_list,
                          imsize_list, pb_list,cell_list,
                          npol_list,
                          vel_res_list,
                          wsu_freq_list,
                          wsu_bandwidth_initial, wsu_bandwidth_final, wsu_bandwidth_spw,
                          wsu_specwidth_finest, wsu_chanavg_finest, 
                          wsu_specwidth_stepped, wsu_chanavg_stepped,
                          wsu_specwidth_stepped2, wsu_chanavg_stepped2, 
                          tint_list],
                         names=('mous','proposal_id','array',
                                'nant_typical','nant_array','nant_all',
                                'band','ntarget','target_name',
                                's_fov','s_resolution','mosaic',
                                'imsize','pb','cell',
                                'npol',
                                'velocity_resolution_current',
                                'wsu_freq',
                                'wsu_bandwidth_initial','wsu_bandwidth_final','wsu_bandwidth_spw',
                                'wsu_specwidth_finest','wsu_chanavg_finest',
                                'wsu_specwidth_stepped','wsu_chanavg_stepped',
                                'wsu_specwidth_stepped2','wsu_chanavg_stepped2',
                                'tint'))
    
   
    ### IF I CALCULATE IN ORIGINAL DATABASE, DON'T NEED THE BELOW DOWN TO *****
    # calculate some additional quantities
    #pixel_per_beam = 5.0
    #if_mous_tab['cell'] = if_mous_tab['s_resolution']/pixel_per_beam
    
    # calculate primary beam
    #if_mous_tab['pb'] = np.zeros(len(if_mous_tab))                           
    #idx_7m = if_mous_tab['array'] == '7m'
    #if_mous_tab['pb'][idx_7m] = 33.3 * 300.0/if_mous_tab[idx_7m]['wsu_freq'] 
    #idx_12m = if_mous_tab['array'] == '12m'
    #if_mous_tab['pb'][idx_12m] = 19.4 * 300/if_mous_tab[idx_12m]['wsu_freq']
    
    # calculate imsize
    #if_mous_tab['imsize'] = np.zeros(len(if_mous_tab))
    ## if sf:
    # scale factor for FWHM to 0.2. Using math from pipeline: 
    # https://open-bitbucket.nrao.edu/projects/PIPE/repos/pipeline/browse/pipeline/hif/heuristics/imageparams_base.py#990
    #idx_sf = if_mous_tab['mosaic'] == 'F'
    #scale_02pb = 1.1 * (1.12/1.22)*math.sqrt(-math.log(0.2) / math.log(2.0)) 
    #if_mous_tab['imsize'][idx_sf] = (if_mous_tab['s_fov'][idx_sf].to('arcsec') * scale_02pb / if_mous_tab['cell'][idx_sf]).round(decimals=-1)                             
    
    ## if mosaic:
    ## Pipeline logic:
    ## nxpix = int((1.5 * beam_radius_v + xspread) / cellx_v)
    ## I'm adding primary beam radius to the s_fov under the assumption that the s_fov is something like the half power point.
    #idx_mosaic = if_mous_tab['mosaic'] == 'T'
    #if_mous_tab['imsize'][idx_mosaic] = ((if_mous_tab['s_fov'][idx_mosaic].to('arcsec').value + 1.0*if_mous_tab['pb'][idx_mosaic])/if_mous_tab['cell'][idx_mosaic]).round(decimals=-1)
    
    ### *****************
    
    # calculate number of baselines for each case.
    # --------------------------------------------
    
    if_mous_tab['nbase_typical'] = if_mous_tab['nant_typical'] * (if_mous_tab['nant_typical'] -1 )/2.0
    if_mous_tab['nbase_array'] = if_mous_tab['nant_array'] * (if_mous_tab['nant_array'] -1 )/2.0
    if_mous_tab['nbase_all'] = if_mous_tab['nant_all'] * (if_mous_tab['nant_all'] - 1)/2.0
    
    # calculate number of channels
    # ----------------------------

    #### MAX BW
    
    # figure out max allowed channels for full 2x BW
    nchan_max_talon = 14880 * 80 # 80 frequency slices each with 14880 channels
    nchan_max_mous_finest = np.floor(nchan_max_talon / if_mous_tab['wsu_chanavg_finest']) # max channels if averaged
    nchan_max_mous_stepped = np.floor(nchan_max_talon / if_mous_tab['wsu_chanavg_stepped']) # max channels if averaged
    nchan_max_mous_stepped2 = np.floor(nchan_max_talon / if_mous_tab['wsu_chanavg_stepped2']) # max channels if averaged

    # calculate nchan for final bandwidth and finest channels
    if_mous_tab['wsu_nchan_final_finest'] = np.floor((if_mous_tab['wsu_bandwidth_final']/if_mous_tab['wsu_specwidth_finest']).decompose())
    idx = if_mous_tab['wsu_nchan_final_finest'] > nchan_max_mous_finest 
    if np.sum(idx) > 0:
        print("MAX BW, finest: Adjusting number of channels to meet TALON max: " + str(np.sum(idx)))
        if_mous_tab['wsu_nchan_final_finest'][idx] = nchan_max_mous_finest[idx] 

    # calculate nchan for final bandwidth and stepped channels
    if_mous_tab['wsu_nchan_final_stepped'] = np.floor((if_mous_tab['wsu_bandwidth_final']/if_mous_tab['wsu_specwidth_stepped']).decompose())
    idx = if_mous_tab['wsu_nchan_final_stepped'] > nchan_max_mous_stepped 
    if np.sum(idx) > 0:
        print("MAX BW, stepped: Adjusting number of channels to meet TALON max: " + str(np.sum(idx)))        
        if_mous_tab['wsu_nchan_final_stepped'][idx] = nchan_max_mous_stepped[idx] 

    # calculate nchan for final bandwidth and stepped2 channels
    if_mous_tab['wsu_nchan_final_stepped2'] = np.floor((if_mous_tab['wsu_bandwidth_final']/if_mous_tab['wsu_specwidth_stepped2']).decompose())    
    idx =  if_mous_tab['wsu_nchan_final_stepped2'] > nchan_max_mous_stepped2 
    if np.sum(idx) > 0:
        print("MAX BW, stepped2: Adjusting number of channels to meet TALON max: " + str(np.sum(idx)))
        if_mous_tab['wsu_nchan_final_stepped2'][idx] = nchan_max_mous_stepped2[idx]

    #### SPW BW
        
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

    ### INITIAL BW -- not clear how useful this is. Hard to explain.
        
    # calculate n chan for initial bandwidth and finest channels
    if_mous_tab['wsu_nchan_initial_finest'] = np.floor((if_mous_tab['wsu_bandwidth_initial']/if_mous_tab['wsu_specwidth_finest']).decompose())
    
    # fix cases where they go over the total number of nchan for TALON
    idx = if_mous_tab['wsu_nchan_initial_finest'] > nchan_max_mous_finest
    if np.sum(idx) > 0:
        print("INITIAL BW, finest: Adjusting number of channels to meet TALON max: " + str(np.sum(idx)))
        if_mous_tab['wsu_nchan_initial_finest'][idx] = nchan_max_mous_finest[idx] 

    # calculate n chan for inital bandwidth and stepped channels
    if_mous_tab['wsu_nchan_initial_stepped'] = np.floor((if_mous_tab['wsu_bandwidth_initial']/if_mous_tab['wsu_specwidth_stepped']).decompose())

    idx =  if_mous_tab['wsu_nchan_initial_stepped'] > nchan_max_mous_stepped 
    if np.sum(idx) > 0:
        print("INITIAL BW, stepped: Adjusting number of channels to meet TALON max: " + str(np.sum(idx)))
        if_mous_tab['wsu_nchan_initial_stepped'][idx] = nchan_max_mous_stepped[idx] 

    # calculate n chan for inital bandwidth and stepped2 channels
    if_mous_tab['wsu_nchan_initial_stepped2'] = np.floor((if_mous_tab['wsu_bandwidth_initial']/if_mous_tab['wsu_specwidth_stepped2']).decompose())

    idx = if_mous_tab['wsu_nchan_initial_stepped2'] > nchan_max_mous_stepped2
    if np.sum(idx) > 0:
        print("INITIAL BW, stepped2: Adjusting number of channels to meet TALON max: " + str(np.sum(idx)))
        if_mous_tab['wsu_nchan_initial_stepped2'][idx] = nchan_max_mous_stepped2[idx] 

    
    # calculate visibility rate (GVis/Hr)
    # -----------------------------------

    ## initial BW & typical number of antennas
    if_mous_tab['vis_rate_typical_initial_finest'] = (2.0 * if_mous_tab['npol'] * if_mous_tab['nbase_typical'] * if_mous_tab['wsu_nchan_initial_finest']  / 1e9) / (if_mous_tab['tint'].to('hr'))
    if_mous_tab['vis_rate_typical_initial_stepped'] = (2.0 * if_mous_tab['npol'] * if_mous_tab['nbase_typical'] * if_mous_tab['wsu_nchan_initial_stepped']  / 1e9) / (if_mous_tab['tint'].to('hr'))
    if_mous_tab['vis_rate_typical_initial_stepped2'] = (2.0 * if_mous_tab['npol'] * if_mous_tab['nbase_typical'] * if_mous_tab['wsu_nchan_initial_stepped2']  / 1e9) / (if_mous_tab['tint'].to('hr'))

    
    ## initial BW & all antennas in array
    if_mous_tab['vis_rate_array_initial_finest'] = (2.0 * if_mous_tab['npol'] * if_mous_tab['nbase_array'] * if_mous_tab['wsu_nchan_initial_finest']  / 1e9) / (if_mous_tab['tint'].to('hr'))
    if_mous_tab['vis_rate_array_initial_stepped'] = (2.0 * if_mous_tab['npol'] * if_mous_tab['nbase_array'] * if_mous_tab['wsu_nchan_initial_stepped']  / 1e9) / (if_mous_tab['tint'].to('hr'))
    if_mous_tab['vis_rate_array_initial_stepped2'] = (2.0 * if_mous_tab['npol'] * if_mous_tab['nbase_array'] * if_mous_tab['wsu_nchan_initial_stepped2']  / 1e9) / (if_mous_tab['tint'].to('hr'))

    ## initial BW & all antennas together
    if_mous_tab['vis_rate_all_initial_finest'] = (2.0 * if_mous_tab['npol'] * if_mous_tab['nbase_all'] * if_mous_tab['wsu_nchan_initial_finest']  / 1e9) / (if_mous_tab['tint'].to('hr'))
    if_mous_tab['vis_rate_all_initial_stepped'] = (2.0 * if_mous_tab['npol'] * if_mous_tab['nbase_all'] * if_mous_tab['wsu_nchan_initial_stepped']  / 1e9) / (if_mous_tab['tint'].to('hr'))
    if_mous_tab['vis_rate_all_initial_stepped2'] = (2.0 * if_mous_tab['npol'] * if_mous_tab['nbase_all'] * if_mous_tab['wsu_nchan_initial_stepped2']  / 1e9) / (if_mous_tab['tint'].to('hr'))

    ## final BW & typical number of antennas
    if_mous_tab['vis_rate_typical_final_finest'] = (2.0 * if_mous_tab['npol'] * if_mous_tab['nbase_typical'] * if_mous_tab['wsu_nchan_final_finest']  /1e9) / (if_mous_tab['tint'].to('hr'))
    if_mous_tab['vis_rate_typical_final_stepped'] = (2.0 * if_mous_tab['npol'] * if_mous_tab['nbase_typical'] * if_mous_tab['wsu_nchan_final_stepped']  /1e9) / (if_mous_tab['tint'].to('hr'))
    if_mous_tab['vis_rate_typical_final_stepped2'] = (2.0 * if_mous_tab['npol'] * if_mous_tab['nbase_typical'] * if_mous_tab['wsu_nchan_final_stepped2']  /1e9) / (if_mous_tab['tint'].to('hr'))

    # final BW & all antennas in array
    if_mous_tab['vis_rate_array_final_finest'] = (2.0 * if_mous_tab['npol'] * if_mous_tab['nbase_array'] * if_mous_tab['wsu_nchan_final_finest']  /1e9) / (if_mous_tab['tint'].to('hr'))
    if_mous_tab['vis_rate_array_final_stepped'] = (2.0 * if_mous_tab['npol'] * if_mous_tab['nbase_array'] * if_mous_tab['wsu_nchan_final_stepped']  /1e9) / (if_mous_tab['tint'].to('hr'))
    if_mous_tab['vis_rate_array_final_stepped2'] = (2.0 * if_mous_tab['npol'] * if_mous_tab['nbase_array'] * if_mous_tab['wsu_nchan_final_stepped2']  /1e9) / (if_mous_tab['tint'].to('hr'))
    
    ## final BW & all antennas together
    if_mous_tab['vis_rate_all_final_finest'] = (2.0 * if_mous_tab['npol'] * if_mous_tab['nbase_all'] * if_mous_tab['wsu_nchan_final_finest']  /1e9) / (if_mous_tab['tint'].to('hr'))
    if_mous_tab['vis_rate_all_final_stepped'] = (2.0 * if_mous_tab['npol'] * if_mous_tab['nbase_all'] * if_mous_tab['wsu_nchan_final_stepped']  /1e9) / (if_mous_tab['tint'].to('hr'))
    if_mous_tab['vis_rate_all_final_stepped2'] = (2.0 * if_mous_tab['npol'] * if_mous_tab['nbase_all'] * if_mous_tab['wsu_nchan_final_stepped2']  /1e9) / (if_mous_tab['tint'].to('hr'))


    ## per SPW & typical number of antennas
    if_mous_tab['vis_rate_typical_final_finest'] = (2.0 * if_mous_tab['npol'] * if_mous_tab['nbase_typical'] * if_mous_tab['wsu_nchan_spw_finest']  /1e9) / (if_mous_tab['tint'].to('hr'))
    if_mous_tab['vis_rate_typical_final_stepped'] = (2.0 * if_mous_tab['npol'] * if_mous_tab['nbase_typical'] * if_mous_tab['wsu_nchan_spw_stepped']  /1e9) / (if_mous_tab['tint'].to('hr'))
    if_mous_tab['vis_rate_typical_final_stepped2'] = (2.0 * if_mous_tab['npol'] * if_mous_tab['nbase_typical'] * if_mous_tab['wsu_nchan_spw_stepped2']  /1e9) / (if_mous_tab['tint'].to('hr'))
    
    ## per SPW & all antennas in array
    if_mous_tab['vis_rate_array_final_finest'] = (2.0 * if_mous_tab['npol'] * if_mous_tab['nbase_array'] * if_mous_tab['wsu_nchan_spw_finest']  /1e9) / (if_mous_tab['tint'].to('hr'))
    if_mous_tab['vis_rate_array_final_stepped'] = (2.0 * if_mous_tab['npol'] * if_mous_tab['nbase_array'] * if_mous_tab['wsu_nchan_spw_stepped']  /1e9) / (if_mous_tab['tint'].to('hr'))
    if_mous_tab['vis_rate_array_final_stepped2'] = (2.0 * if_mous_tab['npol'] * if_mous_tab['nbase_array'] * if_mous_tab['wsu_nchan_spw_stepped2']  /1e9) / (if_mous_tab['tint'].to('hr'))

    ## per spw & all antennas together
    if_mous_tab['vis_rate_all_final_finest'] = (2.0 * if_mous_tab['npol'] * if_mous_tab['nbase_all'] * if_mous_tab['wsu_nchan_spw_finest']  /1e9) / (if_mous_tab['tint'].to('hr'))
    if_mous_tab['vis_rate_all_final_stepped'] = (2.0 * if_mous_tab['npol'] * if_mous_tab['nbase_all'] * if_mous_tab['wsu_nchan_spw_stepped']  /1e9) / (if_mous_tab['tint'].to('hr'))
    if_mous_tab['vis_rate_all_final_stepped2'] = (2.0 * if_mous_tab['npol'] * if_mous_tab['nbase_all'] * if_mous_tab['wsu_nchan_spw_stepped2']  /1e9) / (if_mous_tab['tint'].to('hr'))
    
    # fractional bandwidth
    # ---------------------
    
    if_mous_tab['frac_bw_initial'] = if_mous_tab['wsu_bandwidth_initial']/if_mous_tab['wsu_freq']
    if_mous_tab['frac_bw_final'] = if_mous_tab['wsu_bandwidth_final']/if_mous_tab['wsu_freq']
    if_mous_tab['frac_bw_spw'] = if_mous_tab['wsu_bandwidth_spw']/if_mous_tab['wsu_freq']

    return if_mous_tab


def add_tos_to_db(orig_db, tos_db):
    '''
    Purpose: Add time on source for sources and calibrators to data base. Needed
    for size of compute estimate.

    Date        Programmer      Description of Changes
    ----------------------------------------------------------------------
    11/25/2022  A.A. Kepley     Original Code
    '''

    new_db = join(orig_db,tos_db,keys=['mous','target_name'], join_type='left')
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



