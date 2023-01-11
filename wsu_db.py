from astropy.table import Table, QTable, join, unique
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
    blc_tint = []

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
            
            # array info
            array_list.append(array)
            
            if array == '12m':
                nant_typical = 47
                nant_array = 54
                nant_all = 66 #12m+7m+TP
                wsu_tint = 3.024 #s
            elif array == '7m':
                nant_typical = 10
                nant_array = 12
                nant_all = 16 # total power plus 7m
                wsu_tint = 10.08 #s

            nant_typical_list.append(nant_typical)
            nant_array_list.append(nant_array)
            nant_all_list.append(nant_all)
            wsu_tint_list.append(wsu_tint)
            
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

            
            # WSU spectral resolution
            # -----------------------------
            
            # I believe that spec_width is what i want because that is the spectral 
            # resolution which is greater than the channel spacing for cases where 
            # averaging isn't happening for the channels
            
            ## finest
            (specwidth_finest_talon, chanavg_finest_talon) = calc_talon_specwidth(specwidth_finest)
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
            (specwidth_stepped_talon,chanavg_stepped_talon) = calc_talon_specwidth(specwidth_tmp)
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
            (specwidth_stepped2_talon,chanavg_stepped2_talon) = calc_talon_specwidth(specwidth_tmp)
            wsu_specwidth_stepped2.append(specwidth_stepped2_talon)
            wsu_chanavg_stepped2.append(chanavg_stepped2_talon) 

            velres_stepped2_tmp = (specwidth_stepped2_talon * 1e3 / (freq*1e9)) * const.c.to('km/s').value
            wsu_velres_stepped2.append(velres_stepped2_tmp)
                                    
            # WSU BW
            # ------------

            # get band information
            band_list = np.unique(cycle7tab[idx]['band_list'])
            if len(band_list) > 1:
                print("multiple bands in same MOUS " + mymous)
            band_list_array.append(band_list) ## is append going to cause problems here
            
            # each spw likely to have 1.6GHz BW -- based on 1st F at antenna
            spw_bw = 1.6 # GHz 
            wsu_bandwidth_spw.append(spw_bw)

            # but at beginning only band 6 and band 2 will be upgraded. Band 2 is under dev now, so no band 2 in cycle 7.
            if band_list == 6:
                bw = 16.0
            elif (band_list >= 3) & (band_list <= 8) & (band_list != 6):
                bw = 8.0
            elif (band_list >= 9 & band_list <= 10):
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
    if_mous_tab = QTable([np.squeeze(if_mous_list), np.squeeze(proposal_id_list), np.squeeze(array_list), 
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
                         names=('mous','proposal_id','array',
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
    
    if_mous_tab['wsu_frac_bw_early'] = if_mous_tab['wsu_bandwidth_early']/if_mous_tab['wsu_freq']
    if_mous_tab['wsu_frac_bw_later_2x'] = if_mous_tab['wsu_bandwidth_later_2x']/if_mous_tab['wsu_freq']
    if_mous_tab['wsu_frac_bw_later_4x'] = if_mous_tab['wsu_bandwidth_later_4x']/if_mous_tab['wsu_freq']
    if_mous_tab['wsu_frac_bw_spw'] = if_mous_tab['wsu_bandwidth_spw']/if_mous_tab['wsu_freq']
    

    # calculate number of baselines for each case.
    # --------------------------------------------    

    for myarray in ['typical','array','all']:
        if_mous_tab['nbase_'+myarray] = if_mous_tab['nant_'+myarray] * (if_mous_tab['nant_'+myarray] -1 )/2.0

    ## TODO: I have a function to do this
    ## calculate cube sizes and total data sizes??
    
        
    return if_mous_tab


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
    tint_12m_long = 2.02 #s ## also see values of 3.02s for some projects
    
    orig_db['blc_tint'] = np.ones(len(orig_db)) * u.s

    # set 7m
    idx = orig_db['array'] == '7m'
    orig_db['blc_tint'][idx] = tint_7m * orig_db['blc_tint'][idx] 
    
    # add 12m values
    idx = (orig_db['array'] == '12m') & (orig_db['L80'] > breakpt_12m)
    orig_db['blc_tint'][idx] = tint_12m_long * orig_db['blc_tint'][idx]

    idx = (orig_db['array'] == '12m') & (orig_db['L80'] <= breakpt_12m )
    orig_db['blc_tint'][idx] = tint_12m_short * orig_db['blc_tint'][idx]



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

def add_rates_to_db(mydb):
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

    #array_list = ['typical','array']
    array_list = ['typical']
    #velres_list = ['stepped','stepped2']
    velres_list = ['stepped2']
        
    # calculate cube sizes, data rates, and data volumes.
    for array in array_list:
        for velres in velres_list:

            # calculate cube sizes
            # only depends on number of channels per spw
            mydb['wsu_cubesize_'+velres] = calc_cube_size(mydb['imsize'], mydb['wsu_nchan_spw_'+velres])

            for stage in ['early','later_2x','later_4x']:

                # calculate product size
                # depends on number of cubes
                mydb['wsu_productsize_'+stage+'_'+velres] = 2.0 * (mydb['wsu_cubesize_'+velres] + mydb['mfssize']) * mydb['wsu_nspw_'+stage]

                # calculate data rates and data volumes for the visibilities
                calc_rates_for_db(mydb,array=array,correlator='wsu',stage=stage,velres=velres)                

        # calculate BLC correlator values
        calc_rates_for_db(mydb,array=array,correlator='blc',stage='',velres='')
                
    
def calc_rates_for_db(mydb, array='typical',correlator='wsu',stage='early', velres='stepped2'):
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
    else:
        Nchannels = mydb[correlator+'_nchan_agg']
        mylabel = array
        
    Tintegration = mydb[correlator+'_tint']

    Npols = mydb[correlator+'_npol']
    
    mydb[correlator+'_datarate_'+mylabel] = calc_datarate(Nbyte, Napc, Nant, Nchannels, Npols, Tintegration) #GB/s
    mydb[correlator+'_visrate_'+mylabel] = calc_visrate(Nant, Npols, Nchannels, Tintegration)  #Gvis/hr

    mydb[correlator+'_datavol_'+mylabel+'_target'] = mydb[correlator+'_datarate_'+mylabel] * mydb['target_time'] # GB/s * s = GB
    mydb[correlator+'_datavol_'+mylabel+'_target_tot'] = mydb[correlator+'_datarate_'+mylabel] * mydb['target_time_tot'] # GB/s * s = GB

    mydb[correlator+'_datavol_'+mylabel+'_cal'] = mydb[correlator+'_datarate_'+mylabel] * mydb['cal_time'] # GB/s * s = GB
    mydb[correlator+'_datavol_'+mylabel+'_total'] = mydb[correlator+'_datarate_'+mylabel] * mydb['time_tot'] # GB/s * s = GB

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

    # define Gvis unit
    gvis = u.def_unit('Gvis')

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
            
    return mous_db



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

    mous_mit_db = join(mous_db,mit_db)

    
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
    for mykey in ['totaltime','imgtime','cubetime','aggtime','fctime']:
        mous_mit_db[mykey].unit = u.hr

    for mykey in ['allowedcubesize','allowedcubelimit','predcubesize','mitigatedcubesize',
                  'allowedprodsize','initialprodsize','mitigatedprodsize']:
        mous_mit_db[mykey].unit = u.GB
    
    # fix mitigated column for cases where there was a warning but no mitigation
    idx = (mous_mit_db['mitigatedprodsize'] == mous_mit_db['initialprodsize']) & (mous_mit_db['mitigated'] == True)
    mous_mit_db['mitigated'][idx] = False

    # change column names to clearly indicate that they are pl times
    for mykey in ['totaltime','imgtime','cubetime','aggtime','fctime']:
        mous_mit_db.rename_column(mykey,'pl_'+mykey)

    # calculate calibration time
    mous_mit_db['pl_caltime'] = mous_mit_db['pl_totaltime'] - mous_mit_db['pl_imgtime']
        
    return mous_mit_db
