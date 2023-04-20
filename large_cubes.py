# Code associated with large cube investigation for PL2023 and WSU
# based on original notebook: largeCube_search_original.ipynb

import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
import re
import math
from astropy.table import Table, QTable, join
import ipdb
#from ast import literal_eval


### NOT SURE THIS IS HELPFUL INFORMATION ANY MORE
# things needed by all functions
band2specscan = {'nchan':595200.0,
                 'imsize':10670.0,
                 'vis_rate_typical': 3063.86,
                 'vis_rate_array':4055.86,
                 'vis_rate_all':6079.54,
                 'freq':75.0,
                 'bandwidth':16.0}

band2specscan_500MBs = {'nchan':148800.0,
                        'imsize':10670.0,
                        'vis_rate_typical': 765.97,
                        'freq':75.0,
                        'bandwidth':16.0}

band2specscan_160MBs = {'nchan': 49600.0,
                        'imsize':10670.0,
                        'vis_rate_typical': 255.32,
                        'freq':75.0,
                        'bandwidth':16.0}


band2specscan['frac_bw'] = band2specscan['bandwidth']/band2specscan['freq']

### TODO -- ADD CONFIG INFO HERE IF WORKS?


def under_to_slash(uid):
    """
    Purpose: 
        Transition a given UID in "___" form to "://" form.

    Inputs: 
        UID : string
            An ALMA UID string (e.g. ASDM, MOUS)

    Outputs:
        UID in opposite form

    Example:
        >>> mous = 'uid___A001_X1341_X1b'
        >>> mous = under_to_slash(mous)
        >>> print(mous)
        uid://A001/X1341/X1b\

    ## CODE BY ANDY LIPNICKY
    
    """
    # If it's empty, return it.
    if not uid:
        return uid
    # If it's in the correct form already, simply return it.
    if '://' in uid:
        return uid
    # Replace all underscores with slashes, split (and thus 
    # remove) the first slash and join it with a colon.
    return ':'.join(uid.replace("_", "/").split("/",1))

def slash_to_under(uid):
    """
    Purpose: 
        Transition a given UID in "://" form to "___" form.

    Inputs: 
        UID : string
            An ALMA UID string (e.g. ASDM, MOUS)

    Outputs:
        UID in opposite form

    Example:
        >>> mous = 'uid://A001/X1341/X1b'
        >>> mous = slash_to_under(mous)
        >>> print(mous)
        uid___A001_X1341_X1b

    ## CODE BY ANDY LIPNICKY
    
    """
    # If it's empty, return it
    if not uid:
        return uid
    # If it's in the correct form already, simply return it.
    if '___' in uid:
        return uid
    # Simply replace all colon and slash chararcters with underscores
    return uid.replace(":", "_").replace("/", "_")

def fix_mous_col(mydb):
    '''
    Purpose: If the mous column has underscores rather than slashes, switch to slashes, so we can join tables.

    
    Date        Programmer      Description of Changes
    --------------------------------------------------- 
    1/9/2023    A.A. Kepley     Original Code
    '''

    orig_mous_list = mydb['mous']
    new_mous_list = []
    for mous in orig_mous_list:
        new_mous_list.append(under_to_slash(mous))
        
    mydb.replace_column('mous',new_mous_list)

       

def get_archive_info(year=2019,filename='test.csv'):
    '''
    get information from archive on projects

    year: year in proposal id for which to search
    filename: output file
    
    Date        Programmer      Description of Changes
    ----------------------------------------------------------------------
    9/14/2022   A.A. Kepley     Original Code
    
    '''

    from astroquery.alma import Alma
    tap_query_str = f"SELECT * FROM ivoa.obscore WHERE proposal_id like '"+str(year)+".%' AND scientific_category NOT like 'Sun' AND scientific_category NOT like 'Solar system' AND scan_intent like 'TARGET' AND science_observation like 'T'"
    
    #result = Alma.query_tap(f"SELECT * FROM ivoa.obscore WHERE proposal_id like '2019.%' AND scientific_category NOT like 'Sun' AND scientific_category NOT like 'Solar system' AND scan_intent like 'TARGET' AND science_observation like 'T'").to_table()

    result = Alma.query_tap(tap_query_str).to_table()
    
    result.remove_columns(['authors','bib_reference','first_author','obs_collection','obs_creator_name','obs_title','proposal_authors','pub_abstract','pub_title','s_region','obs_release_date','lastModified'])

    result.write(filename)
    
    return result

def read_archive_info(filename):
    '''
    read in the information downloaded from the archive and saved to disk

    filename: filename to read information from
    
    Date        Programmer      Description of Changes
    ----------------------------------------------------------------------
    9/14/2022   A.A. Kepley     Original Code
    '''

    result = Table.read(filename)

    return result
    

def munge_archive_info(result,filename, l80_file=None):
    '''
    munge archive info into state that I can use

    TODO: add imsize calculation based on current archive description
    
    Date        Programmer      Description of Changes
    ----------------------------------------------------------------------
    9/14/2022   A.A. Kepley     Original Code
    '''

    from astropy import constants as const

    # get rid of weird sb
    idx = result['schedblock_name'] == 'DO_NOT_OBSERVE'
    if np.sum(idx) > 0:
        result.remove_rows(idx)

    # get rid of planets which mess up the fov calculation
    idx = result['target_name'] == 'Europa'
    result.remove_rows(idx)

    # get rid of comets, which mess of fov claculation
    idx = result['target_name'] == 'PANSTARRS_C2021_O3'
    result.remove_rows(idx)

    idx = result['target_name'] == 'PANSTARRS_C2017_K2'
    result.remove_rows(idx)
        
    # determining array
    array = np.full(np.shape(result['s_fov']),'',dtype='<U4')
    for i in np.arange(len(result)):
        myrow = result[i]
        if re.search('7M',myrow['schedblock_name']):
            array[i] = '7m'
        elif (re.search('TM1',myrow['schedblock_name']) or re.search('TM2',myrow['schedblock_name'])):
            array[i] = '12m'
        elif (re.search('TP',myrow['schedblock_name'])):
            array[i] = 'TP'

    if 'array' not in result.columns:
        result.add_column(array,name='array')
    else:
        result.replace_column('array',array)

    # calculating points per fov
    #s_resolution is supposedly in deg, but comparing to the ALMA archive directly it's in arcsec.
    points_per_fov = (result['s_fov']*3600.0/2.0)**2/(result['s_resolution']/2.0)**2
    if 'points_per_fov' not in result.columns:
        result.add_column(points_per_fov,name='points_per_fov')
    else:
        result.replace_column('points_per_fov',points_per_fov)
        
    # calculate frequency information and bandwidth
    freq = const.c.value*(1.0/result['em_min'] + 1.0/result['em_max'])/1e9/2.0 #GHz
    spec_width = (freq / result['em_res_power'])*1e6 #kHz

    if 'spw_freq' not in result.columns:
        result.add_column(freq,name='spw_freq')
    else:
        result.replace_column('spw_freq',freq)
    
    if 'spw_specwidth' not in result.columns:
        result.add_column(spec_width,name='spw_specwidth')
    else:
        result.replace_column('spw_specwidth',spec_width)

    # calculate actual velocity resolution
    result['velocity_resolution'] = const.c.value * (result['spw_specwidth']*1e3)/(result['spw_freq'] *1e9)/1000.0 #km/s
        
    # calculate nchan
    nchan = calculate_nchan(result)
    if 'spw_nchan' in result.columns:
        result.replace_column('spw_nchan',nchan)
    else:
        result.add_column( nchan, name='spw_nchan')
        
    # calculating the primary beam
    result['pb'] = np.zeros(len(result))                           
    idx_7m = result['array'] == '7m'
    result['pb'][idx_7m] = 33.3 * 300.0/result[idx_7m]['spw_freq'] 
    idx_12m = result['array'] == '12m'
    result['pb'][idx_12m] = 19.4 * 300/result[idx_12m]['spw_freq']

    # calculating estimated cell sizes
    pixel_per_beam = 5.0
    result['cell'] = result['s_resolution']/pixel_per_beam

    # calculate imsize
    result['imsize'] = np.zeros(len(result))
    ## if sf:
    # scale factor for FWHM to 0.2. Using math from pipeline: 
    # https://open-bitbucket.nrao.edu/projects/PIPE/repos/pipeline/browse/pipeline/hif/heuristics/imageparams_base.py#990
    idx_sf = result['is_mosaic'] == 'F'
    scale_02pb = 1.1 * (1.12/1.22)*math.sqrt(-math.log(0.2) / math.log(2.0)) 
    #result['imsize'][idx_sf] = (result['s_fov'][idx_sf].to('arcsec') * scale_02pb / result['cell'][idx_sf]).round(decimals=-1)
    result['imsize'][idx_sf] = (result['s_fov'][idx_sf]*3600.0 * scale_02pb / result['cell'][idx_sf]).round(decimals=-1)                              
    ## if mosaic:
    ## Pipeline logic:
    ## nxpix = int((1.5 * beam_radius_v + xspread) / cellx_v)
    ## I'm adding primary beam diameter to the s_fov under the assumption that the s_fov is something like the half power point. From the code, beam_radius_v is really beam_diameter_v
    ## TODO: Ask Crystal about this. Maybe  I should just add 0.5*pb??
    idx_mosaic = result['is_mosaic'] == 'T'
    #result['imsize'][idx_mosaic] = ((result['s_fov'][idx_mosaic].to('arcsec').value + 0.70*result['pb'][idx_mosaic])/result['cell'][idx_mosaic]).round(decimals=-1)
    result['imsize'][idx_mosaic] = ((result['s_fov'][idx_mosaic]*3600.0 + 0.70*result['pb'][idx_mosaic])/result['cell'][idx_mosaic]).round(decimals=-1)
    
    # calculate mitigated projects
    cube_limit = 40 #GB

    mitigated = np.full(np.shape(result['spw_nchan']),False)
    nchan_max = calc_nchan_max_points_per_fov(cube_limit, result['points_per_fov'])
    idx = result['spw_nchan'] > nchan_max
    mitigated[idx] = True

    if 'spw_nchan_max' in result.columns:
        result.replace_column('spw_nchan_max',nchan)
    else:
        result.add_column(nchan_max,name='spw_nchan_max')

    if 'mitigated' in result.columns:
        result.replace_column('mitigated', mitigated)
    else:
        result.add_column(mitigated,name='mitigated')
    
    # calculate failed mitigations for nbin=1
    cube_limit = 60 #GB
    frac_fov = math.log(0.7)/math.log(0.2)

    failed_mitigation = np.full(np.shape(result['spw_nchan']),False)

    nchan_fail = calc_nchan_max_points_per_fov(cube_limit, result['points_per_fov'],
                                nbin=1.0, pixels_per_beam=9.0, frac_fov=math.log(0.7)/math.log(0.2) )
    idx = result['spw_nchan'] > nchan_fail
    
    failed_mitigation[idx] = True

    if 'failed_mitigation_nbin1' in result.columns:
        result.replace_column('failed_mitigation_nbin1',failed_mitigation)
    else:
        result.add_column(failed_mitigation,name='failed_mitigation_nbin1')

    # calculate failed mitigations for nbin=2
    cube_limit = 60 #GB
    frac_fov = math.log(0.7)/math.log(0.2)

    failed_mitigation_nbin2 = np.full(np.shape(result['spw_nchan']),False)

    nchan_fail_nbin2 = calc_nchan_max_points_per_fov(cube_limit, result['points_per_fov'],
                                      nbin=2.0, pixels_per_beam=9.0, frac_fov=math.log(0.7)/math.log(0.2) )
    idx = result['spw_nchan'] > nchan_fail_nbin2
    
    failed_mitigation_nbin2[idx] = True
    
    if 'failed_mitigation_nbin2' in result.columns:
        result.replace_column('failed_mitigation_nbin2',failed_mitigation)
    else:
        result.add_column(failed_mitigation,name='failed_mitigation_nbin2')

    # calculate nspws
    nspw = np.zeros(len(result['frequency_support']))

    for i  in np.arange(len(result['frequency_support'])) :
        freq_support = result[i]['frequency_support']
        nspw[i] = len(freq_support.split('U'))

    # calculate number of sources:
    mous_list = np.unique(result['member_ous_uid'])

    ntarget_arr = np.zeros(len(result['member_ous_uid']))

    for mous in mous_list:
        idx = (result['member_ous_uid'] == mous) 
        ntarget = len(np.unique(result[idx]['target_name']))
        ntarget_arr[idx] = ntarget
        
    if 'ntarget' not in result.columns:
        result.add_column(ntarget_arr,name='ntarget')
    else:
        result.replace_column('ntarget',ntarget_arr)

    # write out results
    result.write(filename,overwrite=True)
        


def calculate_nchan(result):
    '''
    reverse engineer nchan based on technical handbook information

    Date        Programmer      Description of Changes
    ----------------------------------------------------------------------
    9/14/2022   A.A. Kepley     Original Code
    
    '''

    nchan = np.zeros(np.shape(result['array']))

    for i in np.arange(len(result)):
    
        if result[i]['pol_states'] == '/XX/':
            factor = 2
        elif result[i]['pol_states'] == '/XX/XY/YX/YY/':
            factor = 0.5
        else:
            factor = 1
    
        if result[i]['array'] == '12m':
            base_chan = 3840
        elif result[i]['array'] == '7m':
            base_chan = 4096
        elif result[i]['array'] == 'TP':
            base_chan = 4096
        else:
            print('array not found for mous: ', result[i]['member_ous_uid'])
            continue
   
        if (result[i]['bandwidth'] == 2000e6) & (result[i]['spw_specwidth'] > 15000):
            nchan[i] = 128 * factor
        elif (result[i]['bandwidth'] >=  1875e6):
            if result[i]['spw_specwidth'] > 7700.0:
                nchan[i] = (base_chan/16) * factor
            elif result[i]['spw_specwidth'] > 3800:
                nchan[i] = (base_chan / 8) * factor
            elif result[i]['spw_specwidth'] > 1900:
                nchan[i] = (base_chan / 4) * factor
            elif result[i]['spw_specwidth'] > 1100:
                nchan[i] = (base_chan/2) * factor
            elif result[i]['spw_specwidth'] > 900:
                nchan[i] = base_chan * factor
                # single pol smallest resolution
            elif result[i]['spw_specwidth'] > 480:
                nchan[i] = base_chan * 2
            else:
                print('spw_specwidth not recognized: ', result[i]['bandwidth','spw_specwidth'])

        elif (result[i]['bandwidth'] >= 937.5e6) :
            if result[i]['spw_specwidth'] > 3800:
                nchan[i] = (base_chan/16) * factor
            elif result[i]['spw_specwidth'] > 1900:
                nchan[i] = (base_chan/8) * factor
            elif result[i]['spw_specwidth'] > 960:
                nchan[i] = (base_chan/4) * factor
            elif result[i]['spw_specwidth'] > 560:
                nchan[i] = (base_chan/2) * factor
            elif result[i]['spw_specwidth'] > 480:
                nchan[i] = base_chan * factor
                # fix for Lorant's weird project '2019.1.01719.S' and 2019.2.00093.S
            elif (result[i]['spw_specwidth'] > 280.0):
                nchan[i] = base_chan
            else:
                print('spw_specwidth not recognized', result[i]['bandwidth','spw_specwidth','proposal_id','member_ous_uid'])

        elif (result[i]['bandwidth'] >= 468.75e6):
            if result[i]['spw_specwidth'] > 1950:
                nchan[i] = (base_chan/16) * factor
            elif result[i]['spw_specwidth'] > 970:
                nchan[i] = (base_chan/8) * factor
            elif result[i]['spw_specwidth'] > 480:
                nchan[i] = (base_chan/4) * factor
            elif result[i]['spw_specwidth'] > 280:
                nchan[i] = (base_chan/2) * factor
            elif result[i]['spw_specwidth'] > 240:
                nchan[i] = base_chan * factor
            else:
                print('spw_specwidth not recognized', result[i]['bandwidth','spw_specwidth'])

        elif (result[i]['bandwidth'] >= 234.375e6):
            if result[i]['spw_specwidth'] > 970:
                nchan[i] = (base_chan/16) * factor
            elif result[i]['spw_specwidth'] > 480:
                nchan[i] = (base_chan/8) * factor
            elif result[i]['spw_specwidth'] > 240:
                nchan[i] = (base_chan/4) * factor
            elif result[i]['spw_specwidth'] > 138:
                nchan[i] = (base_chan/2) * factor
            elif result[i]['spw_specwidth'] > 118:
                nchan[i] = (base_chan) * factor
            else:
                print('spw_specwidth not recognized', result[i]['bandwidth','spw_specwidth'])

        elif (result[i]['bandwidth'] >= 117.1875e6) :
            if result[i]['spw_specwidth'] > 480:
                nchan[i] = (base_chan/16) * factor
            elif result[i]['spw_specwidth'] > 240:
                nchan[i] = (base_chan/8) * factor
            elif result[i]['spw_specwidth'] > 118:
                nchan[i] = (base_chan/4) * factor
            elif result[i]['spw_specwidth'] > 65:
                nchan[i] = (base_chan/2) * factor
            elif result[i]['spw_specwidth'] > 56:
                nchan[i] = (base_chan) * factor
            else:
                print('spw_specwidth not recognized', result[i]['bandwidth','spw_specwidth'])

        elif (result[i]['bandwidth'] >= 58.593750e6):
            if result[i]['spw_specwidth'] > 240:
                nchan[i] = (base_chan/16) * factor
            elif result[i]['spw_specwidth'] >120:
                nchan[i] = (base_chan/8) * factor
            elif result[i]['spw_specwidth'] > 56:
                nchan[i] = (base_chan/4) * factor
            elif result[i]['spw_specwidth'] > 30:
                nchan[i] = (base_chan/2) * factor
            elif result[i]['spw_specwidth'] > 25:
                nchan[i] = (base_chan) * factor
            else:
                print('spw_specwidth not recognized', result[i]['bandwidth','spw_specwidth'])

    return nchan


def calc_time_on_source(cal_info_file,
                        min_date=2019, debug=False):
    '''
    Read in information about calibration sources and write out a file with total time on
    -- bandpass
    -- check source
    -- phase calibrator
    -- science target

    Date        Programmer      Description of Code
    -----------------------------------------------
    11/23/2022  A.A.Kepley      Original Code
    '''

    import astropy.units as u
    
    t = Table.read(cal_info_file, format='ascii',delimiter='|',
                   guess=False,
                   names=('project_id','mous','band','array','asdm','asdm_size_gb','na1','target_name','intent','tos_s','na2'))

    # bogus extra column due to how things are delimited.
    t.remove_column('na1')
    t.remove_column('na2')

    # get list of mous'es
    mous_list = np.unique(t['mous'])

    # set up output arrays
    project_id_arr = []
    mous_arr = []
    band_arr = []
    array_arr = []
    bp_time_arr = []
    flux_time_arr = []
    phase_time_arr = []
    check_time_arr = []
    pol_time_arr = []
    ntarget_arr = []
    target_name_arr = []
    target_time_arr = []
    target_time_tot_arr = []
    time_tot_arr = []
    cal_time_arr = []
    
    # let's iterate over MOUS
    for mous in mous_list:
        idx_mous = t['mous'] == mous

        # skip TP
        if ( t[idx_mous]['array'][0] == 'TP'):
            if debug:
                print("MOUS from TP array: skipping")
            continue
        else:
            array = t[idx_mous]['array'][0] 
        
        # get project id and skip it if something is weird or it's too old
        project_id = np.unique(t[idx_mous]['project_id'])
        if len(project_id) > 1:
            print("project_id list greater than 1. This shouldn't happen. MOUS:", mous)
            continue
        else:
            if float(project_id[0].split('.')[0]) < min_date:
                if debug:
                    print("project_id less than minimum date")
                continue
        project_id = project_id[0]

        # band
        # TODO: Will this break with b2b? YES. SKIPPING B2B FOR NOW  
        try:
            band = float(t[idx_mous]['band'][0])
        except:
            continue
            
        # TODO:
        ## need to skip those projects that aren't observing modes
        ## we want to focus on. (DIFFGAIN,APPPHASE_ACTIVE BANDPASS AND PHASE project)
            
        # now go through list of ASDMS     
        asdm_list = np.unique(t[idx_mous]['asdm'])

        # initialize values
        bp_time = 0.0
        flux_time = 0.0
        phase_time = 0.0
        pol_time = 0.0
        check_time = 0.0
        target_dict = {}
        target_time_tot = 0.0
        time_tot = 0.0
        cal_time = 0.0
        n_src = 0.0
        
        for asdm in asdm_list:
            idx_mous_asdm = (t['mous'] == mous) & (t['asdm'] == asdm)
                      
            for row in t[idx_mous_asdm]:
                if row['intent'] == 'BANDPASS FLUX WVR':
                    bp_time += row['tos_s']
                elif row['intent'] == 'BANDPASS FLUX POLARIZATION WVR':
                    bp_time += row['tos_s']
                elif row['intent'] == 'BANDPASS WVR':
                    bp_time += row['tos_s']
                elif row['intent'] == 'FLUX WVR':
                    flux_time += row['tos_s']
                elif row['intent'] == 'PHASE WVR':
                    phase_time += row['tos_s']
                elif row['intent'] == 'POLARIZATION WVR':
                    pol_time += row['tos_s']
                elif row['intent'] == 'CHECK WVR':
                    check_time += row['tos_s']
                elif row['intent'] == 'TARGET':
                    target_time_tot += row['tos_s']

                    # add to time on target if source already observed
                    if row['target_name'] in target_dict.keys():
                        target_dict[row['target_name']] += row['tos_s']
                    # add source to dictionary
                    else:
                        target_dict[row['target_name']] = row['tos_s']
                else:
                    print("Intent not recognized: " + row['intent'])

        # calculate total time and number of target sources
        cal_time = bp_time + flux_time + phase_time + pol_time + check_time      
        time_tot = cal_time + target_time_tot 
        n_src = len(target_dict)

        #ipdb.set_trace()

        ## TODO: ADD NEBS?
        
        # add values to array
        project_id_arr.extend([project_id] * n_src)
        mous_arr.extend([mous] * n_src)
        band_arr.extend([band] * n_src)
        array_arr.extend([array] * n_src)
        bp_time_arr.extend([bp_time] * n_src) 
        flux_time_arr.extend([flux_time] * n_src) 
        phase_time_arr.extend([phase_time] * n_src) 
        pol_time_arr.extend([pol_time]*n_src)  
        check_time_arr.extend([check_time] * n_src) 
        target_time_arr.extend(target_dict.values())  # no repeat needed b/c have all sources
        target_name_arr.extend(target_dict.keys()) # no repeat needed b/c have all sources
        target_time_tot_arr.extend([target_time_tot]*n_src)        
        ntarget_arr.extend([n_src]*n_src)
        time_tot_arr.extend([time_tot]*n_src) 
        cal_time_arr.extend([cal_time]*n_src) 

        #ipdb.set_trace()

    print('made it to table creation')

    ## TODO: make the below a Qtable and add appropriate units?
    
    # create final table
    tout = QTable(data=[project_id_arr,
                        mous_arr,
                        band_arr,
                        array_arr,
                        bp_time_arr * u.s,
                        flux_time_arr * u.s,
                        phase_time_arr * u.s,
                        pol_time_arr * u.s,
                        check_time_arr * u.s,
                        target_time_arr * u.s,
                        target_name_arr,
                        target_time_tot_arr * u.s,                       
                        np.array(ntarget_arr,dtype=np.float),
                        time_tot_arr * u.s,
                        cal_time_arr * u.s],
                 names=['proposal_id',
                        'mous',
                        'band',
                        'array',
                        'bp_time',
                        'flux_time',
                        'phase_time',
                        'pol_time',
                        'check_time',
                        'target_time',
                        'target_name',
                        'target_time_tot',
                        'ntarget',
                        'time_tot',
                        'cal_time'])


    return tout

    
def calc_nchan_max_points_per_fov(cube_limit, points_per_fov, pixels_per_beam=25.0, nbin=1.0, chan_limit = 7680.0,frac_fov=1.0):
    '''
    calculate maximum number of channels permitted by cube mitigation limit.
    
    '''
    
    # default: assumes 25 pixels per resolution element
    # 4.0 because fits images are single precision
    # 1.0 = assuming imaging full fov (i.e., out to 0.2) -- AAK (9/14/2022): actually this is only the full FOV, not out to 0.2
    # nbin = 1 -- number of channels binned together
    # add in factor of 2.6 since that gives better results compared to pipeline
    npix = 2.6 * (points_per_fov * frac_fov) * pixels_per_beam * 4.0 
    nchan_max = cube_limit * nbin * 1e9 / npix
    if type(points_per_fov) == float:
        if nchan_max > chan_limit:
            nchan_max = chan_limit
    else:
        nchan_max[nchan_max > chan_limit] = chan_limit
    return nchan_max


def calc_nchan_max(imsize, cube_limit, chan_limit=80*14880):
    '''
    calculate maximum nchan for a given cube limit
  
    imsize: pixels
    limit: in GB
    chan_limit: max number of channels that can be produced by correlator. default is wsu value.

    '''
    
    nchan_max = cube_limit * 1e9 / (4.0*imsize**2)
    
    # if greater than wsu max set to wsu max
    if type(nchan_max) == float:
        if nchan_max > chan_limit:
            nchan_max =  chan_limit
    else:
        nchan_max[nchan_max > chan_limit] = chan_limit
        
    return nchan_max


def calc_cube_size(imsize, nchan):

    cube_size = 4.0 * imsize**2 * nchan /1.0e9 * u.GB

    return cube_size

def calc_mfs_size(imsize):

    mfs_size = 4.0 * imsize**2 /1.0e9 * u.GB

    return mfs_size


def mem_per_plane(ims):
    '''
    Calculate the memory used per image plane, given imsize.  
       Assume float precision.  
      (Double precision and Complex valued images will be integral multiples
       of this basic unit)
    '''
    
    ## Return mem size in GB
    size_float = 4  # bytes
    bytes_per_gb = 1024*1024*1024.0
    one_plane = (ims * ims * 1 * size_float) / bytes_per_gb
    return one_plane



def calc_minor(log_imsize = np.arange(1.5,4.1,0.1), 
               log_nchan = np.arange(3.0,6.1,0.1),
               nproc=100, 
               mem_per_proc=4,
               plotit=False):
    """

    Calculate the total number of iterations, per chunk, per minorcycle set 
    as a function of imsize and nchan per chunk 
    Niter per process is a proxy for minor cycle compute load, for scaling purposes.
    
    The defaults reflect the ALMA Cycle 7 range of imsizes and nchans

    log_imsize = A list of log(imsize) values to cover the range of interest
                 Note that npixels = imsize**2
    log_nchan = A list of log(nchan) values to cover the range of interest
    nproc : Number of processors across which to divide the work
    mem_per_proc : GB : Available memory per processor

    Date        Programmer      Description of Changes
    ----------------------------------------------------------------------
    9/14/2022   Urvashi Rau     Original Code

    """   



    #    Initialize grids to evaluate metrics for all combinations of imsize, nchan
    imsize, nchan = np.meshgrid( 10**(log_imsize), 10**(log_nchan))
    
    #############################################################################
    # 1 # Number of iterations per chan, in one set of minorcycle iterations
    #############################################################################
    #     Assume there is flux in a quarter of the image, 
    #     and 100 iterations are needed per pixel in that inner quarter
    #     This is an overkill, but it can happen.
    niter_per_chan = 0.25 *100  * imsize**2  

    #############################################################################
    # 2 # Calculate the number of channels per chunk 
    #############################################################################
    #     Two criteria determine the number of channels per chunk. 
    #      (a) The number of image planes that fit in the available memory per process
    #      (b) The number of processors available for parallel compute.
    
    # (a)  We need about 10 copies of float-images per plane, per imaging run. 
    mem_chan = 10 * mem_per_plane(imsize)  
    #      Calculate nchan_per_chunk. For small nchan, it could be all of nchan. 
    nchan_per_chunk_memory = np.minimum( mem_per_proc / mem_chan, nchan)
    
    # (b) Try to use all the available processors
    nchan_per_chunk_compute = nchan/nproc
    
    #     The actual number of channels per chunk is the minimum of (a) and (b)
    #     because the chunks need to fit in memory, and it is ok to have more 
    #     chunks than processes.
    nchan_per_chunk = np.minimum(nchan_per_chunk_memory, nchan_per_chunk_compute)
    
    #############################################################################
    # 3 # Calculate the number of iterations per chunk (in units of 1M)
    #############################################################################
    niter_m = niter_per_chan * nchan_per_chunk / 1e+6
    
    
    #############################################################################
    # 4 # Plot it.... 
    #############################################################################
    if plotit:
        fig, axs = plt.subplots(1,3, figsize=(20,5))
        i=0
        for ax in axs:
            if i==0:
                img = ax.imshow(np.sqrt(niter_m),origin='lower',cmap='jet',extent=[np.min(log_imsize),np.max(log_imsize),np.min(log_nchan),np.max(log_nchan)])
                ax.set_title('niter_per_chunk/1e+6\n(nproc=%d, mem_per_proc=%d GB )'%(nproc,mem_per_proc) )
            if i==1:
                img = ax.imshow(np.sqrt(nchan_per_chunk)/1e+3,origin='lower',cmap='jet',extent=[np.min(log_imsize),np.max(log_imsize),np.min(log_nchan),np.max(log_nchan)])
                ax.set_title('nchan_per_chunk/1e+3\n(nproc=%d, mem_per_proc=%d GB )'%(nproc,mem_per_proc) )
                #            img = ax.imshow(sqrt(nchan_per_chunk_memory)/1e+3,origin='lower',cmap='jet',extent=[min(log_imsize),max(log_imsize),min(log_nchan),max(log_nchan)])
                #            ax.set_title('nchan_per_chunk_memory/1e+3\n(nproc=%d, mem_per_proc=%d GB )'%(nproc,mem_per_proc) )
            if i==2:
                img = ax.imshow(np.sqrt(niter_per_chan)/1e+6,origin='lower',cmap='jet',extent=[np.min(log_imsize),np.max(log_imsize),np.min(log_nchan),np.max(log_nchan)])
                ax.set_title('niter_per_chan/1e+6\n(nproc=%d, mem_per_proc=%d GB )'%(nproc,mem_per_proc) )
                #            img = ax.imshow(sqrt(nchan_per_chunk_compute)/1e+3,origin='lower',cmap='jet',extent=[min(log_imsize),max(log_imsize),min(log_nchan),max(log_nchan)])
                #            ax.set_title('nchan_per_chunk_compute/1e+3\n(nproc=%d, mem_per_proc=%d GB )'%(nproc,mem_per_proc) )
            
            ax.set_xticks(log_imsize)
            ax.set_yticks(log_nchan)
            plt.colorbar(img,ax=ax)
            ax.set_xlabel('log(imsize)')
            ax.set_ylabel('log(nchan)')
        i=i+1
        
    return niter_m



def make_compute_load_theory_plot(log_imsize_range = (1.6,4.1),
                                  log_imsize_step = 0.1,
                                  log_nchan_range = (1.7,4.0),
                                  log_nchan_step = 0.1,
                                  nproc=8,
                                  mem_per_proc=32,
                                  mit_threshold=40, #
                                  max_cube_limit=60, #GB
                                  pltname=None
                                  ):
    '''
    compare theoretical computing load calculation to current mitigation limit

    Date        Programmer      Description of Changes
    ----------------------------------------------------------------------
    9/26/2022   A.A. Kepley     Original Code
    
    '''
    
    mit_imsize = np.logspace(log_imsize_range[0],log_imsize_range[1],1000)
    mit_thresh = calc_nchan_max(mit_imsize,mit_threshold,chan_limit=7680.0)
    mit_limit = calc_nchan_max(mit_imsize,max_cube_limit,chan_limit=7680.0)

    # create bins -- add extra end point
    log_imsize_bins = np.arange(log_imsize_range[0], log_imsize_range[1] + log_imsize_step, log_imsize_step)
    log_nchan_bins = np.arange(log_nchan_range[0], log_nchan_range[1] + log_nchan_step, log_nchan_step)

    # calculate values
    niter_m = calc_minor(log_imsize=log_imsize_bins,
                         log_nchan=log_nchan_bins,
                         nproc=nproc,
                         mem_per_proc=mem_per_proc, plotit=False)
    
    # make plot
    fig = plt.figure(figsize=(12,10),edgecolor='white',facecolor='white')
    ax = plt.subplot(111)

    
    ax.imshow(np.sqrt(niter_m),origin='lower',cmap='jet',extent=[np.min(log_imsize_bins),np.max(log_imsize_bins),np.min(log_nchan_bins),np.max(log_nchan_bins)])

    #ax.contour(log_imsize_bins,log_nchan_bins,np.sqrt(niter_m),levels=[146])

    ax.plot(np.log10(mit_imsize),np.log10(mit_thresh),
            color='white',linewidth=4, linestyle=':',
            label='Mitigation Threshold (40GB)')

    ax.plot(np.log10(mit_imsize),np.log10(mit_limit),
            color='white',linewidth=4, linestyle='-',
            label='Mitigation Limit (60GB)')

    ax.legend()
    ax.set_xticks(log_imsize_bins)
    ax.set_yticks(log_nchan_bins)
    ax.tick_params(labelsize=14)
    
    ax.set_xlabel('log imsize (pixels)',size=18)
    ax.set_ylabel('log nchan',size=18)



    if pltname:
        fig.savefig(pltname)
    
    return niter_m
        
                                                                      
    
        

    
def configuration_info():
    '''
    calculate configuration information

    Date        Programmer      Description of Changes
    ----------------------------------------------------------------------
    9/14/2022   A.A. Kepley     Original Code
    '''
    config_res = [3.38,2.30,1.42, 0.918, 0.545, 0.306, 0.211, 0.096, 0.057,0.042]

    config_dict = {}

    i=1
    for res in config_res:
        pb = 6300/100.0
        points_per_fov = (pb)**2 / res**2
        #print(str(i)+","+str(res)+","+str(points_per_fov))
        config_dict['C'+str(i)] = {'res':res, 'points_per_fov': points_per_fov,'pb':pb}
        i=i+1

    return config_dict






def calc_frac_under_lines(h,
                          maxcube=60, #GB
                          mit_limits = np.array([1,10,500]),
                          max_nchans = np.array([band2specscan_160MBs['nchan'],band2specscan_500MBs['nchan'],band2specscan['nchan']]),
                          log_imsize_range = (1.4, 4.1),
                          log_imsize_step = 0.1,
                          log_nchan_range = (3.0, 6.1),
                          log_nchan_step = 0.1,
                          nspw=10):
    '''
    calculate the fraction of projects for various combinations of
    data rates and mitigation limits

    Date        Programmer      Description of Changes
    --------------------------------------------------------------------
    9/23/2022   A.A. Kepley     Original Code

    '''

     # calculate imsize ranges
    log_imsize = np.arange(log_imsize_range[0],log_imsize_range[1], log_imsize_step)
    log_nchan = np.arange(log_nchan_range[0],log_nchan_range[1], log_nchan_step)

    # calculate cube size
    imsize, nchan = np.meshgrid( 10**(log_imsize), 10**(log_nchan))
    cube_size = calc_cube_size(imsize,nchan)
    
    # calculate numbers
    num_total = np.nansum(np.transpose(h))
    print("Total Number: ", num_total)
    
    if len(mit_limits) != len(max_nchans):
        print("Length of mit_limits array must equal length of max_nchans array")
        return

    print("Mit. limit          Max Nchan         Number Below             Number Total           % Below")
    for (mylimit,mymaxnchan) in zip(mit_limits, max_nchans):    

        #ipdb.set_trace() 
        mysel =  np.where( (cube_size < maxcube * mylimit) & (nchan < mymaxnchan),1,0)
        num_below_limit = np.nansum(np.transpose(h) * mysel)
        frac_below_limit = 100.0*num_below_limit / num_total

        
        mystr = '{0:5.1f}                {1:7.1f}               {2:7.2f}                  {3:7.2f}             {4:5.2f}\n'.format(mylimit,mymaxnchan,num_below_limit,num_total, frac_below_limit)
        print(mystr)
        
        
                          





                                                                                            

