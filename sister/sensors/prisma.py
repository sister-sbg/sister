#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SISTER
Space-based Imaging Spectroscopy and Thermal PathfindER
Author: Adam Chlus
"""

import datetime as dt
import logging
import os
import zipfile
import shutil
import h5py
import hytools as ht
import pandas as pd
from hytools.io.envi import WriteENVI,envi_header_dict
from hytools.topo.topo import calc_cosine_i
import numpy as np
from scipy.interpolate import interp1d
import pyproj
from pysolar import solar
from skimage.util import view_as_blocks
from scipy.ndimage import uniform_filter
from ..utils.terrain import *
from ..utils.geometry import *
from ..utils.ancillary import *


home = os.path.expanduser("~")

def he5_to_envi(l1_zip,out_dir,temp_dir,elev_dir,
                smile = None,match=None,rfl = False, project = True,
                res = 30):
    '''
    This function exports three files:
        *_rad* : Merged and optionally smile corrected radiance cube
        *_obs* : Observables file in the format of JPL obs files:
                1. Pathlength (m)
                2. Sensor view azimuth angle (degrees)
                3. Sensor view zenith angle (degrees)
                4. Solar azimuth angle (degrees)
                5. Solar zenith angle (degrees)
                6. Sensor view azimuth angle in degrees
                7. Sensor view azimuth angle in degrees
        *_loc* : Location file in the following format:
                1. Longitude (decimal degrees)
                2. Longitude (decimal degrees)
                3. Elevation (m)

    l1(str): L1 zipped radiance data product path
    l2(str): L2C zipped reflectance data product path
    out_dir(str): Output directory of ENVI datasets
    temp_dir(str): Temporary directory for intermediate
    elev_dir (str): Directory zipped elevation tiles
    smile (str) : Pathname of smile correction surface file
    match (str or list) : Pathname to Landsat image(s) for image re-registration (recommended)
    project (bool) : Project image to UTM grid
    res (int) : Resolution of projected image, 30 should be one of its factors

    '''

    base_name = os.path.basename(l1_zip)[16:-4]

    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    temp_dir = '%s/PRISMA_%s/'% (temp_dir,base_name)
    if not os.path.isdir(temp_dir):
        os.mkdir(temp_dir)

    zip_base  =os.path.basename(l1_zip)
    logging.info('Unzipping %s' % zip_base)
    with zipfile.ZipFile(l1_zip,'r') as zipped:
        zipped.extractall(temp_dir)

    l1_obj  = h5py.File('%sPRS_L1_STD_OFFL_%s.he5' % (temp_dir,base_name),'r')

    file_suffixes =  ['rad','loc','obs']

    if smile:
        smile_obj = ht.HyTools()
        smile_obj.read_file(smile, 'envi')
        shift_surf_smooth = smile_obj.get_band(0)
        smile_correct = True


    measurement = 'rad'
    logging.info('Exporting radiance data')

    # Export VNIR to temporary ENVI
    vnir_data =  l1_obj['HDFEOS']["SWATHS"]['PRS_L1_HCO']['Data Fields']['VNIR_Cube']
    vnir_waves = l1_obj.attrs.get('List_Cw_Vnir')
    vnir_fwhm = l1_obj.attrs.get('List_Fwhm_Vnir')

    rad_dict = envi_header_dict ()
    rad_dict['lines']= vnir_data.shape[0]
    rad_dict['samples']= vnir_data.shape[2]
    rad_dict['bands']=  vnir_data.shape[1]
    rad_dict['wavelength']= vnir_waves
    rad_dict['fwhm']= vnir_fwhm
    rad_dict['interleave']= 'bsq'
    rad_dict['data type'] = 12
    rad_dict['wavelength units'] = "nanometers"
    rad_dict['byte order'] = 0
    vnir_temp = '%sPRISMA_-%s_%s_vnir' % (temp_dir,base_name,measurement)

    writer = WriteENVI(vnir_temp,rad_dict )
    writer.write_chunk(np.moveaxis(vnir_data[:,:,:],1,2), 0,0)

    # Export SWIR to temporary ENVI
    swir_data =  l1_obj['HDFEOS']["SWATHS"]['PRS_L1_HCO']['Data Fields']['SWIR_Cube']
    swir_waves = l1_obj.attrs.get('List_Cw_Swir')
    swir_fwhm = l1_obj.attrs.get('List_Fwhm_Swir')

    rad_dict = envi_header_dict ()
    rad_dict['lines']= swir_data.shape[0]
    rad_dict['samples']= swir_data.shape[2]
    rad_dict['bands']=  swir_data.shape[1]
    rad_dict['wavelength']= swir_waves
    rad_dict['fwhm']= swir_fwhm
    rad_dict['interleave']= 'bil'
    rad_dict['data type'] = 12
    rad_dict['wavelength units'] = "nanometers"
    rad_dict['byte order'] = 0
    swir_temp = '%sPRISMA_%s_%s_swir' % (temp_dir,base_name,measurement)

    writer = WriteENVI(swir_temp,rad_dict )
    writer.write_chunk(np.moveaxis(swir_data[:,:,:],1,2), 0,0)

    vnir_waves = np.flip(vnir_waves[6:])
    swir_waves = np.flip(swir_waves[:-3])

    vnir_fwhm = np.flip(vnir_fwhm[6:])
    swir_fwhm = np.flip(swir_fwhm[:-3])

    vnir_obj = ht.HyTools()
    vnir_obj.read_file(vnir_temp, 'envi')

    swir_obj = ht.HyTools()
    swir_obj.read_file(swir_temp, 'envi')

    if project:
        output_name = '%sPRISMA_%s_%s_unprj' % (temp_dir,base_name,measurement)
    else:
        output_name = '%sPRISMA_%s_%s_unprj' % (out_dir,base_name,measurement)

    rad_dict  = envi_header_dict()
    rad_dict ['lines']= vnir_obj.lines-4 #Clip edges of array
    rad_dict ['samples']=vnir_obj.columns-4  #Clip edges of array
    rad_dict ['bands']= len(vnir_waves.tolist() + swir_waves.tolist())
    rad_dict ['wavelength']= vnir_waves.tolist() + swir_waves.tolist()
    rad_dict ['fwhm']= vnir_fwhm.tolist() + swir_fwhm.tolist()
    rad_dict ['interleave']= 'bil'
    rad_dict ['data type'] = 4
    rad_dict ['wavelength units'] = "nanometers"
    rad_dict ['byte order'] = 0

    writer = WriteENVI(output_name,rad_dict)
    iterator_v =vnir_obj.iterate(by = 'line')
    iterator_s =swir_obj.iterate(by = 'line')

    while not iterator_v.complete:
        chunk_v = iterator_v.read_next()[:,6:]
        chunk_v =np.flip(chunk_v,axis=1)
        chunk_s = iterator_s.read_next()[:,:-3]
        chunk_s =np.flip(chunk_s,axis=1)

        if (iterator_v.current_line >=2) and (iterator_v.current_line <= 997):
            if (measurement == 'rad') & smile_correct:
                vnir_interpolator = interp1d(shift_surf_smooth[iterator_v.current_line,:60],
                                               chunk_v,fill_value = "extrapolate",kind='cubic')
                chunk_v = vnir_interpolator(vnir_waves)
                swir_interpolator = interp1d(shift_surf_smooth[iterator_v.current_line,60:],
                                               chunk_s,fill_value = "extrapolate",kind='cubic')
                chunk_s = swir_interpolator(swir_waves)

            line = np.concatenate([chunk_v,chunk_s],axis=1)/1000.
            writer.write_line(line[2:-2,:], iterator_v.current_line-2)

    #Load ancillary datasets
    geo =  l1_obj['HDFEOS']["SWATHS"]['PRS_L1_HCO']['Geolocation Fields']
    pvs =  l1_obj['Info']["Ancillary"]['PVSdata']

    # Time
    '''1. Convert from MJD2000 to UTC hours
       2. Fit line to estimate continous time.
    '''

    def dhour(day):
        epoch = dt.datetime(2000,1, 1,)
        epoch = epoch.replace(tzinfo=dt.timezone.utc)

        hour =  (day-day//1)*24
        minute =  (hour-hour//1)*60
        second= (minute-minute//1)*60
        microsecond= (second-second//1)*1000000
        time = epoch + dt.timedelta(days=day//1,hours=hour//1,
                                    minutes=minute//1,seconds=second,
                                    microseconds =microsecond)
        return time.hour + time.minute/60. + time.second/3600.

    v_dhour = np.vectorize(dhour)
    utc_time = v_dhour(np.array(geo['Time'][:]))
    utc_time = np.ones(geo['Longitude_VNIR'][:,:].shape[0]) *utc_time[:,np.newaxis]
    utc_time = utc_time[2:-2,2:-2]

    # Solar geometries
    '''Solar geometry is calculated based on the mean scene acquisition time
    which varies by less than 5 seconds from start to end of the scene and is
    computationally more efficient.
    '''
    mjd2000_epoch = dt.datetime(2000,1, 1,)
    mjd2000_epoch = mjd2000_epoch.replace(tzinfo=dt.timezone.utc)
    mean_time = mjd2000_epoch + dt.timedelta(days=np.array(geo['Time'][:]).mean())

    solar_az = solar.get_azimuth(geo['Latitude_VNIR'][:,:],geo['Longitude_VNIR'][:,:],mean_time)[2:-2,2:-2]
    solar_zn = 90-solar.get_altitude(geo['Latitude_VNIR'][:,:],geo['Longitude_VNIR'][:,:],mean_time)[2:-2,2:-2]

    longitude= geo['Longitude_VNIR'][2:-2,2:-2]
    latitude= geo['Latitude_VNIR'][2:-2,2:-2]

    #Create initial elevation raster
    elevation= dem_generate(longitude,latitude,elev_dir,temp_dir)
    zone,direction = utm_zone(longitude,latitude)

    # Calculate satellite X,Y,Z position for each line
    ''' GPS data are sampled at 1Hz resulting in steps in the
        position data, a line is fit to each dimension to estimate
        continuous position.

        There are more GPS samples than there are lines, to allign
        the GPS signal with the line, we use the provided 'Time'
        information for each line to match with the GPS data.

        When converting GPS time to UTC we use 17 sec difference
        instead of 18 sec because it matches the time provided in
        the time array.
    '''

    # Convert satellite GPS position time to UTC
    sat_t = []
    for second,week in  zip(pvs['GPS_Time_of_Last_Position'][:].flatten(),pvs['Week_Number'][:].flatten()):
        gps_second = (week*7*24*60*60) + second
        gps_epoch = dt.datetime(1980, 1, 6)
        gps_time  = gps_epoch+ dt.timedelta(seconds=gps_second - 17)
        sat_t.append(gps_time.hour*3600 + gps_time.minute*60. + gps_time.second)
    sat_t = np.array(sat_t)[:,np.newaxis]

    # Convert line MJD2000 to UTC
    grd_t = []
    for day in geo['Time'][:].flatten():
        time = mjd2000_epoch + dt.timedelta(days=day)
        grd_t.append(time.hour*3600 + time.minute*60. + time.second)
    grd_t = np.array(grd_t)[:,np.newaxis]

    #Fit a line to ground time
    X = np.concatenate([np.arange(1000)[:,np.newaxis], np.ones(grd_t.shape)],axis=1)
    slope, intercept = np.linalg.lstsq(X,grd_t,rcond=-1)[0].flatten()
    line_t_linear = slope*np.arange(1000)+ intercept

    #Fit a line to satellite time
    measurements = np.arange(len(sat_t))
    X = np.concatenate([measurements[:,np.newaxis], np.ones(sat_t.shape)],axis=1)
    slope, intercept = np.linalg.lstsq(X,sat_t,rcond=-1)[0].flatten()
    sat_t_linear = slope*measurements+ intercept

    # Interpolate x,y,z satelite positions
    sat_xyz = []
    for sat_pos in ['x','y','z']:
        sat_p = np.array(pvs['Wgs84_pos_%s' % sat_pos][:])
        slope, intercept = np.linalg.lstsq(X,sat_p,rcond=-1)[0].flatten()
        sat_p_linear = slope*measurements+ intercept
        interpolator = interp1d(sat_t_linear,sat_p_linear,
                                fill_value="extrapolate",kind = 'linear')
        sat_interp = interpolator(line_t_linear)
        sat_xyz.append(sat_interp[2:-2])
    sat_xyz = np.array(sat_xyz)

    # Calculate sensor to ground pathlength
    grd_xyz = np.array(dda2ecef(longitude,latitude,elevation))
    path = pathlength(sat_xyz,grd_xyz)


    # Export satellite position to csv
    sat_lon,sat_lat,sat_alt = ecef2dda(sat_xyz[0],sat_xyz[1],sat_xyz[2])
    satellite_df = pd.DataFrame()
    satellite_df['lat'] = sat_lat
    satellite_df['lon'] = sat_lon
    satellite_df['alt'] = sat_alt
    satellite_df.to_csv('%sPRISMA_%s_satellite_loc.csv' % (out_dir,base_name))

    # Convert satellite coords to local ENU
    sat_enu  = np.array(dda2utm(sat_lon,sat_lat,sat_alt,
                       utm_zone(longitude,latitude)))
    # Convert ground coords to local ENU
    easting,northing,up  =dda2utm(longitude,latitude,
                                elevation)

    # Calculate sensor
    sensor_zn,sensor_az = sensor_view_angles(sat_enu,
                                             np.array([easting,northing,up]))

    # Perform image matching
    if match:
        easting,northing,up =  dda2utm(longitude,latitude,elevation)
        coords =np.concatenate([np.expand_dims(easting.flatten(),axis=1),
                                np.expand_dims(northing.flatten(),axis=1)],axis=1)

        project = Projector()
        project.create_tree(coords,easting.shape)
        project.query_tree(easting.min()-100,northing.max()+100,30)

        sensor_az_prj = project.project_band(sensor_az,-9999)
        sensor_zn_prj = project.project_band(sensor_zn,-9999)
        elevation_prj = project.project_band(elevation.astype(np.float),-9999)

        rad_file = '%sPRISMA_%s_rad_unprj' % (temp_dir,base_name)
        radiance = ht.HyTools()
        radiance.read_file(rad_file, 'envi')

        #Average over Landsat 8 Band 5 bandwidth
        warp_band = np.zeros(longitude.shape)
        for wave in range(850,890,10):
            warp_band += radiance.get_wave(wave)/7.
        warp_band = project.project_band(warp_band,-9999)
        warp_band = 16000*(warp_band-warp_band.min())/warp_band.max()

        #Calculate optimal shift
        y_model,x_model = image_match(match,warp_band,
                                      easting.min()-100,northing.max()+100,
                                      sensor_zn_prj,sensor_az_prj,elevation_prj)

        #Apply uniform filter
        smooth_elevation = uniform_filter(elevation,25)
        smooth_az = uniform_filter(sensor_az,25)
        smooth_zn = uniform_filter(sensor_zn,25)

        i,a,b,c = y_model
        y_offset = i + a*smooth_zn +b*smooth_az + c*smooth_elevation

        i,a,b,c= x_model
        x_offset = i + a*smooth_zn +b*smooth_az + c*smooth_elevation

        easting = easting+  30*x_offset
        northing = northing- 30*y_offset

        zone,direction = utm_zone(longitude,latitude)
        longitude,latitude = utm2dd(easting,northing,zone,direction)

        #Recalculate elevation with new coordinates
        logging.info('Rebuilding DEM')
        elevation= dem_generate(longitude,latitude,elev_dir,temp_dir)

    # Export location datacube
    if project:
        loc_file = '%sPRISMA_%s_loc_unprj' % (temp_dir,base_name)
    else:
        loc_file = '%sPRISMA_%s_loc_unprj' % (out_dir,base_name)
    loc_export(loc_file,longitude,latitude,elevation)


    # Generate remaining observable layers
    slope,aspect = slope_aspect(elevation,temp_dir)
    cosine_i = calc_cosine_i(np.radians(solar_zn),
                             np.radians(solar_az),
                             np.radians(slope),
                             np.radians(aspect))
    rel_zn = np.radians(solar_az-sensor_zn)
    phase =  np.arccos(np.cos(np.radians(solar_zn)))*np.cos(np.radians(solar_zn))
    phase += np.sin(np.radians(solar_zn))*np.sin(np.radians(solar_zn))*np.cos(rel_zn)

    # Export observables datacube
    if project:
        obs_file = '%sPRISMA_%s_obs_unprj' % (temp_dir,base_name)
    else:
        obs_file = '%sPRISMA_%s_obs_unprj' % (out_dir,base_name)

    obs_export(obs_file,path,sensor_az,sensor_zn,
               solar_az,solar_zn,phase,slope,aspect,
               cosine_i,utc_time)

    if project:
        #Create new projector with corrected coordinates
        new_coords =np.concatenate([np.expand_dims(easting.flatten(),axis=1),
                        np.expand_dims(northing.flatten(),axis=1)],axis=1)

        project = Projector()
        project.create_tree(new_coords,easting.shape)
        project.query_tree(easting.min()-100,northing.max()+100,30)

        blocksize = int(res/30)
        map_info = ['UTM', 1, 1, easting.min()-100, northing.max()+100,res,
                           res,zone,direction, 'WGS-84' , 'units=Meters']
        out_cols = int(blocksize* (project.output_shape[1]//blocksize))
        out_lines = int(blocksize* (project.output_shape[0]//blocksize))

        logging.info('Georeferencing datasets to %sm resolution' % res)
        for file in file_suffixes:
            input_name = '%sPRISMA_%s_%s_unprj' % (temp_dir,base_name,file)
            hy_obj = ht.HyTools()
            hy_obj.read_file(input_name, 'envi')
            iterator =hy_obj.iterate(by = 'band')

            out_header = hy_obj.get_header()
            out_header['lines']= project.output_shape[0]//blocksize
            out_header['samples']=project.output_shape[1]//blocksize
            out_header['data ignore value'] = -9999
            out_header['map info'] = map_info

            output_name = '%sPRISMA_%s_%s_geo' % (out_dir,base_name,file)
            writer = WriteENVI(output_name,out_header)

            while not iterator.complete:
                band = project.project_band(iterator.read_next(),-9999)
                band[band == -9999] = np.nan
                band = np.nanmean(view_as_blocks(band[:out_lines,:out_cols], (blocksize,blocksize)),axis=(2,3))
                if file == 'rad':
                    band[band<0] = 0
                band[np.isnan(band)] = -9999
                writer.write_band(band,iterator.current_band)


    shutil.rmtree(temp_dir)
