import pandas as pd
import numpy as np
import os
import h5py
from pymsis import msis, utils
import xarray as xr
from datetime import datetime, timedelta
import glob
from tqdm import tqdm
import sys
import scipy.io
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from mpl_toolkits.mplot3d import Axes3D
import re
from bisect import bisect_right

def madrigal_fpi_loader(dir):

    '''Reads local data from the Madrigal FPI HDF5 files and returns a DataFrame with the data

    Args:
        dir: The directory containing the Madrigal FPI data
    
    Returns:
        zonal_winds: DataFrame with zonal wind data
        meridional_winds: DataFrame with meridional wind data
        temps: DataFrame with temperature data
    '''

    # List the subdirectories inside the main directory
    subdirectories = [os.path.join(dir, subdir) for subdir in os.listdir(dir) if os.path.isdir(os.path.join(dir, subdir))]

    zonal_winds = pd.DataFrame()
    meridional_winds = pd.DataFrame()
    temps = pd.DataFrame()

    # Now iterate through each directory (main and subdirectories)
    for subdir in subdirectories:
        
        sitename = os.path.basename(subdir)

        # Initialize an empty location_winds to store the concatenated data
        location_winds = pd.DataFrame() 
        print(subdir)
        files = os.listdir(subdir)
        hdf5_files = [file for file in files if file.endswith('hdf5')]
        print(f'{len(hdf5_files)} files found.')

        # Iterate over each CSV file and append it to the end of the concatenated location_winds
        for file in hdf5_files:
            filepath = os.path.join(subdir, file)
            hdf5_file = h5py.File(filepath, 'r')
            dataset = hdf5_file['Data/Table Layout'][:]
            df_post_processed = pd.DataFrame(dataset)
            hdf5_file.close()

            df_post_processed['datetime'] = pd.to_datetime(df_post_processed['ut1_unix'], unit='s')

            try:
                df_post_processed = df_post_processed[['datetime', 'year', 'month', 'day', 'hour','slt', 'mlt', 'altb', 'alte', 'gdalt',
                                                    'azm', 'elm', 'gdlat', 'glon', 'szen', 'kp', 'dst', 'ap3', 'ap',
                                                    'f10.7', 'fbar', 'tn', 'dtn', 'vn1', 'dvn1', 'vn2', 'dvn2']]


            except KeyError:
                df_post_processed = df_post_processed[['datetime', 'year', 'month', 'day', 'hour','slt', 'mlt', 'altb', 'alte', 'gdalt',
                                                    'azm', 'elm', 'gdlat', 'glon', 'szen', 'kp', 'dst', 'ap3', 'ap',
                                                    'f10.7', 'fbar', 'tn', 'dtn', 'vn1', 'dvn1', 'vn2', 'dvn2']]

            df_post_processed = df_post_processed.rename(columns={
                'slt': 'solar_local_time',
                'mlt': 'magnetic_local_time',
                'gdalt': 'geodetic_altitude',
                'altb': 'min_altitude', 
                'alte': 'max_altitude', 
                'gdlat': 'geodetic_latitude', 
                'glon': 'geodetic_longitude',
                'azm': 'azimuth_angle',
                'elm': 'elevation_angle', 
                'szen': 'solar_zenith_angle',
                'tn': 'temperature',
                'dtn': 'temperature_error',
                'vn1': 'zonal_wind_speed', 
                'dvn1': 'zws_error', 
                'vn2': 'meridional_wind_speed', 
                'dvn2': 'mws_error',
                'ap3': 'ap_3_hour', 
                'ap': 'ap_daily',
                'fbar': 'multiday_f10.7', 
                'bxgsm': 'bz_imf', 
                'bygsm': 'by_imf', 
                'bzgsm': 'bx_imf', 
                'bimf': 'b_imf', 
                'swden': 'solar_wind_plasma_density', 
                'swspd': 'solar_wind_speed',
                'ne_iri': 'iri_e_density'
            })

            df_post_processed['site_name'] = sitename

            # Round to the nearest whole number and replace 24 with 0
            df_post_processed['hour_local'] = df_post_processed['solar_local_time'].apply(lambda x: 0 if round(x) == 24 else round(x))
        
            #concatenate new df to master location_temps
            location_winds = pd.concat([location_winds, df_post_processed])

        location_winds = location_winds.sort_values(by='datetime')

        location_temps = location_winds[['datetime','year', 'month', 'day', 'hour', 'solar_local_time', 'hour_local',
                                        'magnetic_local_time', 'site_name', 'min_altitude', 'max_altitude',
                                        'geodetic_altitude', 'elevation_angle',
                                        'geodetic_latitude', 'geodetic_longitude', 'solar_zenith_angle', 'kp',
                                        'dst', 'f10.7', 'multiday_f10.7', 'ap_3_hour','ap_daily', 'temperature', 'temperature_error']]     
        location_temps = location_temps.dropna(subset=['temperature', 'temperature_error'], how='all')

        location_zw = location_winds[['datetime','year', 'month', 'day', 'hour', 'solar_local_time', 'hour_local',
                                        'magnetic_local_time', 'site_name', 'min_altitude', 'max_altitude',
                                        'geodetic_altitude', 'elevation_angle',
                                        'geodetic_latitude', 'geodetic_longitude', 'solar_zenith_angle', 'kp',
                                        'dst', 'f10.7', 'multiday_f10.7', 'ap_3_hour','ap_daily', 'zonal_wind_speed','zws_error']]     
        location_zw = location_zw.dropna(subset=['zonal_wind_speed','zws_error'], how='all')

        location_mw = location_winds[['datetime','year', 'month', 'day', 'hour', 'solar_local_time', 'hour_local',
                                        'magnetic_local_time', 'site_name', 'min_altitude', 'max_altitude',
                                        'geodetic_altitude', 'elevation_angle',
                                        'geodetic_latitude', 'geodetic_longitude', 'solar_zenith_angle', 'kp',
                                        'dst', 'f10.7', 'multiday_f10.7', 'ap_3_hour','ap_daily', 'meridional_wind_speed','mws_error']]     
        location_mw = location_mw.dropna(subset=['meridional_wind_speed','mws_error'], how='all')

        zonal_winds = pd.concat([zonal_winds, location_zw])
        meridional_winds = pd.concat([meridional_winds, location_mw])
        temps = pd.concat([temps, location_temps])

    zonal_winds = zonal_winds.drop_duplicates()
    meridional_winds = meridional_winds.drop_duplicates()
    temps = temps.drop_duplicates()

    return zonal_winds, meridional_winds, temps

    """
    Matches each target datetime to MSIS-predicted values, and constructs
    f107, f107a, and apnp arrays formatted for NRLMSIS.

    Args:
        target_df (pd.DataFrame): DataFrame with a 'datetime' column.
        predicted_df (pd.DataFrame): DataFrame with 'datetime', 'f10_index',
                                     'f10_54day_avg', and 'ap_index'.

    Returns:
        f107 (np.ndarray): f10_index values matched to target times.
        f107a (np.ndarray): f10_54day_avg values matched to target times.
        apnp (np.ndarray): Array of shape (N, 7) with formatted ap values for NRLMSIS.
    """

    # Ensure both are sorted
    predicted_df = prep_msis_predicted_data(forecast_dir)
    target_df = target_df.sort_values('datetime').reset_index(drop=True)
    predicted_df = predicted_df.sort_values('datetime').reset_index(drop=True)

    # Forecast window range
    forecast_start = predicted_df['datetime'].to_numpy()
    forecast_end = forecast_start + np.timedelta64(3, 'h')

# Build lookup table for ap_index (using floored 3-hour bins)
    ap_series = predicted_df[['datetime', 'ap_index']].copy()
    ap_series['datetime'] = ap_series['datetime'].dt.floor('3h')
    ap_series = ap_series.drop_duplicates('datetime').sort_values('datetime')
    ap_times = ap_series['datetime'].to_list()
    ap_values = ap_series['ap_index'].to_list()

    # For fast lookups
    def get_closest_ap(time):
        i = bisect_right(ap_times, time)
        return ap_values[i-1] if i > 0 else np.nan

    f107 = []
    f107a = []
    apnp = []

    for time in target_df['datetime']:
        # f107/f107a from 3h window
        i = ((forecast_start <= time) & (time < forecast_end)).nonzero()[0]
        if len(i) > 0:
            row = predicted_df.iloc[i[0]]
            f107.append(row['f10_index'])
            f107a.append(row['f10_54day_avg'])
        else:
            f107.append(np.nan)
            f107a.append(np.nan)

        # --- apnp[7] construction ---
        ap_block = []

        # 0–1: daily Ap and current 3h ap (same for now)
        t0 = time.floor('3h')
        ap_block.append(get_closest_ap(t0))  # daily proxy
        ap_block.append(get_closest_ap(t0))  # current

        # 2–4: 3, 6, 9 hours before
        for offset in [3, 6, 9]:
            t = (time - pd.Timedelta(hours=offset)).floor('3h')
            ap_block.append(get_closest_ap(t))

        # 5: avg of 8 values from 12–33 hours before
        range1 = [(time - pd.Timedelta(hours=h)).floor('3h') for h in range(12, 34, 3)]
        vals1 = [get_closest_ap(t) for t in range1]
        vals1 = [v for v in vals1 if not np.isnan(v)]
        ap_block.append(np.mean(vals1) if vals1 else np.nan)

        # 6: avg of 8 values from 36–57 hours before
        range2 = [(time - pd.Timedelta(hours=h)).floor('3h') for h in range(36, 58, 3)]
        vals2 = [get_closest_ap(t) for t in range2]
        vals2 = [v for v in vals2 if not np.isnan(v)]
        ap_block.append(np.mean(vals2) if vals2 else np.nan)

        apnp.append(ap_block)

    return (
        np.array(f107),
        np.array(f107a),
        np.array(apnp)
    )

def run_msis_specific_location(df, version, date_range=None):

    '''Runs the NRLMSIS model on a specific location and returns the data

    Inputs:
        df: DataFrame with the datetime, geodetic latitude, longitude, and altitude

    Returns:
        data: DataFrame with the MSIS data
    '''
    
    start_date = pd.to_datetime(df.datetime.min())
    end_date = pd.to_datetime(df.datetime.max())

    if date_range == 'hourly':
        date_range = pd.DataFrame(
            {'datetime': pd.date_range(start=start_date, end=end_date, freq='h')}
        )

        f107, f107a, aps = utils.get_f107_ap(date_range['datetime'])
        data = msis.run(date_range['datetime'], df['geodetic_latitude'].values[0], df['geodetic_longitude'].values[0], df['geodetic_altitude'].values[0], aps=aps, f107s=f107, f107as=f107a, geomagnetic_activity=-1, version=version)
    
    else:
        f107, f107a, aps = utils.get_f107_ap(df['datetime'])
        data = msis.run(df['datetime'], df['geodetic_latitude'].values[0], df['geodetic_longitude'].values[0], df['geodetic_altitude'].values[0], aps=aps, f107s=f107, f107as=f107a, geomagnetic_activity=-1, version=version)
    
    data = data.squeeze()
    data = pd.DataFrame(data)
    
    data = data.rename(columns={
    0: 'msis_mass_density',
    1: 'n2_density',
    2: 'o2_density',
    3: 'o_density',
    4: 'he_density',
    5: 'h_density',
    6: 'ar_density',
    7: 'n_density',
    8: 'anomalous_o_density',
    9: 'no_density',   
    10: 'msis_temperature'})

    #ata['datetime'] = data['datetime'].to_list()

    return data

def load_imf_data_nc(dir):

    '''Loads the IMF data from the NetCDF files and returns a DataFrame
    
    Args:
        dir: Directory containing the IMF data

    Returns:
        df: DataFrame with the IMF data
    '''

    # Get a list of all NetCDF files in the directory
    file_list = [os.path.join(dir, f) for f in os.listdir(dir) if f.endswith('.nc')]

    # Open and combine the datasets
    datasets = [xr.open_dataset(file) for file in file_list]

    # Assuming the files share the same variables and dimensions
    combined_dataset = xr.concat(datasets, dim="time")

    # Remember to close the individual datasets to free resources
    for ds in datasets:
        ds.close()

    df = combined_dataset.to_dataframe().reset_index()
    df = df[['time', 'bt','bx_gse', 'by_gse', 'bz_gse', 'theta_gse', 'phi_gse', 'bx_gsm',
       'by_gsm', 'bz_gsm', 'theta_gsm', 'phi_gsm']]
    
    df = df.rename(columns={'time': 'datetime'})
    df = df.sort_values(by='datetime').dropna()
    
    return df

def load_omni_imf_data(dir):

    '''Loads the IMF data from the NetCDF files and returns a DataFrame
    
    Args:
        dir: Directory containing the IMF data

    Returns:
        df: DataFrame with the IMF data
    '''
    # List and sort all .lst files
    lst_files = sorted([f for f in os.listdir(dir) if f.endswith(".lst")])

    # Load and combine them
    df_list = []
    for file in lst_files:
        file_path = os.path.join(dir, file)
        
        # Read ASCII file: space-separated, no header, skip comment lines (if any start with # or :)
        df = pd.read_csv(
            file_path,
            sep='\s+',
            header=None,
            comment='#',     # or use comment=':' depending on the file format
            engine='python'  # more tolerant of spacing
        )
        
        df_list.append(df)

    # Combine into one large DataFrame
    combined_df = pd.concat(df_list, ignore_index=True)
    combined_df.columns = [
    "year", "day", "hour", "minute", "b_avg", "bx_gse_gsm", "by_gse", "bz_gse", "by_gsm", "bz_gsm",
    "b_scalar_rms_sd", "b_vector_rms_sd", "speed_kms", "vx_kms", "vy_kms", "vz_kms",
    "proton_density", "proton_temperature", "ae_index", "al_index", "au_index"]

    # Create datetime using day-of-year properly
    combined_df["datetime"] = pd.to_datetime(
        combined_df["year"] * 1000 + combined_df["day"], format="%Y%j"
    ) + pd.to_timedelta(
        combined_df["hour"], unit="h"
    ) + pd.to_timedelta(
        combined_df["minute"], unit="m"
    )

    combined_df = combined_df.sort_values(by='datetime').dropna()
    combined_df = combined_df[["datetime", "day", "hour", "minute", "b_avg", "bx_gse_gsm", "by_gse", "bz_gse", "by_gsm", "bz_gsm",
    "b_scalar_rms_sd", "b_vector_rms_sd", "speed_kms", "vx_kms", "vy_kms", "vz_kms",
    "proton_density", "proton_temperature", "ae_index", "al_index", "au_index"]]

    return combined_df

def load_madrigal_imf(madrigal_dir, imf_dir):

    '''Loads the full dataset by merging the Madrigal FPI data with the IMF data
    
    Args:
        madrigal_dir: Directory containing the Madrigal FPI data
        imf_dir: Directory containing the IMF data

    Returns:
        zw_df: DataFrame with zonal wind data
        mw_df: DataFrame with meridional wind data
        temps_df: DataFrame with temperature data
    '''

    zonal_winds, meridional_winds, temps = madrigal_fpi_loader(madrigal_dir)
    imf_df = load_imf_data(imf_dir)
    
    # Sort both DataFrames by the datetime column
    zonal_winds = zonal_winds.sort_values('datetime')
    meridional_winds = meridional_winds.sort_values('datetime')
    temps = temps.sort_values('datetime')
    imf_data = imf_df.sort_values('datetime')

    print('Lengths before merge:')
    print(f'ZW: {len(zonal_winds)}, MW: {len(meridional_winds)}, TEMPS: {len(temps)}, IMF: {len(imf_data)}')

    # Perform the asof merge with optional tolerance
    zw_df = pd.merge_asof(
        zonal_winds,
        imf_df,
        on='datetime',
        direction='nearest',  # Closest match in both directions
        tolerance=pd.Timedelta(minutes=5)  # Optional: limit the time difference
    ).dropna()
    mw_df = pd.merge_asof(
        meridional_winds,
        imf_df,
        on='datetime',
        direction='nearest',  # Closest match in both directions
        tolerance=pd.Timedelta(minutes=5)  # Optional: limit the time difference
    ).dropna()
    temps_df = pd.merge_asof(
        temps,
        imf_df,
        on='datetime',
        direction='nearest',  # Closest match in both directions
        tolerance=pd.Timedelta(minutes=5)  # Optional: limit the time difference
    ).dropna()

    print('Lengths after merge:')
    print(f'ZW: {len(zw_df)}, MW: {len(mw_df)}, TEMPS: {len(temps_df)}')

    return zw_df, mw_df, temps_df

def eiscat_hdf5_loader(dir):

    '''Loads the EISCAT data from the HDF5 files and returns a DataFrame
    
    Args:
        dir: Directory containing the EISCAT data

    Returns:
        final_combined_df: DataFrame with the EISCAT data
    '''

    # Function to convert days since 0000-01-01 to datetime
    def days_to_datetime(days):
        julian_origin = datetime(1, 1, 1)  # Julian origin
        return julian_origin + timedelta(days=days)

    # Get a list of all HDF5 files in the directory
    hdf5_files = glob.glob(os.path.join(dir, "*.h5"))

    # Initialize an empty list to store DataFrames from all files
    all_dataframes = []
    n=0

    # Process each HDF5 file
    for file in hdf5_files:
        n+=1
        print(f"Processing file {n} of {len(hdf5_files)}")
        with h5py.File(file, "r") as hdf:
            # Collect all dataset names
            dfs = [name for name, obj in hdf.items() if isinstance(obj, h5py.Dataset)]
            dataframes = {name: pd.DataFrame(hdf[name]) for name in dfs}
            
            # Convert datasets for time columns to datetime and create a new DataFrame with all columns
            combined_rows = []
            for i in range(dataframes['alt_pp'].shape[0]):  # Iterate over rows
                for j in range(dataframes['alt_pp'].shape[1]):  # Iterate over columns
                    if not any(np.isnan(dataframes[key].iloc[i, j]) for key in ['alt_pp', 'pp', 'pp_err']):
                        time1_days, time2_days = dataframes['time1_pp'].iloc[j, 0], dataframes['time2_pp'].iloc[j, 0]
                        time1, time2 = days_to_datetime(time1_days), days_to_datetime(time2_days)
                        combined_rows.append([dataframes[key].iloc[i, j] for key in ['alt_pp', 'pp', 'pp_err']] + [time1, time2])

            # Create a DataFrame for the current file
            new_df = pd.DataFrame(combined_rows, columns=['alt_pp', 'pp', 'pp_err', 'time1_pp', 'time2_pp'])
            final_df = new_df[['time1_pp', 'time2_pp', 'alt_pp', 'pp', 'pp_err']].sort_values(by=['time1_pp']).reset_index(drop=True)
            final_df.rename(columns={'time1_pp': 'datetime_start', 'time2_pp': 'datetime_end', 'alt_pp': 'altitude', 'pp': 'electron_density', 'pp_err': 'electron_density_error'}, inplace=True)

            # Append the current DataFrame to the list
            all_dataframes.append(final_df)

    # Concatenate all the DataFrames into one
    final_combined_df = pd.concat(all_dataframes, ignore_index=True)

    # Sort the concatenated DataFrame by 'datetime_start'
    final_combined_df = final_combined_df.sort_values(by='datetime_start').reset_index(drop=True)
    final_combined_df['altitude'] = final_combined_df['altitude'].astype(float)
    final_combined_df['electron_density'] = final_combined_df['electron_density'].astype(float)
    final_combined_df['electron_density_error'] = final_combined_df['electron_density_error'].astype(float)
    final_combined_df['altitude'] = final_combined_df['altitude'].apply(lambda x: x/1000)  # Convert altitude to km

    return final_combined_df

def process_windii_files(base_directory, type):
    
    """Processes all .txt files within subdirectories of the base directory.

    Args:
        base_directory (str): The directory containing the .txt files.
        type (str): The type of data to process. Either 'temps' or 'winds'.

    Returns:
        pd.DataFrame: A DataFrame containing the processed data.
    """

    def parse_windii_data(file_path):
        """Extracts windii data from a given .txt file."""
        data = []
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        i = 0
        while i < len(lines):
            try:
                # Read the first two lines for metadata
                header_values = list(map(float, lines[i].strip().split())) + list(map(int, lines[i + 1].strip().split()))
                altitude_count = header_values[-1]  # Extract ALTITUDE_NUMBER
                i += 2  # Move past header lines
                
                # Read the next 'altitude_count' lines for wind data
                for _ in range(altitude_count):
                    if i >= len(lines):
                        break
                    wind_values = list(map(float, lines[i].strip().split()))
                    full_row = header_values + wind_values  # Combine metadata with wind data
                    data.append(full_row)
                    i += 1

            except (ValueError, IndexError) as e:
                print(f"Skipping line {i} in {file_path} due to parsing error: {e}")
                i += 1  # Skip problematic line and continue

        return data

    all_data = []
    file_paths = glob.glob(os.path.join(base_directory, "**", "*.dat"), recursive=True)
    
    print(f"Found {len(file_paths)} .txt files. Processing...")
    
    for file_path in file_paths:
        print(f"Processing: {file_path}")
        file_data = parse_windii_data(file_path)
        all_data.extend(file_data)  # Append data from this file to the master list

        # Define column names
    columns_winds = ["doy", "ut", "glat", "glon", "local_time", "sza", "profile_number", "altitude_number", 
            "alt", "zonal_wind_speed", "meridional_wind_speed"]

    columns_temps = ["doy", "ut", "glat", "glon", "local_time", "sza", "profile_number", "altitude_number", 
           "alt", "volume_emission_rate", "temperature"]
    
    # Create DataFrame

    if type == 'temperatures':
        df = pd.DataFrame(all_data, columns=columns_temps)

    elif type == 'winds':
        df = pd.DataFrame(all_data, columns=columns_winds)

    def parse_datetime(row):
        doy_str = str(int(row["doy"]))  # Ensure DOY is a string
        year_prefix = int(doy_str[:2])  # Extract first two characters as year
        doy = int(doy_str[2:])  # Extract last three characters as day-of-year
        year = 1900 + year_prefix

        # Convert DOY to a date
        date = datetime(year, 1, 1) + timedelta(days=doy - 1)
        # Convert UT to timedelta
        time = timedelta(hours=float(row["ut"]))

        return date + time

    # Enable progress_apply
    tqdm.pandas()
    
    # Apply the function to each row
    df["datetime"] = df.progress_apply(parse_datetime, axis=1)
    df = df.sort_values("datetime").reset_index(drop=True)

    columns_winds = ['profile_number', 'altitude_number', 'datetime', 'glat', 'glon', 'local_time', 'sza',  'alt', 'zonal_wind_speed', 'meridional_wind_speed']
    columns_temps = ['profile_number', 'altitude_number', 'datetime', 'glat', 'glon', 'local_time', 'sza',  'alt', "volume_emission_rate", "temperature"]

    if type == 'temperatures':
        df = df[columns_temps]

    elif type == 'winds':
        df = df[columns_winds]  

    return df

def indra_fpi_loader(directory):

    '''This function loads in all of the .dat files and concatenate them simultaneously, performing some data trabformations along the way for ease of use later on

    Inputs:
        directory: the directory of fpi .dat files
        site: site of the FPI (ie Kiruna)
        
    Returns:
        dataframe: a concatenated pandas dataframe of all of the .dat files containing the fpi data
    '''
    # Find all files that end with 'pkCol.dat' in subdirectories
    files = glob.glob(f"{directory}/**/*pkCol.dat", recursive=True)

    site_dict = {
    'g': 'kiruna_red',
    'd': 'kiruna_green',
    'f': 'sodankyla_red',
    's': 'svalbard_red',
    'e': 'svalbard_green',
    'W': 'scandi_red',
    'Y': 'scandi_green',
    'h': 'not_sure'}

    def adjust_date(row):
        '''This function adds one day to the date if above 24, as the date from each .dat file is over two days but only written down as one

        Inputs:
            row: the row of the dataframe for the date to be adjusted

        Returns:
            row: the row with the adjusted datetime
        '''
        new_time = float(row['time'])

        #have to set as a float becuase when used this 
        if float(row['time']) > 24:
            row['date'] += timedelta(days=1)
            new_time = float(row['time']) - 24 

        # Convert decimal hours to seconds
        seconds = round(new_time * 3600)

        # Create a timedelta object
        time_delta = timedelta(seconds=seconds)
        
        # Use a starting date to add the timedelta and extract the time
        start_date = datetime(1900, 1, 1)
        result_time = (start_date + time_delta).time()
        row['time'] = str(result_time)

        # need to create a unified datetime object with the date and time objects 
        time_object = datetime.strptime(str(result_time), "%H:%M:%S")
        datetime_combined = datetime(row['date'].year, row['date'].month, row['date'].day,
                                time_object.hour, time_object.minute, time_object.second)
        
        # assign the datetime value to the entry in the date column
        row['date'] = datetime_combined 
        return row

    # Initialize an empty DataFrame to store the concatenated data
    dataframe = pd.DataFrame()
    n=0
    # Iterate over each CSV file and append it to the end of the concatenated DataFrame
    for file in files:
        n+=1
        print(f"Processing file {n} of {len(files)}")
        site_code, date_code = file[-16], file[-15:-10]
        year, day = f"20{date_code[:2]}", date_code[3:5]
        month = int(date_code[2], 16)

        # Create and convert date string
        date_object = datetime.strptime(f"{year}-{month}-{day}", "%Y-%m-%d")

        # Read lines from the file starting from line 14 as the data above isn't tabular it is in written form so unusable if unspecified (check a FP .dat file to see what I mean)
        with open(file, 'r') as file:
            lines = file.readlines()[13:]

        # Create a DataFrame from the lines and add the date column
        df = pd.DataFrame([line.strip().split() for line in lines])
        df['date'] = date_object
        df['site'] = site_dict[site_code]
        df.rename(columns={0: "time", 1: "mirror", 2: "intensity", 3: "intensity_error", 4:"wind_speed", 5:"wind_speed_error", 6:"temperature", 7:"temperature_error", 8:"chi_squared", 9:"snr", 10:"peak"}, inplace=True)

        # Apply the function based on the condition of if more than 24 , change the date to the net day
        df = df.apply(adjust_date, axis=1)
        df.rename(columns={'date': 'datetime'}, inplace=True)
        
        #concatenate new df to master dataframe
        dataframe = pd.concat([dataframe, df])

    dataframe['datetime'] = pd.to_datetime(dataframe['datetime'])
    dataframe['intensity'] = dataframe['intensity'].astype(float)
    dataframe['intensity_error'] = dataframe['intensity_error'].astype(float)
    dataframe['wind_speed'] = dataframe['wind_speed'].astype(float)
    dataframe['wind_speed_error'] = dataframe['wind_speed_error'].astype(float)
    dataframe['temperature'] = dataframe['temperature'].astype(float)
    dataframe['temperature_error'] = dataframe['temperature_error'].astype(float)
    dataframe['chi_squared'] = dataframe['chi_squared'].astype(float)
    dataframe['snr'] = dataframe['snr'].astype(float)

    # mapping the mirror codes to their look directions
    mirror_map = {'1': 'N','2': 'E','3': 'S','4': 'W','5': 'NW','6': 'NE','7': 'Zen','8': 'Cal','9': 'SW','10': 'SE','11': 'Kir A','12': 'Kir B','14': 'Sod A','15': 'Sod B', '16':'EISCAT C'}

    dataframe['look_direction'] = dataframe['mirror'].map(mirror_map)

    # need a function for mapping azimuth angles, have only assigned angles to the 8 compass directions, the rest are assigned None values
    def angle_mapping(row):
        site = row['site']
        if 'sodankyla' in site.lower():
            mapping = {
                '1': 8, '2': 98, '3': 188, '4': 278, '5': 323, '6': 53,'7': None,'8': None,'9': 233,'10': 143,'11': None,'12': None,'14': 335, '15': 220, '16': 296
            }
            return mapping[row['mirror']]
        elif 'kiruna' in site.lower():
            # kiruna changed direction in 2009, need to account for this changed mirror map
            if row['datetime'].date() < datetime.strptime('2009-01-01', "%Y-%m-%d").date():
                mapping = {
                '1': 348,'2': 78,'3': 168,'4': 258,'5': 303,'6': 33,'7': None,'8': None,'9': 213,'10': 123,'11': 43,'12': 158,'14': None,'15': None, '16': None
            }
                return mapping[row['mirror']]
            else:
                mapping = {
                '1': 0, '2': 90, '3': 180, '4': 270, '5': 315, '6': 45, '7': None, '8': None, '9': 225, '10': 135, '11': 44, '12': 157, '14': None, '15': None, '16': 330
            }
                return mapping[row['mirror']]
        else: 
            return None
        
    dataframe['azimuth_angle'] = dataframe.apply(angle_mapping, axis=1)
    dataframe['wind_direction'] = dataframe['azimuth_angle']

    dataframe['wind_speed'] = dataframe['wind_speed'].astype(float)
    dataframe['azimuth_angle'] = dataframe['azimuth_angle'].astype(float)

    # Identify the rows where column 'A' values are less than 0
    mask = dataframe['wind_speed'] < 0

    # Multiply the negative values in column 'A' by -1
    dataframe.loc[mask, 'wind_speed'] = dataframe.loc[mask, 'wind_speed'] * -1

    # Adjust the values in column 'C' by adding 180 and then taking modulo 360
    dataframe.loc[mask, 'wind_direction'] = (dataframe.loc[mask, 'wind_direction'] + 180) % 360

    # setting the correct datatypes and dropping 'peak' column as it is irrelevant
    dataframe = dataframe.drop(['peak', 'time'], axis=1)

    dataframe['look_direction'] = dataframe['look_direction'].astype(str)
    dataframe['mirror'] = dataframe['mirror'].astype(int)
    dataframe['azimuth_angle'] = dataframe['azimuth_angle'].astype(float)

    # sort the dataframe by date descedning and reset the idnex for a clean dataset
    dataframe = dataframe.sort_values(by='datetime')
    dataframe = dataframe.reset_index(drop=True)

    '''!!!!!!!!!!!!!!
    REMOVE THE BELOW LINE WHEN WE HAVE FULL DATASET
    !!!!!!!!!!!!!!!!'''

    # Earth's radius in km
    R = 6378
    locations = {
        'kiruna_red': [67.87, 21.03, 45],  # Latitude, Longitude, Angle
        'kiruna_green': [67.87, 21.03, 45], # Latitude, Longitude, Angle
        'sodankyla_red': [67.37, 26.63, 45], # Latitude, Longitude, Angle
        'not_sure': (np.nan, np.nan, np.nan)
    }

    sites = ['kiruna_red', 'kiruna_green', 'sodankyla_red']
    dataframe = dataframe[dataframe['site'].isin(sites)].copy()

    print('Sites being used:', dataframe['site'].unique())

    dataframe['fpi_latitude'] = dataframe['site'].map(lambda x: locations[x][0])
    dataframe['fpi_longitude'] = dataframe['site'].map(lambda x: locations[x][1])
    dataframe['elevation_angle'] = dataframe['site'].map(lambda x: locations[x][2])

    dataframe['altitude'] = np.where(
    dataframe['site'].str.endswith('green'), 120,
    np.where(dataframe['site'].str.endswith('red'), 240, None)  # Default value if neither matches
    ).astype(float)

    # Handle cases where azimuth_angle is NaN
    missing_azimuth = dataframe['azimuth_angle'].isna()

    # Convert degrees to radians for calculation
    dataframe['azimuth_rad'] = np.radians(dataframe['azimuth_angle'])
    dataframe['elevation_rad'] = np.radians(dataframe['elevation_angle'])
    dataframe['latitude_rad'] = np.radians(dataframe['fpi_latitude'])
    dataframe['longitude_rad'] = np.radians(dataframe['fpi_longitude'])

    # Compute slant distance (distance to phenomenon)
    dataframe['observation_distance'] = (dataframe['altitude']) / np.sin(dataframe['elevation_rad'])

    # Compute latitude of phenomenon
    dataframe['observation_latitude'] = np.where(
        missing_azimuth, 
        dataframe['fpi_latitude'],
        np.degrees(dataframe['latitude_rad'] + 
                (dataframe['observation_distance'] * np.cos(dataframe['azimuth_rad']) / R))
    )

    # Compute longitude of phenomenon
    dataframe['observation_longitude'] = np.where(
        missing_azimuth, 
        dataframe['fpi_longitude'],
        np.degrees(dataframe['longitude_rad'] + 
                (dataframe['observation_distance'] * np.sin(dataframe['azimuth_rad']) / 
                (R * np.cos(dataframe['latitude_rad']))))
    )

    # Drop intermediate calculations if not needed
    dataframe.drop(columns=['azimuth_rad', 'elevation_rad', 'latitude_rad', 'longitude_rad'], inplace=True)

    # reordering the columns
    dataframe = dataframe[['datetime', 'site', 'mirror', 'look_direction', 'azimuth_angle', 'altitude', 'fpi_latitude', 'fpi_longitude', 'observation_latitude', 'observation_longitude', 'intensity', 'intensity_error', 'wind_speed', 
    'wind_speed_error', 'wind_direction', 'temperature', 'temperature_error', 'chi_squared', 'snr']]

    dataframe = dataframe[(dataframe['look_direction'] != 'Cal') & (dataframe['look_direction'] != 'Zen')]

    return dataframe

def get_variable(eiscat_vars, possible_keys):
    """
    Searches for a variable in the loaded MATLAB dictionary using a list of possible key names.
    
    Parameters:
      eiscat_vars: dict, the dictionary loaded from the MATLAB .mat file.
      possible_keys: list of str, possible key names (e.g., ['alt', 'Alt']).
    
    Returns:
      The variable corresponding to the first matching key.
      
    Raises:
      KeyError if none of the keys are found.
    """
    for key in possible_keys:
        if key in eiscat_vars:
            return eiscat_vars[key]
    raise KeyError(f"None of the keys {possible_keys} were found in the file.")

def matlab_to_datetime(matlab_datenum):
    """
    Converts MATLAB serial date numbers to Python datetime objects.
    
    MATLAB's datenum counts days from 1-Jan-0000 (adjusted by 366 days for Python),
    so this function converts either a scalar or an array of MATLAB date numbers.
    
    Parameters:
      matlab_datenum: A scalar or NumPy array of MATLAB serial date numbers.
      
    Returns:
      A Python datetime object or an array of datetime objects.
    """
    if np.isscalar(matlab_datenum):
        ordinal = int(matlab_datenum)
        day_fraction = matlab_datenum % 1
        date = datetime.datetime.fromordinal(ordinal) + datetime.timedelta(days=day_fraction) - datetime.timedelta(days=366)
        return date
    else:
        ordinal = np.floor(matlab_datenum).astype(int)
        day_fraction = matlab_datenum - ordinal
        dates = np.array([
            datetime.datetime.fromordinal(int(o)) + datetime.timedelta(days=df) - datetime.timedelta(days=366)
            for o, df in zip(ordinal, day_fraction)
        ])
        return dates

def interpolate_variable(var_data, alt_data, common_alt_grid, method='linear'):
    """
    Interpolates a variable onto a common altitude grid.
    
    Parameters:
      var_data: 2D array (altitude, time) of variable measurements.
      alt_data: 2D array (altitude, time) corresponding to altitude values.
      common_alt_grid: 1D array of common altitude points to interpolate to.
      method: String specifying the interpolation method (e.g., 'linear', 'cubic').
      
    Returns:
      A 2D array of interpolated data with shape (len(common_alt_grid), number_of_time_points).
    """
    num_common_alt = len(common_alt_grid)
    num_time = var_data.shape[1]
    interp_data = np.full((num_common_alt, num_time), np.nan)

    # Interpolate each time slice onto the common altitude grid.
    for t in range(num_time):
        valid_mask = ~np.isnan(var_data[:, t])
        if np.sum(valid_mask) > 1:
            valid_altitudes = alt_data[valid_mask, t]
            valid_values = var_data[valid_mask, t]
            # Create an interpolation function; values outside range become NaN.
            f = interp1d(valid_altitudes, valid_values, kind=method, bounds_error=False, fill_value=np.nan)
            interp_data[:, t] = f(common_alt_grid)
    return interp_data

def interpolate_errors(var_err, alt_data, common_alt_grid, method='linear', variance_based=False):
    """
    Interpolates error arrays onto a common altitude grid.
    
    Parameters:
      var_err: 2D array (altitude, time) of standard deviations.
      alt_data: 2D array (altitude, time) corresponding to variable measurements.
      common_alt_grid: 1D array of common altitude points.
      method: Interpolation method to use ('linear', 'cubic', etc.).
      variance_based: If True, interpolates variances (squared errors) then converts back.
      
    Returns:
      A 2D array of interpolated errors.
    """
    if variance_based:
        # Convert standard deviations to variances, interpolate, then convert back.
        var_err_squared = var_err ** 2
        interp_variances = interpolate_variable(var_err_squared, alt_data, common_alt_grid, method=method)
        interp_errors = np.sqrt(interp_variances)
    else:
        interp_errors = interpolate_variable(var_err, alt_data, common_alt_grid, method=method)
    return interp_errors

def eiscat_netCDF_loader(eiscat_file, common_altitude=None, method='linear'):

    """
    Processes EISCAT data from a given file. If the file is a MATLAB .mat file, the function converts MATLAB dates, converts altitude values to kilometers (if needed), interpolates the data to a common altitude grid, and returns an
    xarray.Dataset. Otherwise, it attempts to load the file as an xarray.Dataset.
    
    Enhanced Robustness:
      - Checks whether altitude values are in meters or kilometers. It computes the minimum nonzero altitude; if it is greater than 5,000, it assumes the values are in meters and converts them.
      - Derives a common altitude grid (in km) if not provided.
      - Uses flexible variable names for keys (e.g., 'alt' or 'Alt') for robustness.
    
    Parameters:
      eiscat_file: str, path to the file containing EISCAT data.
      common_altitude: array-like, common altitude grid (in km) to interpolate to.
                       If None, a grid is defined based on the altitude range in the file.
      method: str, interpolation method to use ('linear', 'cubic', etc.).
      
    Returns: An xarray.Dataset with interpolated EISCAT data and associated metadata.
    """
    import os, sys, scipy.io, datetime, numpy as np, pandas as pd, xarray as xr
    from scipy.interpolate import interp1d

    if not os.path.exists(eiscat_file):
        raise FileNotFoundError(f"EISCAT file '{eiscat_file}' not found.")

    # Check file extension; if not .mat, try to load with xarray.
    file_ext = os.path.splitext(eiscat_file)[1].lower()
    if file_ext != '.mat':
        #print("Warning: The provided file does not have a .mat extension.")
        print("Attempting to load the file as an xarray.Dataset (e.g., if it is in NetCDF format)...")
        try:
            ds = xr.open_dataset(eiscat_file)
            print("File successfully loaded as an xarray.Dataset.")
            return ds
        except Exception as e:
            raise ValueError("The file is neither a valid .mat file nor an xarray-compatible dataset. Error encountered: " + str(e))
    
    # Load the MATLAB .mat file.
    eiscat_vars = scipy.io.loadmat(eiscat_file)
    
    # Use the helper function to allow flexible variable names.
    times1    = get_variable(eiscat_vars, ['time1', 'Time1'])
    times2    = get_variable(eiscat_vars, ['time2', 'Time2'])
    altitude  = get_variable(eiscat_vars, ['alt', 'Alt'])
    azimuth   = get_variable(eiscat_vars, ['az', 'Az'])
    elevation = get_variable(eiscat_vars, ['el', 'El'])
    Ne        = get_variable(eiscat_vars, ['ne', 'Ne'])
    Ne_err    = get_variable(eiscat_vars, ['ne_err', 'Ne_err'])
    Te        = get_variable(eiscat_vars, ['te', 'Te'])
    Te_err    = get_variable(eiscat_vars, ['te_err', 'Te_err'])
    Ti        = get_variable(eiscat_vars, ['ti', 'Ti'])
    Ti_err    = get_variable(eiscat_vars, ['ti_err', 'Ti_err'])
    Vi        = get_variable(eiscat_vars, ['vi', 'Vi'])
    Vi_err    = get_variable(eiscat_vars, ['vi_err', 'Vi_err'])

    # Convert MATLAB serial dates to Python datetime objects.
    datetime1 = matlab_to_datetime(times1.squeeze())
    datetime2 = matlab_to_datetime(times2.squeeze())

    # Create pandas DatetimeIndex objects for time manipulation.
    time1_pd = pd.to_datetime(datetime1)
    time2_pd = pd.to_datetime(datetime2)
    time_avg = time1_pd + (time2_pd - time1_pd) / 2

    # Check if altitude values are in meters or kilometers.
    min_nonzero = np.nanmin(altitude[altitude > 0])       # Compute the minimum nonzero non-NaN altitude.
    if min_nonzero > 5000:
        print("** Important to Note here::")
        print("Minimum nonzero altitude is greater than 5,000; assuming altitude is in meters. Converting to kilometers.")
        altitude_km = altitude / 1000.0
    else:
        print("Altitude values appear to be in kilometers. No conversion needed.")
        altitude_km = altitude

    # Define the common altitude grid:
    # If the user did not provide one, derive it based on the actual altitude range in km.
    if common_altitude is None:
        alt_min = np.nanmin(altitude_km)
        alt_max = np.nanmax(altitude_km)
        # Convert alt_min and alt_max to integers (in km)
        alt_min_int = int(alt_min)
        alt_max_int = int(alt_max)
        # Create an array from alt_min_int to alt_max_int (inclusive) in steps of 2 km
        common_altitude = np.arange(alt_min_int, alt_max_int + 1, 2)

    # Interpolate the measurement variables onto the common altitude grid.
    Ne_interp = interpolate_variable(Ne, altitude_km, common_altitude, method=method)
    Te_interp = interpolate_variable(Te, altitude_km, common_altitude, method=method)
    Ti_interp = interpolate_variable(Ti, altitude_km, common_altitude, method=method)
    Vi_interp = interpolate_variable(Vi, altitude_km, common_altitude, method=method)

    # Interpolate the error arrays using variance-based interpolation.
    Ne_err_interp = interpolate_errors(Ne_err, altitude_km, common_altitude, method=method, variance_based=True)
    Te_err_interp = interpolate_errors(Te_err, altitude_km, common_altitude, method=method, variance_based=True)
    Ti_err_interp = interpolate_errors(Ti_err, altitude_km, common_altitude, method=method, variance_based=True)
    Vi_err_interp = interpolate_errors(Vi_err, altitude_km, common_altitude, method=method, variance_based=True)

    # Prepare data variables with metadata for the final xarray.Dataset.
    interpolated_data_vars = {
        'Ne': (['altitude', 'time'], Ne_interp, {'units': 'm^-3', 'description': 'Electron Density interpolated to common altitude grid'}),
        'Te': (['altitude', 'time'], Te_interp, {'units': 'K', 'description': 'Electron Temperature interpolated to common altitude grid'}),
        'Ti': (['altitude', 'time'], Ti_interp, {'units': 'K', 'description': 'Ion Temperature interpolated to common altitude grid'}),
        'Vi': (['altitude', 'time'], Vi_interp, {'units': 'm/s', 'description': 'LOS Ion Velocity interpolated to common altitude grid'}),
        'Ne_err': (['altitude', 'time'], Ne_err_interp, {'units': 'm^-3', 'description': 'Error in Electron Density'}),
        'Te_err': (['altitude', 'time'], Te_err_interp, {'units': 'K', 'description': 'Error in Electron Temperature'}),
        'Ti_err': (['altitude', 'time'], Ti_err_interp, {'units': 'K', 'description': 'Error in Ion Temperature'}),
        'Vi_err': (['altitude', 'time'], Vi_err_interp, {'units': 'm/s', 'description': 'Error in LOS Ion Velocity'})
    }

    # Non-interpolated angular variables with metadata.
    non_interpolated_data_vars = {
        'azimuth': (['time'], azimuth.squeeze(), {'units': 'degrees', 'description': 'Azimuth angle'}),
        'elevation': (['time'], elevation.squeeze(), {'units': 'degrees', 'description': 'Elevation angle'})
    }

    # Set up coordinate variables with metadata.
    time_coords = {
        'time': ('time', time_avg, {'description': 'Average time between time1 and time2'}),
        'time1': ('time', datetime1, {'description': 'Start time'}),
        'time2': ('time', datetime2, {'description': 'End time'}),
        'azimuth': ('time', azimuth.squeeze(), {'units': 'degrees', 'description': 'Azimuth angle'}),
        'elevation': ('time', elevation.squeeze(), {'units': 'degrees', 'description': 'Elevation angle'})
    }
    altitude_coords = {
        'altitude': ('altitude', common_altitude, {'units': 'km', 'description': 'Common altitude grid'})
    }

    # Assemble the final xarray.Dataset.
    eiscat_dataset = xr.Dataset(
        data_vars=interpolated_data_vars,
        coords={**time_coords, **altitude_coords},
        attrs={
            'description': 'EISCAT measurements interpolated to common altitude grid',
            'source_file': eiscat_file,
            'creator': 'process_eiscat_data function',
            'creation_date': pd.Timestamp.now().isoformat()
        }
    )

    # Add non-interpolated angular variables.
    for var_name, (dims, data, attrs) in non_interpolated_data_vars.items():
        eiscat_dataset[var_name] = (dims, data, attrs)

    return eiscat_dataset

def load_full_eiscat_netCDF_data(directory, method='linear'):
    # List all files in the directory (adjust the file type if necessary)
    files = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.nc')]  # Example: .nc files

    # Process each file and load the xarray.Dataset into a list
    ds_list = [eiscat_netCDF_loader(file, method=method) for file in files]

    # Convert each xarray to a pandas DataFrame
    dfs = [x.to_dataframe() for x in ds_list]
    concatenated_df = pd.concat(dfs, axis=0)
    df = concatenated_df.reset_index()
    df = df[df['time'] >= '2013-01-01']
    df = df.sort_values(by=['altitude','time'])

    # Fixed EISCAT site coordinates
    lat_eiscat = 69.5864  # degrees
    lon_eiscat = 19.2272  # degrees
    R = 6378  # Earth's radius in km

    # Convert fixed site coordinates to radians
    lat_eiscat_rad = np.radians(lat_eiscat)
    lon_eiscat_rad = np.radians(lon_eiscat)

    # Assume 'df' is your xarray.Dataset with variables: 'azimuth', 'elevation', and index 'altitude'

    # Convert azimuth and elevation angles to radians
    df['azimuth_rad'] = np.radians(df['azimuth'])
    df['elevation_rad'] = np.radians(df['elevation'])

    # Compute slant distance
    df['observation_distance'] = df['altitude'] / np.sin(df['elevation_rad'])

    # Compute observation latitude
    df['observation_latitude'] = np.degrees(
        lat_eiscat_rad + (df['observation_distance'] * np.cos(df['azimuth_rad']) / R)
    )

    # Compute observation longitude
    df['observation_longitude'] = np.degrees(
        lon_eiscat_rad + (df['observation_distance'] * np.sin(df['azimuth_rad']) / 
        (R * np.cos(lat_eiscat_rad)))
    )

    # Drop intermediate calculation variables if not needed
    df = df.drop(columns=['azimuth_rad', 'elevation_rad', 'observation_distance', 'time1', 'time2'])
    df = df.reset_index(drop=True)

    return df