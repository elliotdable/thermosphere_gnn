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
import matplotlib.pyplot as plt

def plot_observation_locations_2D(dfs_to_plot, df_names, sizes):

    # Create a map with Cartopy
    fig, ax = plt.subplots(figsize=(8, 4), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent([10, 32, 63, 75], crs=ccrs.PlateCarree())  # Covers Scandinavia

    # Add features
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.LAND, edgecolor='black', facecolor='lightgray')

    for df, df_name, size in zip(dfs_to_plot, df_names, sizes):
        unique_pairs = df[['observation_longitude', 'observation_latitude']].drop_duplicates()
        ax.scatter(unique_pairs['observation_longitude'], unique_pairs['observation_latitude'], marker='o', s=size, label=df_name, 
                   color='red', zorder=5)

    # Add latitude lines across the x-axis
    gl = ax.gridlines(draw_labels=True, linestyle="--", alpha=0.5)
    gl.top_labels = False  # Hide top labels
    gl.right_labels = False  # Hide right labels
    gl.xlines = False  # Hide longitude lines
    gl.ylocator = plt.MaxNLocator(6)  # Adjust number of latitude lines

    plt.legend()
    plt.title("Thermosphere-Ionosphere Observations over Scandinavia")
    plt.show()

def plot_observation_locations_3D(dfs_to_plot, df_names, elevation, azimuth):
    """
    Plots 3D observation locations for multiple datasets.

    Parameters:
    - dfs_to_plot: List of DataFrames to plot.
    - df_names: List of corresponding names for the DataFrames.
    - elevation: Elevation angle for 3D plot.
    - azimuth: Azimuth angle for 3D plot.
    """
    # Define the observation_latitude/observation_longitude bounds for Scandinavia
    lat_min, lat_max = 65, 75  # Latitude range
    lon_min, lon_max = 0, 30   # Longitude range
    R_earth = 6371  # Earth's average radius in km

    # Create a 3D figure
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the Earth's surface for the region
    u, v = np.mgrid[lon_min:lon_max:50j, lat_min:lat_max:25j]
    X = R_earth * np.cos(np.radians(v)) * np.cos(np.radians(u))
    Y = R_earth * np.cos(np.radians(v)) * np.sin(np.radians(u))
    Z = R_earth * np.sin(np.radians(v))
    ax.plot_surface(X, Y, Z, color='lightblue', alpha=0.3, edgecolor='k')

    # Process each DataFrame
    for df, df_name in zip(dfs_to_plot, df_names):
        # Filter data based on elevation and geographical bounds
        df_filtered = df[(df['observation_latitude'] >= lat_min) & (df['observation_latitude'] <= lat_max) &
                         (df['observation_longitude'] >= lon_min) & (df['observation_longitude'] <= lon_max)]

        # Convert degrees to radians
        lat_radians = np.radians(df_filtered['observation_latitude'])
        lon_radians = np.radians(df_filtered['observation_longitude'])

        # Compute Cartesian coordinates
        alt_km = df_filtered['altitude']  # Assuming altitude is in km
        r = R_earth + alt_km  # Total radius at given altitude
        x = r * np.cos(lat_radians) * np.cos(lon_radians)
        y = r * np.cos(lat_radians) * np.sin(lon_radians)
        z = r * np.sin(lat_radians)

        # Plot data points
        ax.scatter(x, y, z, s=1, alpha=0.3, label=df_name)

    # Labels and title
    ax.set_title("3D Spherical Plot of Observations Over Scandinavia")
    ax.set_xlabel("Radius (km)")
    ax.set_ylabel("Radius (km)")
    ax.set_zlabel("Radius (km)")

    # Adjust viewing angle
    ax.view_init(elev=elevation, azim=azimuth)

    # Remove axis details for better visualization
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)
    ax.set_axis_off()

    # Show legend and plot
    plt.legend()
    plt.show()
