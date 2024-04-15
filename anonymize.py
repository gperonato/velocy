#!/usr/bin/env python3
"""
Anonimize GPX

Created on Mon Apr 15 21:36:37 2024

@author: gperonato
"""

import pandas as pd
import geopandas as gpd
import numpy as np
import os

for filename in os.listdir("gpx_orig"):
    print(filename)
    points_gdf = gpd.read_file(os.path.join("gpx_orig", filename), layer="track_points")
    
    # Make sure time is parsed correctly
    points_gdf['time'] = pd.to_datetime(points_gdf['time'], format='ISO8601')

    lat = 0 # latitude is preserved to keep projected distances
    lon = np.random.uniform(-90, +90) # degrees diff
    days = np.random.randint(-30, +30) # days diff
    minutes = np.random.randint(-30, +30) # minutes diff
    year = np.random.randint(1970, 2050) # year 
    elevation = np.random.uniform(-100, +100) # m diff
     
    points_gdf.geometry = points_gdf.translate(lon, lat)
    points_gdf.time = points_gdf.time + pd.Timedelta(days, "days") + pd.Timedelta(minutes, "minutes")
    points_gdf.time =  points_gdf['time'].apply(lambda x: x.replace(year=year))
    points_gdf.ele += elevation
    
    points_gdf[["ele","time","geometry"]].to_file(os.path.join("gpx",filename))