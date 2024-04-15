#!/usr/bin/env python3
"""
Train and test a model to predict cycling time from GPX tracks.

Created on Sat Apr 13 11:11:40 2024

@author: gperonato
"""

import geopandas as gpd
from shapely.geometry import LineString, MultiLineString
import fiona
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
import joblib

def gpx_points_to_segments(points_gdf):
    """
    Create segments from gpx track points

    Parameters
    ----------
    points_gdf : GeoDataFrame
        GDF with timestamp and elevation.

    Returns
    -------
    lines_gdf : GeoDataFrame
        GDF with statistics.

    """
    
    # Initialize empty lists
    lines = []
    dts = []
    elevs = []
    slopes = []
    times_from_break = []
    times_last_break = []
    
    # Initialize counters
    time_from_break = 0
    last_break_time = np.nan
    
    # Iterate through the points to create line segments
    for i in range(len(points_gdf) - 1):
        # Get the current and next point
        current_point = points_gdf.geometry.iloc[i]
        next_point = points_gdf.geometry.iloc[i + 1]
        
        dt = points_gdf.time.iloc[i+1] - points_gdf.time.iloc[i]
        ele = points_gdf.ele.iloc[i+1] - points_gdf.ele.iloc[i]
        
        # Remove breaks
        if dt.seconds > 20:
            time_from_break = 0
            last_break_time = dt.seconds/60
            continue
        
        # Create a line segment
        line = LineString([current_point, next_point])
        if line.length == 0:
            continue
        
        # Append to lists
        lines.append(line)
        dts.append(dt)
        elevs.append(ele)
        slopes.append(ele/line.length)
        times_from_break.append(time_from_break)
        times_last_break.append(last_break_time)
        
        # Update counters
        time_from_break += dt.seconds/60
    
    # Create a GeoDataFrame for the line segments
    lines_gdf = gpd.GeoDataFrame(geometry=lines, crs=points_gdf.crs)
    
    # Assign attributes
    lines_gdf["dtime"] = dts
    lines_gdf["elevation"] = elevs
    lines_gdf["slope"] = slopes
    lines_gdf["minutes"] = lines_gdf["dtime"].apply(lambda x: x.seconds/60)
    lines_gdf["speed"] = (lines_gdf.length/1000) / lines_gdf["dtime"].apply(lambda x: x.seconds/3600)
    lines_gdf["time_from_break"] = times_from_break
    lines_gdf["time_last_break"] = times_last_break

    lines_gdf["cum_pos_elevation"] = lines_gdf.loc[lines_gdf.elevation>0,"elevation"].cumsum()
    lines_gdf["cum_pos_elevation"] = lines_gdf["cum_pos_elevation"].ffill().fillna(0)
    
    lines_gdf["cum_time_mins"] = lines_gdf["dtime"].cumsum().apply(lambda x: x.seconds/60)
    lines_gdf["cum_dist"] = lines_gdf.length.cumsum()/1000
    lines_gdf["length"] = lines_gdf.length
    
    return lines_gdf


def unary_union(group):
    merged_geometry = group.geometry.unary_union
    return merged_geometry

def resample(gdf, target_length=100, window_length=1000):
    """
    Resample the GPX segments to a target length.

    Parameters
    ----------
    gdf : GeoDataFrame
        Geodataframe with segments.
    target_length : int, optional
        Length of the target segments. The default is 10 m.
    window_length : int, optional
        Length of the rolling window for additional statistics. The default is 1000 m.

    Returns
    -------
    resampled : GeoDataFrame
        Resampled line geometry.

    """
    crs = gdf.crs
    gdf["sequential_group"] = (gdf.length.cumsum()/100).round().astype(int)
    resampled = gdf.groupby('sequential_group').agg({"geometry": unary_union,
                                                 "dtime": "sum",
                                                 "minutes": "sum",
                                                 "elevation": "sum",
                                                 "length": "sum",
                                                 "slope": "mean",
                                                 "cum_dist": "last",
                                                 "cum_pos_elevation": "last",
                                                 "cum_time_mins": "last",
                                                 "time_from_break": "first",
                                                 "time_last_break": "first"
                                                 })
    # Calculate rolling window of cumulated elevation
    resampled["cum_ele_rolling"] = resampled["elevation"].rolling(window=window_length//target_length).sum().fillna(0)
    resampled["speed"] = (resampled["length"]/1000) / (resampled["minutes"]/60)

    # Recreate GDF
    resampled = gpd.GeoDataFrame(resampled)
    resampled = resampled.set_geometry("geometry")
    resampled.crs = crs
    return resampled

if __name__ == "__main__":
        
    points_gdf = gpd.read_file("gpx_orig/activity_14671204237.gpx",
                                layer = "track_points",
                                # layer = "waypoints" # as exported by anonymize.py
                               )
        
    points_gdf = points_gdf.to_crs("EPSG:3857") # Projected Web Mercator
    
    # Make sure time is parsed correctly
    points_gdf['time'] = pd.to_datetime(points_gdf['time'], format='ISO8601')

    gdf = resample(gpx_points_to_segments(points_gdf))
    df = gdf.drop("geometry", axis=1)
    
    # Create a StandardScaler object
    scaler = StandardScaler()

    # Splitting the data into features (X) and target variable (y)
    features = ["elevation", "length", "cum_dist", "cum_ele_rolling"]
    target = "minutes"
    X = df[features]
    y = df[target]  # Target variable
    
    # Shuffle the dataset
    X, y = shuffle(X, y, random_state=42)
    
    # Splitting the data into training and testing sets (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Fit the scaler to the training data and transform it
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Transform the testing data using the scaler fitted on the training data
    X_test_scaled = scaler.transform(X_test)
    
    # Creating a linear regression model
    model = LinearRegression()
    
    # Training the model using the training sets
    model.fit(X_train_scaled, y_train)
    
    # Making predictions on the testing set
    y_pred = model.predict(X_test_scaled)
    
    print("Actual time:", round(sum(y_test),1), "minutes")
    print("Predicted time:", round(sum(y_pred),1), "minutes")
    
    # Evaluating the model
    rmse = root_mean_squared_error((y_test/X_test.length) * 1000, (y_pred/X_test.length)*1000)
    print("Root Mean Squared Error:", round(rmse,4), "minutes per km")
    
    mean_cumul_error = ((y_pred.sum() - y_test.sum())/y_test.sum()) * 100
    print("Cumulated Mean Error:", round(mean_cumul_error,3), "%")
    
    # Accessing the coefficients (weights) of the features
    coefficients = model.coef_
    intercept = model.intercept_
    
    # Mapping coefficients to feature names
    feature_coefficients = dict(zip(X.columns, coefficients))
    
    # Feature coefficients
    feature_coefficients = pd.Series(feature_coefficients).sort_values(ascending=False)
    pd.Series(feature_coefficients).plot(kind="bar", title="Feature coefficients")
 

    scaler_filename = "scaler.save"
    joblib.dump(scaler, "scaler.save") 
    joblib.dump(model, "model.save") 
