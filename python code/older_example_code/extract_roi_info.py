#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 21:27:17 2024

@author: emmaodom
"""
import os
import re
import glob
import numpy as np
import pandas as pd
import read_roi
import zipfile
#from read_roi import read_roi_file
from read_roi import read_roi_zip
from scipy.spatial import distance
from src import helper_functions as hf

def get_roi_zip_path(directory):
    '''
    The purpose of this function is to search a directory (imaging session folder)
    for the RoiSet.zip file containing ROI data.

    Parameters
    ----------
    directory : str
        The path to the directory to search.

    Returns
    -------
    roi_set_zip : str or None
        The file path to the RoiSet.zip file, or None if not found.
    '''
    # Search for RoiSet.zip in the directory
    roi_zip_files = glob.glob(os.path.join(directory, 'RoiSet.zip'))
    # Check if any RoiSet.zip files were found
    if len(roi_zip_files) == 0:
        print("No RoiSet.zip files found.")
        return None
    # Assuming only one RoiSet.zip file per directory
    roi_zip_path = roi_zip_files[0]
    return roi_zip_path

def get_roi_sess(directory):
    roi_zip_path = get_roi_zip_path(directory)
    if roi_zip_path is None:
        return None
    try:
        rois = read_roi_zip(roi_zip_path)
    except zipfile.BadZipFile:
        print(f"Error: File is not a zip file - {roi_zip_path}")
        return None
    except Exception as e:
        print(f"An error occurred while processing {roi_zip_path}: {e}")
        return None
    ROIs = pd.DataFrame.from_dict(rois, orient='index')
    #get microns per pixel from metadata
    microns_per_pixel = hf.get_microns_per_pixel(directory)
    #add scaled area as column to df #scaling_factor = microns_per_pixel**2
    #i hope microns x,y in the metadata updates based on objective AND optical zoom!!
    if microns_per_pixel is not None:
        ROIs['area (u^2)'] = np.pi * (ROIs['width'] / 2) * (ROIs['height'] / 2) * (microns_per_pixel**2)
    else:
        ROIs['area (u^2)'] = None
    ROIs['directory'] = directory
    #get cell id and dend id
    match = re.search(r'(?i)_cell(\d+)_dend(\d+)_', directory)
    if match:
        cell_id = match.group(1)
        dend_id = match.group(2)
    else:
        cell_id = None
        dend_id = None
    ROIs['cell'] = cell_id
    ROIs['dend'] = dend_id
    ROIs['date'] = hf.get_imaging_date(directory)
    # Search for the pattern in the directory string
    match_id = re.search(r'\/(\d{3}[A-Za-z])\/', directory)
    # Extract and return the animal ID if found
    if match_id:
        animal_id = match_id.group(1)
    else:
        animal_id = None
    ROIs['animal'] = animal_id
    return ROIs

def pool_roi_info(input_dir):
    data = pd.DataFrame()
    for root, dirs, files in os.walk(input_dir):
        for dir in dirs:
            directory = os.path.join(root, dir)
            ROIs = get_roi_sess(directory)
            if ROIs is not None:
                data = pd.concat([data, ROIs], ignore_index=True)
    #Put identifier info at front of dataframe. 
    new_column_order = ['directory','animal','cell', 'dend', 'date'] + [col for col in data.columns if col not in ['directory','animal', 'cell', 'dend', 'date']]
    # Reorder the DataFrame columns
    data = data[new_column_order]
    return data

def get_roi_distances(directory, shape = 'matrix'):
    '''
    example uses: 
        dist_matrix = get_roi_distances(directory, 'matrix')
        dist_array = get_roi_distances(directory, 'array')

    Parameters
    ----------
    directory : TYPE
        DESCRIPTION.
    shape : TYPE, optional
        DESCRIPTION. The default is 'matrix'.

    Returns
    -------
    TYPE
        DESCRIPTION.

    '''
    ROIs = get_roi_sess(directory)
    ROIs = ROIs.iloc[::2]
    # Calculate center coordinates for each ROI
    centers = np.column_stack(((ROIs['left'] + ROIs['width'] / 2), 
                           (ROIs['top'] + ROIs['height'] / 2)))
    dist_matrix = distance.cdist(centers, centers, 'euclidean')
    mask = np.triu(np.ones_like(dist_matrix, dtype=bool)) #select upper right half triangle
    if shape == 'matrix':
        dist_matrix_mask = dist_matrix[mask]
        dist_matrix_ = np.zeros_like(dist_matrix)
        dist_matrix_[mask] = dist_matrix_mask       
        return dist_matrix_
    if shape == 'array':
        distances = dist_matrix[mask]
        return distances


directory = '/Volumes/T7/Motor_Spines_Pilot_Data/289N/231024/289N_231024_Cell1_dend1_920nm_20x_10xd_Tseries_56.05um_512_512px_4avg-045'
ROIs = get_roi_sess(directory)
input_dir = '/Volumes/T7/Motor_Spines_Pilot_Data'
all_ROIs = pool_roi_info(input_dir)

#ADD SAVE LINES 
'''
path_parent = '/Volumes/T7/Motor_Spines_Pilot_Data/'
dir_save = 'python_dataframes'
path_save = path_parent + dir_save
if not os.path.exists(path_save):
    os.makedirs(path_save)
# Save the dataframe to a CSV file in python_dataframes dir
all_ROIs.to_csv(os.path.join(path_save, 'master_roi_info.csv'), index=False)
print('all_ROIs Dataframe saved to CSV file at:', path_save)
'''


'''Original pre function code
# Path to the RoiSet.zip file
roi_zip_path = '/Volumes/T7/Motor_Spines_Pilot_Data/289N/231024/289N_231024_Cell1_dend1_920nm_20x_10xd_Tseries_56.05um_512_512px_4avg-045/RoiSet.zip'
rois = read_roi_zip(roi_zip_path)
ROIs = pd.DataFrame.from_dict(rois, orient='index')
directory = os.path.dirname(roi_zip_path)
microns_per_pixel = hf.get_microns_per_pixel(directory)
#scaling_factor = microns_per_pixel**2
ROIs['area (u^2)'] = np.pi * (ROIs['width'] / 2) * (ROIs['height'] / 2) * (microns_per_pixel**2)
'''


