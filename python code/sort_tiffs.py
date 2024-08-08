#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 00:06:43 2024

@author: emmaodom
"""

import os
import re
import numpy as np
import tifffile as tiff
from skimage.io import imread, imsave
from scipy.ndimage import maximum

#%% * define functions and file naming patterns
def load_and_concatenate(files):
    images = [tiff.imread(file) for file in sorted(files)]
    return np.stack(images, axis=0)

def process_directory(animal_dir, pattern, label_pattern):
    group_filenames = {} #dictionary
    label = None
    for file in os.listdir(animal_dir):
        file_path = os.path.join(animal_dir,file)
        if os.path.isfile(file_path):
            #get label / animal_ID
            if not label:  # Only need to find the label once
                label_match = label_pattern.search(file)
                if label_match:
                    label = label_match.group(1)
            #get slice num, z , channel
            match = pattern.search(file)
            if match:
                slice_num = int(match.group(1))
                z_plane = int(match.group(2))
                channel = int(match.group(3))
                if slice_num not in group_filenames:
                    #create dictionary key for this slice
                    group_filenames[slice_num] = {}
                if channel not in group_filenames[slice_num]:
                    #creates channel key nested under slice
                    group_filenames[slice_num][channel] = []
                #append file path to correct slice_channel group
                group_filenames[slice_num][channel].append(file_path)
    #at end of for loop above, all tiff filenames should be sorted into slice_channel groups
    for slice_num, channels in group_filenames.items():
        for channel, files in channels.items():
            concatenated = load_and_concatenate(files)
            output_file = os.path.join(animal_dir, f'{label}_S{slice_num}_ch{channel}.tif')
            tiff.imwrite(output_file, concatenated)
            print(f"Saved {output_file}")
            for file in files:
                os.remove(file)
                print(f"Deleted {file}")
    # marker file to indicate the directory has been processed
    marker_file = os.path.join(animal_dir, 'concatenated.marker')
    open(marker_file, 'a').close()
    print(f"Created marker file {marker_file}")
    return group_filenames;
#%% test process_dir on individual directory
# Regex pattern to extract slice, z-plane, and color channel information
pattern = re.compile(r"S(\d+)_z(\d+)_ch(\d+).tif")
label_pattern = re.compile(r'([^_]+)')
animal_dir = '/Users/emmaodom/Documents/ATLAS_dualSiteInj_VISp_MOp_0724/TIFFS/exp89.animal2.VISp.ORB copy'
process_directory(animal_dir,pattern,label_pattern)
#%% * apply processing to each subdir independently
parent_dir = '/Users/emmaodom/Documents/ATLAS_dualSiteInj_VISp_MOp_0724/TIFFS2'
# Regex pattern to extract slice, z-plane, and color channel information
pattern = re.compile(r"S(\d+)_z(\d+)_ch(\d+).tif")
label_pattern = re.compile(r'([^_]+)')
# Iterate over each subdirectory in the parent directory
for subdir in os.listdir(parent_dir):
    subdir_path = os.path.join(parent_dir, subdir)
    if os.path.isdir(subdir_path):
        # Check for marker file
        marker_file = os.path.join(subdir_path, 'processed.marker')
        if os.path.exists(marker_file):
            print(f"Skipping already processed directory: {subdir_path}")
            continue
        try:
            print(f"Processing directory: {subdir_path}")
            process_directory(subdir_path, pattern, label_pattern)
        except Exception as e:
            print(f"Error processing directory {subdir_path}: {e}")

print("Processing complete.")

#%% test case for finding initial segment/image label

# Example file names
files = [
    "exp89.animal2.VISp.ORB_S3_z0_ch00.tif",
    "exp89.animal2.VISp.ACA_S2_z1_ch01.tif",
    "exp88.animal1.MOp.ACA_S1_z2_ch02.tif",
    "exp88.animal1.MOp.ORB_S4_z3_ch03.tif",
    "exp90.animal3.VISp.ACA_S5_z4_ch00.tif"
]

# Regex pattern to extract the label
pattern = re.compile(r'([^_]+)')
#pattern = re.compile(r'(exp\d+\.\w+\.\w+\.\w+)_')

# Process each file and extract the label
for file in files:
    label_match = pattern.match(file)
    if label_match:
        label = label_match.group(1)
        print(f"File: {file}")
        print(f"Extracted Label: {label}\n")
    else:
        print(f"File: {file}")
        print("No match found.\n")

#%% * function to copy all S#_ch2.tif files!! because imagej macro is DUMB

def save_max_projection_ch2_files(source_dir, target_dir):
    """
    Saves maximum Z-projected versions of all files ending with 'S#_ch2.tif' from the source directory and its subdirectories to the target directory.
    
    Parameters:
    source_dir (str): The directory to search for files.
    target_dir (str): The directory where the files will be saved.
    """
    # Create the target directory if it doesn't exist
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    # Walk through all files and subdirectories in the source directory
    for root, dirs, files in os.walk(source_dir):
        for filename in files:
            if filename.endswith('_ch2.tif'):
                # Construct the full file path
                file_path = os.path.join(root, filename)
                
                # Load the image
                img = imread(file_path)
                
                # Perform maximum Z-projection
                max_projection = np.max(img, axis=0)
                
                # Save the max projection as a TIFF to the target directory
                save_path = os.path.join(target_dir, os.path.relpath(file_path, source_dir))
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                tiff.imwrite(save_path, max_projection.astype(np.uint16))
                print(f'Saved max projection: {save_path}')

#%% * call function
source_directory = '/Users/emmaodom/Documents/ATLAS_dualSiteInj_VISp_MOp_0724/TIFFS2'
target_directory = '/Users/emmaodom/Documents/ATLAS_dualSiteInj_VISp_MOp_0724/Results/All_Ch2_Max'
save_max_projection_ch2_files(source_directory, target_directory)



#%% quick fix filenames, replace part of string

directory = '/Users/emmaodom/Documents/ATLAS_dualSiteInj_VISp_MOp_0724/Results/All_Ch2_Max/exp88.animal2.MOp.ACA'

for filename in os.listdir(directory):
    if "exp88.animal2.MOp" in filename:
        new_filename = filename.replace("exp88.animal2.MOp", "exp88.animal2.MOp.ACA")
        os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename))
        print(f'Renamed: {filename} -> {new_filename}')

print("Renaming completed.")

