#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 12:40:10 2024

@author: emmaodom
"""
#***
import os
import re
import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import read_roi
import zipfile
import skimage as ski
import cv2
from scipy import stats
#from read_roi import read_roi_file
from read_roi import read_roi_zip
from scipy.spatial import distance
from scipy.stats import zscore
#from scipy.stats import mode
from skimage.io import imread
from skimage import io, exposure
from matplotlib.patches import Ellipse, RegularPolygon
from scipy.interpolate import PchipInterpolator

''' LOOKOUT 
#BE CAREFUL WITH DESPECKLE TO VIP CHANNEL, I WANT TO MAKE SURE IT IS CONSISTENTLY APPLIED
if the results look good after despeckle, just apply to the raw images and save them that way. 
if ch==1:
    temp_max_proj = cv2.medianBlur(temp_max_proj, 5)
MAKE SURE THAT THE DEPTH IS CALCULATED AND RECORDED IN UM, 
MAKE SURE IT DOES NOT CREATE DUPLICATES UPON MERGING. 

oh yeah also!! make sure to eliminate double counts of same cell (ie assigned as pv and vip overlap)
might be more complicated, need to manually inspect images to see whats going on. 
'''
#%% things to do in this script...
#get file list
#max z project each channel (store temp max proj image)
#apply roi.zip as mask
#measure features of roi  
#group files by slice 
#get animal label to save results in 
#we should make one master csv with measurements for same rois across all channels 
#and just add a tag for slice number and channel to enable pandas based sorting. 

#the temp max projection
#overrlay rois
#measure features
#find you spine analysis function for looking at rois. 

#pool roi info from extract_roi_info might be useful

#%% * ROI functions
def get_roi_zip_path(directory):
    '''
    This function returns any file under directory that has RoiSet.zip in the filename

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
    roi_zip_files = glob.glob(os.path.join(directory, '*RoiSet.zip'))
    # Check if any RoiSet.zip files were found
    if len(roi_zip_files) == 0:
        print("No RoiSet.zip files found.")
        return None
    return roi_zip_files

def get_roi_info(roi_zip_path):
    '''
    get position and shape data from roi.zip files
    see roi_overlay_trial_code script
    '''
    try:
        rois = read_roi_zip(roi_zip_path)
    except zipfile.BadZipFile:
        print(f"Error: File is not a zip file - {roi_zip_path}")
        return None
    except Exception as e:
        print(f"An error occurred while processing {roi_zip_path}: {e}")
        return None
    roi_info = pd.DataFrame.from_dict(rois, orient='index')
    #were going to want to get label, slice, channel, and add this info to df.
    return roi_info

def calculate_mode(array):
    """Calculate the mode of an array using numpy."""
    values, counts = np.unique(array, return_counts=True)
    m = counts.argmax()
    return values[m]

def overlay_roi_and_measure(roi_info,image,animalID, slice_number, channel,show_overlay=False):
    '''
    include roi_zip_path
    see roi_overlay_trial_code script
    image input should be 'temp_max_proj'
    '''
    measurements = []
    if show_overlay:
        fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
        ax.imshow(image, cmap='gray')
    #this is inefficient, does overlay/measurement per roi??? 
    for index, row in roi_info.iterrows():
        #extract roi data
        left = int(row['left'])
        top = int(row['top'])
        width = int(row['width'])
        height = int(row['height'])
        center_x = left + width / 2
        center_y = top + height / 2
        #mask roi on image
        roi_image = image[top:top + height, left:left + width]
        # Measure statistics
        area = roi_image.size
        mean_val = np.mean(roi_image)
        stddev_val = np.std(roi_image)
        mode_val = calculate_mode(roi_image)
        min_val = np.min(roi_image)
        max_val = np.max(roi_image)
        measurements.append({
        'AnimalID': animalID,
        'Slice': slice_number,
        'Channel': channel,
        'Area': area,
        'Mean': mean_val,
        'StdDev': stddev_val,
        'Mode': mode_val,
        'Min': min_val,
        'Max': max_val,
        'X': center_x,
        'Y': center_y,
        'Width': width,
        'Height': height
        })
        # Optionally add the ROI as an ellipse on the plot
        if show_overlay:
            roi_oval = Ellipse((center_x, center_y), width, height, edgecolor='red', facecolor='none', linewidth=1)
            ax.add_patch(roi_oval)
    if show_overlay:
        plt.show()
    roi_measurements = pd.DataFrame(measurements)
    return roi_measurements

def get_roi_results(csv_path):
    '''
    this function will take the roi_measurements/results.csv and create a formatted dataframe. 
    call after making measurements for same rois across all channels 

    '''
    df = pd.read_csv(csv_path)
    return roi_df

def pool_roi_results(input_dir,save_path):
    '''update to save csv
    This function pools results.csv/measurement information across the entire dataset!
    '''
    all_measurements = pd.DataFrame()
    # Traverse the directory and find all CSV files
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith("_measurements.csv"):
                csv_path = os.path.join(root, file)
                # Read the CSV and concatenate to the DataFrame
                df = pd.read_csv(csv_path)
                all_measurements = pd.concat([all_measurements, df], ignore_index=True)
    output_csv_path = os.path.join(save_path, "pooled_roi_measurements.csv")
    all_measurements.to_csv(output_csv_path, index=False)
    print(f"Saved measurements to {output_csv_path}")
    return all_measurements
#%%test roi info extraction and roi overlay!
roi_zip_path = '/Users/emmaodom/Documents/ATLAS_dualSiteInj_VISp_MOp_0724/Results/ROI_zip/exp88.animal1.MOp.ACA_S1_ch2.tif_RoiSet.zip'
roi_info = get_roi_info(roi_zip_path)

#future improvement merge the info_results but not necessary
csv_path = '/Users/emmaodom/Documents/ATLAS_dualSiteInj_VISp_MOp_0724/Results/ROI_results/exp88.animal1.MOp.ACA_S1_ch2.tif_results.csv'
roi_results = pd.read_csv(csv_path)

# option 1 display image (for measurements) winner.
tiff_path = '/Users/emmaodom/Documents/ATLAS_dualSiteInj_VISp_MOp_0724/Results/All_Ch2_Max/exp88.animal1.MOp.ACA/exp88.animal1.MOp.ACA_S1_ch2.tif'
image = imread(tiff_path)

fig, ax = plt.subplots()
ax.imshow(image, cmap='gray')  

animalID = 'exp88.animal1.MOp.ACA'
slice_number = '2'
ch=0
roi_measurements = overlay_roi_and_measure(roi_info, image, animalID, slice_number, ch, show_overlay=False)
#%% #option 2 display contrast adjusted image (for figures)
image = io.imread(tiff_path)
# Adjust the contrast using Contrast Stretching
p2, p98 = np.percentile(image, (2, 98))
img_rescale = exposure.rescale_intensity(image, in_range=(p2, p98))
# Display the original and contrast-adjusted images
fig, ax = plt.subplots()
ax.imshow(img_rescale, cmap='gray') 
#%% test roi overlay, implemented in overlay_roi_and_measure make the overlay alone into a function???
temp_max_proj = image #later replace w temp_max_proj_ch# when iterating through channels
# Display the original and contrast-adjusted images
fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
ax.imshow(temp_max_proj, cmap='gray') 
# Iterate over the DataFrame and add each ROI as an oval
for index, row in roi_info.iterrows():
    # Extract ROI data
    center_x = row['left'] + row['width'] / 2
    center_y = row['top'] + row['height'] / 2
    width = row['width']
    height = row['height']
    # Create an Ellipse patch
    roi_oval = Ellipse((center_x, center_y), width, height, edgecolor='red', facecolor='none', linewidth=1)
    # Add the patch to the axes
    ax.add_patch(roi_oval)

plt.show()

#%% * execute roi measurements across channels
roi_dir = '/Users/emmaodom/Documents/ATLAS_dualSiteInj_VISp_MOp_0724/Results/ROI_zip'
roi_zip_files = get_roi_zip_path(roi_dir)
#animalID_pattern = re.compile(r'([^_]+)')
#animalID_S_pattern = re.compile(r'(exp\d+\.\w+\.\w+\.\w+_S\d+)')
animalID_S_pattern = re.compile(r'(exp\d+\.\w+\.\w+\.\w+)_S(\d+)_ch2')
parent_TIFFS = '/Users/emmaodom/Documents/ATLAS_dualSiteInj_VISp_MOp_0724/TIFFS'
save_roi_path = '/Users/emmaodom/Documents/ATLAS_dualSiteInj_VISp_MOp_0724/Results/ROI_measurements_python'
for roi_zip in roi_zip_files:
    # Identify which animal_ID & slice
    file_name = os.path.basename(roi_zip)
    print(f"Processing {file_name}")
    # Extract the animal ID and slice number using the regex pattern
    animalID_S_match = animalID_S_pattern.search(file_name)
    if animalID_S_match:
        animalID = animalID_S_match.group(1)
        slice_number = animalID_S_match.group(2)
        print(f"Animal ID: {animalID}, Slice: S{slice_number}")
        animal_tiff_path = os.path.join(parent_TIFFS,animalID)
        # Get ch2 ROI info for animal_ID_slice group (once)
        roi_info = get_roi_info(roi_zip)
        if roi_info is None:
            continue
        
        # Loop through channels ch0, ch1, ch2, ch3 for the given animal_slice
        channels = [0, 1, 2, 3]
        #IF CH1 APPLY DESPECKLE FUNCTION BEFORE MEASURE ROIS
        for ch in channels:
            # Construct the correct TIFF path for each channel
            tiff_name = f"{animalID}_S{slice_number}_ch{ch}.tif"
            tiff_path = os.path.join(animal_tiff_path, tiff_name)
            
            if not os.path.exists(tiff_path):
                print(f"TIFF file not found: {tiff_path}")
                continue
            
            # Read the TIFF image
            image = imread(tiff_path)
            temp_max_proj = np.max(image, axis=0)
            ##ALARMS be aware that there is preprocessing here that is not saved to og image and needs to be consistent!
            if ch==1:#vip
                temp_max_proj = cv2.medianBlur(temp_max_proj, 5)
            if ch==3:#sst
                temp_max_proj = cv2.medianBlur(temp_max_proj, 3)
            # Measure ROI statistics
            roi_measurements = overlay_roi_and_measure(roi_info, temp_max_proj, animalID, slice_number, ch, show_overlay=False)
            if not os.path.exists(save_roi_path):
                os.makedirs(save_roi_path)
            
            output_csv_path = os.path.join(save_roi_path, tiff_name.replace(".tif", "_measurements.csv"))
            roi_measurements.to_csv(output_csv_path, index=False)
            print(f"Saved measurements to {output_csv_path}")
            
    #break

#%% * pool roi measurements
input_dir = save_roi_path
save_path = '/Users/emmaodom/Documents/ATLAS_dualSiteInj_VISp_MOp_0724/Results'
pooled_ROI_measurements = pool_roi_results(input_dir,save_path)
# Extract the region (MOp or VISp) from the AnimalID and create a new column
pooled_ROI_measurements['Region'] = pooled_ROI_measurements['AnimalID'].apply(lambda x: 'MOp' if 'mop' in x.lower() else 'VISp')
pooled_ROI_measurements['Origin'] = pooled_ROI_measurements['AnimalID'].apply(lambda x: 'ACA' if 'aca' in x.lower() else 'ORB')
#%% check unique animal ID (labels)
animals = pooled_ROI_measurements['AnimalID'].unique()
print(animals)
#this list will not accurately reflect animalID origin after correction, since we are only correcting the Origin column. 
#%% * assign correct labels! make sure to use the origin column for future splits.
#ASSIGN CORRECT ORIGIN, leave animalID the same. 
pooled_ROI_measurements.loc[pooled_ROI_measurements['AnimalID'] == 'exp88.animal1.MOp.ORB', 'Origin'] = 'ACA'
pooled_ROI_measurements.loc[pooled_ROI_measurements['AnimalID'] == 'exp88.animal1.VISp.ORB', 'Origin'] = 'ACA'
pooled_ROI_measurements.loc[pooled_ROI_measurements['AnimalID'] == 'exp88.animal2.MOp.ACA', 'Origin'] = 'ORB'
pooled_ROI_measurements.loc[pooled_ROI_measurements['AnimalID'] == 'exp88.animal2.VISp.ACA', 'Origin'] = 'ORB'
#save updated roi measurements csv
results_dir = '/Users/emmaodom/Documents/ATLAS_dualSiteInj_VISp_MOp_0724/Results'
pooled_ROI_measurements.to_csv(os.path.join(results_dir, 'pooled_roi_measurements.csv'), index=False) #save to make sure updates are included in future steps
#%% * KDE visualization function, need to define before determining overlaps
def plot_kde_with_thresholds(pooled_animal, animal_id, thresholds, z_threshold):
    plt.figure(figsize=(10, 6))
    
    # Create a color palette for the KDE lines and thresholds
    palette = sns.color_palette("husl", n_colors=len(thresholds))
    
    # Plot KDE with hue and legend
    for channel in thresholds.index:
        sns.kdeplot(data=pooled_animal[pooled_animal['Channel'] == channel], 
                    x='Mean', 
                    label=f'Channel {channel}', 
                    color=palette[channel],
                    common_norm=False)
    
    # Add vertical lines for thresholds
    for channel in thresholds.index:
        plt.axvline(x=thresholds[channel], linestyle='--', label=f'Channel {channel} Threshold', color=palette[channel])

    plt.title(f'{animal_id}: Density Plot of Mean Values per Channel')
    plt.xlabel('Mean Value')
    plt.ylabel('Density')
    plt.xlim(0, 80)
    plt.ylim(0, 0.15)

    # Print z_threshold on the plot
    plt.text(0.95, 0.95, f'Z-score: {z_threshold}', 
             horizontalalignment='right', 
             verticalalignment='top', 
             transform=plt.gca().transAxes, 
             fontsize=12)

    plt.legend()
    plt.show()
    return

def plot_kde_subplots(pooled_animal, animal_id, thresholds, z_threshold, save_path):
    # Define colors for the channels
    channel_colors = {0: 'blue', 1: 'green', 2: 'red', 3: 'cyan'}
    channels = [0, 1, 2, 3]

    fig, axs = plt.subplots(2,2, figsize=(15, 10))
    axs = axs.flatten()

    for i, channel in enumerate(channels):
        sns.kdeplot(data=pooled_animal[pooled_animal['Channel'] == channel], 
                    x='Mean', 
                    color=channel_colors[channel], 
                    ax=axs[i])
        
        axs[i].axvline(x=thresholds[channel], linestyle='--', color=channel_colors[channel])
        axs[i].set_title(f'Channel {channel}')
        #axs[i].set_xlim(0, 80)
        #axs[i].set_ylim(0, 0.15)
        axs[i].set_xlabel('Mean Value')
        axs[i].set_ylabel('Density')

    plt.suptitle(f'{animal_id}: Density Plot of Mean Values per Channel with Thresholds\nZ-score Threshold: {z_threshold}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    if save_path:
        save_kde = os.path.join(save_path, f'{animal_id}_kde_plots.png')
        plt.savefig(save_kde)
        print(f'Saved KDE plots to {save_kde}')
        
    plt.show()
    return
#%% * Categorize overlaps and plot density/threshold. ADD THIS TO POOL RESULTS FUNCTION ITS IMPORTANT?
#NECESSARY TO RUN FOR OVERLAPS RN

# Load the pooled results
results_dir = '/Users/emmaodom/Documents/ATLAS_dualSiteInj_VISp_MOp_0724/Results'
pooled_ROI_measurements = pd.read_csv(os.path.join(results_dir, 'pooled_roi_measurements.csv'))
save_kde = '/Users/emmaodom/Documents/ATLAS_dualSiteInj_VISp_MOp_0724/Results/visualizations/KDE_per_animal_w_threshold'
# Define Z-score threshold
z_threshold = 2 #can i make this a global variable to make mroe obvious to use everywhere

# Dictionary to store thresholds per animal
animal_thresholds = {}

# Get unique animalID (good bc this means normalization is both by animal and region)
unique_animals = pooled_ROI_measurements['AnimalID'].unique()
#initialize thresholds column
pooled_ROI_measurements['Overlap'] = 0
for animal in unique_animals:
    # Filter DataFrame for the current animal
    pooled_animal = pooled_ROI_measurements[pooled_ROI_measurements['AnimalID'] == animal]

    # Calculate the Z-score for the 'Mean' values for each channel
    pooled_animal['Zscore'] = pooled_animal.groupby('Channel')['Mean'].transform(lambda x: zscore(x))

    # Calculate thresholds
    mean = pooled_animal.groupby('Channel')['Mean'].mean()
    stdev = pooled_animal.groupby('Channel')['Mean'].std()
    threshold = z_threshold * stdev + mean

    # Store thresholds for the current animal
    animal_thresholds[animal] = threshold
    #plot kde with threshold per channel
    plot_kde_subplots(pooled_animal, animal, threshold, z_threshold,save_path = save_kde)
    # Add overlap column indicating whether each row is above the threshold
    #this is a rickety way to label each roi/row
    pooled_ROI_measurements.loc[pooled_ROI_measurements['AnimalID'] == animal, 'Overlap'] = pooled_animal.apply(
        lambda row: 1 if row['Mean'] > animal_thresholds[animal][row['Channel']] else 0, axis=1)
    #set all ch2 rois to 1 in overlap column
    pooled_ROI_measurements.loc[(pooled_ROI_measurements['AnimalID'] == animal) & (pooled_ROI_measurements['Channel'] == 2), 'Overlap'] = 1
    # Print thresholds for the current animal
    print(f"Thresholds for {animal}:")
    print(threshold)


# save the updated DataFrame to a new CSV file
pooled_ROI_measurements.to_csv(os.path.join(results_dir, 'pooled_roi_measurements.csv'), index=False)

#%% *** EXCLUDE bad data by replacing overlap with Nan. 
#style notes: .loc(criteria for which set of data to select, column to select within specified set)
#exclude PV ch0 for 88.2.VISp.ACA and 88.2.VISp.ORB
pooled_ROI_measurements.loc[(pooled_ROI_measurements['AnimalID'] == 'exp88.animal2.VISp.ORB') & (pooled_ROI_measurements['Channel'] == 0), 'Overlap'] = np.nan
pooled_ROI_measurements.loc[(pooled_ROI_measurements['AnimalID'] == 'exp88.animal2.VISp.ACA') & (pooled_ROI_measurements['Channel'] == 0), 'Overlap'] = np.nan
#exclude SST ch3 for 88.2.ACA and 88.2.ORB (for both VISp and MOp)
pooled_ROI_measurements.loc[(pooled_ROI_measurements['AnimalID'] == 'exp88.animal2.VISp.ORB') & (pooled_ROI_measurements['Channel'] == 3), 'Overlap'] = np.nan
pooled_ROI_measurements.loc[(pooled_ROI_measurements['AnimalID'] == 'exp88.animal2.MOp.ORB') & (pooled_ROI_measurements['Channel'] == 3), 'Overlap'] = np.nan
pooled_ROI_measurements.loc[(pooled_ROI_measurements['AnimalID'] == 'exp88.animal2.VISp.ACA') & (pooled_ROI_measurements['Channel'] == 3), 'Overlap'] = np.nan
pooled_ROI_measurements.loc[(pooled_ROI_measurements['AnimalID'] == 'exp88.animal2.MOp.ACA') & (pooled_ROI_measurements['Channel'] == 3), 'Overlap'] = np.nan
#save updated roi measurements csv
results_dir = '/Users/emmaodom/Documents/ATLAS_dualSiteInj_VISp_MOp_0724/Results'
pooled_ROI_measurements.to_csv(os.path.join(results_dir, 'pooled_roi_measurements.csv'), index=False) #save to make sure updates are included in future steps

#THIS IS A VERY LOGICAL PALCE TO END A MEASUREMENTS SCRIPT AND TRANSITION TO A SEPERATE VISUALIZATION SCRIPT 
#%% visualization: histogram
plt.figure()

pooled_ch0 = pooled_ROI_measurements[pooled_ROI_measurements['Channel']==0]
plt.hist(pooled_ch0['Mean'])
plt.figure()
sns.violinplot(data=pooled_ROI_measurements, x='Channel',y='Mean')

#%% visualziation: density plot of roi mean pixel value avg across channels, 
#create this per animal please and overlay vertical line at determined z-threshold. make it a function then iterate through animals
# Create a density plot for the mean values per channel
plt.figure(figsize=(10, 6))
sns.kdeplot(data=pooled_ROI_measurements, x='Mean', hue='Channel', common_norm=False)
plt.title('Density Plot of Mean Values per Channel')
plt.xlabel('Mean Value')
plt.ylabel('Density')
plt.xlim(0,80)
plt.ylim(0,0.15)
plt.show()

#%% visualization: swarm plot, could also make per animal
# Create a swarm plot for the mean values per channel
plt.figure(figsize=(10, 6))
sns.stripplot(data=pooled_ROI_measurements, x='Channel', y='Mean')
plt.title('Swarm Plot of Mean Values per Channel')
plt.xlabel('Channel')
plt.ylabel('Mean Value')
plt.show()

#%%visualization: number atlas cells per slice 
ch2_data = pooled_ROI_measurements[pooled_ROI_measurements['Channel']==2]
#group by animalID and slice to get pooled cell counts per slice
cell_counts = ch2_data.groupby(['AnimalID', 'Slice']).size().reset_index(name='Cell Count')
#bring in identifying features
cell_counts = cell_counts.merge(pooled_ROI_measurements[['AnimalID', 'Slice', 'Region', 'Origin']].drop_duplicates(), on=['AnimalID', 'Slice'], how='left')
cell_counts['Region_Origin'] = cell_counts['Region'] + ' - ' + cell_counts['Origin']

# Plot the results
plt.figure(figsize=(6, 4))
sns.swarmplot(x='Region', y='Cell Count', data=cell_counts)
plt.title('Number of ATLAS Cells per Slice')
plt.xlabel('Region')
plt.ylabel('Number of Cells')
plt.tight_layout()
plt.show()

# Plot the results
plt.figure(figsize=(6, 4))
sns.swarmplot(x='Origin', y='Cell Count', data=cell_counts)
plt.title('Number of ATLAS Cells per Slice')
plt.xlabel('Origin')
plt.ylabel('Number of Cells')
plt.tight_layout()
plt.show()

# Plot the results
plt.figure(figsize=(6, 4))
sns.swarmplot(x='Region_Origin', y='Cell Count', data=cell_counts)
plt.title('Number of ATLAS Cells per Slice')
plt.xlabel('Origin')
plt.ylabel('Number of Cells')
plt.tight_layout()
plt.show()

#%%visualization: number vip/sst/pv cells per slice 
overlap_data = pooled_ROI_measurements[pooled_ROI_measurements['Overlap'] == 1]
# Filter for Channels 0, 1, and 3 (because we set ch2 (self) overlap to always 1)
overlap_data = overlap_data[overlap_data['Channel'].isin([0, 1, 3])]
# Group by Channel, AnimalID, and Slice and count the number of overlap cells
cell_counts = overlap_data.groupby(['Channel', 'AnimalID', 'Slice']).size().reset_index(name='Cell Count')
# Copy the region information to the cell_counts DataFrame
cell_counts = cell_counts.merge(pooled_ROI_measurements[['AnimalID', 'Slice', 'Region']].drop_duplicates(), on=['AnimalID', 'Slice'], how='left')

# Plot the results
plt.figure(figsize=(6, 4))
sns.boxplot(x='Channel', y='Cell Count', hue='Region', data=cell_counts, dodge=True, showfliers=False)
sns.stripplot(x='Channel', y='Cell Count', hue='Region', data=cell_counts, dodge=True, marker='o', alpha=0.5, edgecolor='black', linewidth=0.6)
plt.title('Number of Positive Overlap Cells per Slice by Channel')
plt.xlabel('Channel')
plt.ylabel('Number of Cells')
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles[0:2], labels[0:2], title='Region', bbox_to_anchor=(1.05, 1), loc='upper left')
#plt.legend(title='Region')
plt.xticks(ticks=[0, 1, 2], labels=['PV', 'VIP', 'SST'])
plt.tight_layout()
plt.show()
#%%visualization: number vip/sst/pv cells per animal 
overlap_data = pooled_ROI_measurements[pooled_ROI_measurements['Overlap'] == 1]
# Filter for Channels 0, 1, and 3 (because we set ch2 (self) overlap to always 1)
overlap_data = overlap_data[overlap_data['Channel'].isin([0, 1, 3])]
# Group by Channel, AnimalID, and Slice and count the number of overlap cells
cell_counts = overlap_data.groupby(['Channel', 'AnimalID']).size().reset_index(name='Cell Count')
# Copy the region information to the cell_counts DataFrame
cell_counts = cell_counts.merge(pooled_ROI_measurements[['AnimalID', 'Region']].drop_duplicates(), on=['AnimalID'], how='left')
means = cell_counts.groupby(['Channel', 'Region'])['Cell Count'].mean()
# Plot the results
plt.figure(figsize=(6, 4))
sns.boxplot(x='Channel', y='Cell Count', hue='Region', data=cell_counts, dodge=True, showfliers=False)
sns.stripplot(x='Channel', y='Cell Count', hue='Region', data=cell_counts, dodge=True, marker='o', alpha=0.5, edgecolor='black', linewidth=0.6)
#display mean on plot
for i, mean in means.items():
    print(i[1])
    print(mean)
    #plt.text(i, mean + 0.5, f'{mean:.2f}', ha='center', va='bottom', color='black', fontsize=12)
plt.title('Number of Positive Overlap Cells per Animal by Channel')
plt.ylabel('Number of Cells')
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles[0:2], labels[0:2], title='Region', bbox_to_anchor=(1.05, 1), loc='upper left')
#plt.legend(title='Region')
plt.xticks(ticks=[0, 1, 2], labels=['PV', 'VIP', 'SST'])
plt.tight_layout()
plt.show()

#%% * visualization fraction overlap by channel+region ,lazy copied from chatgpt

# Assuming pooled_ROI_measurements is your DataFrame
def plot_fraction_of_overlap_cells(pooled_ROI_measurements):
    #to ensure fraction is calculated correctly, drop rows with Nan
    pooled_ROI_measurements = pooled_ROI_measurements.dropna(subset=['Overlap'],axis=0)
    # Calculate total cells per channel per animal
    total_cells = pooled_ROI_measurements.groupby(['Channel', 'AnimalID']).size().reset_index(name='Total Cells')
    total_cells = total_cells[total_cells['Channel'].isin([0, 1, 3])]
    # Calculate overlap cells per channel per animal
    overlap_cells = pooled_ROI_measurements[pooled_ROI_measurements['Overlap'] == 1]
    overlap_counts = overlap_cells.groupby(['Channel', 'AnimalID']).size().reset_index(name='Overlap Cells')

    # Merge total cells and overlap cells
    cell_counts = total_cells.merge(overlap_counts, on=['Channel', 'AnimalID'], how='left')
    cell_counts['Overlap Cells'] = cell_counts['Overlap Cells'].fillna(0)

    # Calculate the fraction of overlap cells
    cell_counts['Fraction Overlap'] = cell_counts['Overlap Cells'] / cell_counts['Total Cells']

    # Copy the region information to the cell_counts DataFrame
    cell_counts = cell_counts.merge(pooled_ROI_measurements[['AnimalID', 'Region']].drop_duplicates(), on='AnimalID', how='left')

    # Plot the results
    plt.figure(figsize=(15, 8))
    sns.boxplot(x='Channel', y='Fraction Overlap', hue='Region', data=cell_counts, dodge=True, showmeans=True, meanprops={
        "marker": "o", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": "10"
    })
    sns.swarmplot(x='Channel', y='Fraction Overlap', hue='Region', data=cell_counts, dodge=True, marker='o', alpha=0.5, palette='dark:#5A9_r', edgecolor='gray', linewidth=0.6)

    # Set custom labels for the x-axis ticks
    plt.xticks(ticks=[0, 1, 2], labels=['PV', 'VIP', 'SST'])

    # Annotate means on the plot
    means = cell_counts.groupby('Channel')['Fraction Overlap'].mean() #groupby(['Channel', 'Region']) if you want MOp VISp split. 
    for i, mean in means.items():
        if(i==3):
            i-=1
        plt.text(i, mean + 0.02, f'{mean:.2f}', ha='center', va='bottom', color='black', fontsize=12)

    plt.title('Fraction of Overlap Cells per Channel per Animal')
    plt.xlabel('Channel')
    plt.ylabel('Fraction of Overlap Cells')
    
    # Remove duplicate legends
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles[:2], labels[:2], title='Region', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.show()
    return cell_counts

#implement. can use cell counts df to verify that the correct data is included/contributing to fraction overlap
cell_counts = plot_fraction_of_overlap_cells(pooled_ROI_measurements)

#%% old. initial test of threshold
pooled_ch0 = pooled_ROI_measurements[pooled_ROI_measurements['Channel']==0]
tentative_rois = pooled_ch0[pooled_ch0['Mean']>20]

pooled_ch3 = pooled_ROI_measurements[pooled_ROI_measurements['Channel']==3]
tentative_rois3 = pooled_ch3[pooled_ch3['Mean']>35]
#%% old. IDENTIFY BY Zscore critical step to include in df

# Load the pooled results
AnimalID = 'exp89.animal2.MOp.ORB'
pooled_animal = pooled_ROI_measurements[pooled_ROI_measurements['AnimalID']==AnimalID]

# Calculate the Z-score for the 'Mean' values for each channel
pooled_animal['Mean_Zscore'] = pooled_animal.groupby('Channel')['Mean'].transform(lambda x: zscore(x))

# Filter for values with Z-score > 2 (or another threshold)
z_threshold = 2.5
outliers = pooled_animal[pooled_animal['Mean_Zscore'] > z_threshold] #not.abs() bc we only want values 2 stdev above mean

# Display the outliers
print(outliers)

#get threshold 
mean = pooled_animal.groupby('Channel')['Mean'].mean()
stdev = pooled_animal.groupby('Channel')['Mean'].std()
threshold = z_threshold*stdev + mean
print(threshold)

# Create a swarm plot for the outliers
plt.figure(figsize=(10, 6))
sns.swarmplot(data=outliers, x='Channel', y='Mean')
plt.title(f'{AnimalID}: Swarm Plot of Outlier Mean Values (Z-score > {z_threshold})')
plt.xlabel('Channel')
plt.ylabel('Mean Value')
plt.show()

#%% old. one option for applying threshold based on z-score
#should just replace this with a label column as positive overlap or not for each roi/row
pooled_ch0 = pooled_ROI_measurements[pooled_ROI_measurements['Channel']==0]
tentative_rois0 = pooled_ch0[pooled_ch0['Mean']>threshold[0]]

pooled_ch1 = pooled_ROI_measurements[pooled_ROI_measurements['Channel']==1]
tentative_rois1 = pooled_ch1[pooled_ch1['Mean']>threshold[1]]

pooled_ch3 = pooled_ROI_measurements[pooled_ROI_measurements['Channel']==3]
tentative_rois3 = pooled_ch3[pooled_ch3['Mean']>threshold[3]]
    
#%% * summarize overlaps 

# Filter the DataFrame for relevant channels and overlaps
ch013 = pooled_ROI_measurements[pooled_ROI_measurements['Channel'].isin([0, 1, 3])]
# Count the number of overlaps per channel and region
overlap_counts = ch013[ch013['Overlap'] == 1].groupby(['Channel', 'Region']).size().unstack(fill_value=0)

# Create a bar plot
plt.figure(figsize=(10, 6))
overlap_counts.plot(kind='bar', color=['blue', 'green', 'red'], edgecolor='black')

# Add titles and labels
plt.title('Number of Overlaps per Channel')
plt.xlabel('Channel')
plt.ylabel('Number of Overlaps')
plt.xticks(ticks=[0, 1, 2], labels=['PV', 'VIP', 'SST'], rotation=0)

# Show the plot
plt.tight_layout()
plt.show()

#%%

# Extract the region (MOp or VISp) from the AnimalID and create a new column, case insensitive
#pooled_ROI_measurements['Region'] = pooled_ROI_measurements['AnimalID'].apply(lambda x: 'MOp' if 'mop' in x.lower() else 'VISp')

# filter for Channels 0, 1, and 3 (because we set ch2 (self) overlap to always 1)
ch013 = pooled_ROI_measurements[pooled_ROI_measurements['Channel'].isin([0, 1, 3])]
overlaps = ch013[ch013['Overlap'] == 1]

# Count the number of overlaps per channel and region
overlap_counts = overlaps.groupby(['Channel', 'Region']).size().reset_index(name='Count')

# Plot the results using seaborn
plt.figure(figsize=(12, 8))
sns.barplot(data=overlap_counts, x='Channel', y='Count', hue='Region', palette=['#1f77b4', '#ff7f0e'])  # Blue for MOp, Orange for VISp

# Add titles and labels
plt.title('Number of Overlaps per Channel by Region (MOp and VISp)')
plt.xlabel('Channel')
plt.ylabel('Number of Overlaps')
plt.xticks(ticks=[0, 1, 2], labels=['Channel 0', 'Channel 1', 'Channel 3'])

# Show the plot
plt.tight_layout()
plt.show()

#%% * define overlay/overlap function 
#should i draw all rois and star the overlap ones? to see what im missing?
def plot_overlay_overlap_ROIs(parent_tiff_path,animalID,slice_number,group_df, z_threshold,dpi,save_path):
    ''' customized to our imaging scheme with 4 channels and vip/sst/pv/atlas'''
    # Loop through channels ch0, ch1, ch2, ch3 for the given animal_slice
    channels = [0, 1, 2, 3]
    all_channels_loaded = True
    for ch in channels:
        # Construct the correct TIFF path for each channel
        tiff_name = f"{animalID}_S{slice_number}_ch{ch}.tif"
        tiff_path = os.path.join(parent_tiff_path,animalID, tiff_name)
        if not os.path.exists(tiff_path):
            print(f"TIFF file not found: {tiff_path}")
            all_channels_loaded = False
            continue
        
        # Read the TIFF image
        temp_image = imread(tiff_path)
        temp_max_proj = np.max(temp_image, axis=0)
        #determine resolution and initialize stack
        if (ch==0):
            x,y = temp_max_proj.shape
            #create empty array to store 4 channel in one image stack
            img_stack = np.zeros((x,y,4))
            
        ##ALARMS this is one place where the despeckle consistency matters
        if (ch==1): #despeckle vip (larger kernel bc more + more intense puncta)
            temp_max_proj = cv2.medianBlur(temp_max_proj, 5)
        if (ch==3): #despeckle sst (smaller kernel bc less nosiy) 
            temp_max_proj = cv2.medianBlur(temp_max_proj, 3)
        if (ch!=2):
            temp_rescale = exposure.rescale_intensity(temp_max_proj, in_range=(2, 98))
            temp_max_proj = temp_rescale
        #assign to channel
        img_stack[:,:,ch] = temp_max_proj
    
    if all_channels_loaded:
        fig, axs = plt.subplots(1, 3, figsize=(10.24 * 3, 10.24),dpi=dpi)
        # Overlay ch0 (blue) and ch2 (red)
        overlay_image_0_2 = np.zeros((1024, 1024, 3), dtype=np.float32)
        overlay_image_0_2[:, :, 2] = img_stack[:, :, 0]  # Blue channel (ch0)
        overlay_image_0_2[:, :, 0] = img_stack[:, :, 2]  # Red channel (ch2)
        overlay_image_0_2 = overlay_image_0_2 / overlay_image_0_2.max()
        #plot
        axs[0].imshow(overlay_image_0_2)
        axs[0].set_title(f'{animalID} Slice {slice_number}: Overlay of Channel 0 (Blue) and Channel 2 (Red)')
        axs[0].axis('off')
        #subindex df for ch0 rois that have overlap
        ch0_overlap = group[(group['Channel']==0) & (group['Overlap']==1)]
        for _, row in ch0_overlap.iterrows():
            center_x = row['X']
            center_y = row['Y']
            width = row['Width']
            height = row['Height']
            roi_oval = Ellipse((center_x, center_y), width, height, edgecolor='white', facecolor='none', linewidth=0.8)
            axs[0].add_patch(roi_oval)
            
        # Overlay ch1 (green) and ch2 (red)
        overlay_image_1_2 = np.zeros((1024, 1024, 3), dtype=np.float32)
        overlay_image_1_2[:, :, 1] = img_stack[:, :, 1]  # Green channel (ch1)
        overlay_image_1_2[:, :, 0] = img_stack[:, :, 2]  # Red channel (ch2)
        overlay_image_1_2 = overlay_image_1_2 / overlay_image_1_2.max()
        #plot
        axs[1].imshow(overlay_image_1_2)
        axs[1].set_title(f'{animalID} Slice {slice_number}: Overlay of Channel 1 (Green) and Channel 2 (Red)')
        axs[1].axis('off')
        #subindex df for ch0 rois that have overlap
        ch1_overlap = group[(group['Channel']==1) & (group['Overlap']==1)]
        for _, row in ch1_overlap.iterrows():
            center_x = row['X']
            center_y = row['Y']
            width = row['Width']
            height = row['Height']
            roi_oval = Ellipse((center_x, center_y), width, height, edgecolor='white', facecolor='none', linewidth=0.8)
            axs[1].add_patch(roi_oval)
            
        # Overlay ch3 (violet) and ch2 (red)
        overlay_image_3_2 = np.zeros((1024, 1024, 3), dtype=np.float32)
        overlay_image_3_2[:, :, 0] = img_stack[:, :, 2]  # Red channel (ch2)
        overlay_image_3_2[:, :, 1] = img_stack[:, :, 3]  # Blue channel (ch3)
        overlay_image_3_2[:, :, 2] = img_stack[:, :, 3]  # Blue channel (ch3)
        overlay_image_3_2 = overlay_image_3_2 / overlay_image_3_2.max()
        #plot
        axs[2].imshow(overlay_image_3_2)
        axs[2].set_title(f'{animalID} Slice {slice_number}: Overlay of Channel 3 (Violet) and Channel 2 (Red)')
        axs[2].axis('off')
        #subindex df for ch3 rois that have overlap
        ch3_overlap = group_df[(group_df['Channel']==3) & (group_df['Overlap']==1)]
        for _, row in ch3_overlap.iterrows():
            center_x = row['X']
            center_y = row['Y']
            width = row['Width']
            height = row['Height']
            roi_oval = Ellipse((center_x, center_y), width, height, edgecolor='white', facecolor='none', linewidth=0.8)
            axs[2].add_patch(roi_oval)
        plt.figtext(0.5, 0.005, f'Z-score Threshold: {z_threshold}', ha='center', fontsize=12)
        plt.tight_layout()
        plt.show()
    else:
        print(f"Not all required channels were found for AnimalID: {animalID}, Slice: {slice_number}") 
    if save_path:
        save_overlay = os.path.join(save_path, f'{animalID}_S{slice_number}_overlay_z{z_threshold}.png')
        fig.savefig(save_overlay,bbox_inches='tight')
        print(f'Saved overlay plots to {save_overlay}')
    return

def plot_overlay_overlap_allROIs(parent_tiff_path,animalID,slice_number,group_df, z_threshold,dpi,save_path):
    ''' customized to our imaging scheme with 4 channels and vip/sst/pv/atlas'''
    # Loop through channels ch0, ch1, ch2, ch3 for the given animal_slice
    channels = [0, 1, 2, 3]
    all_channels_loaded = True
    for ch in channels:
        # Construct the correct TIFF path for each channel
        tiff_name = f"{animalID}_S{slice_number}_ch{ch}.tif"
        tiff_path = os.path.join(parent_tiff_path,animalID, tiff_name)
        if not os.path.exists(tiff_path):
            print(f"TIFF file not found: {tiff_path}")
            all_channels_loaded = False
            continue
        
        # Read the TIFF image
        temp_image = imread(tiff_path)
        temp_max_proj = np.max(temp_image, axis=0)
        #determine resolution and initialize stack
        if (ch==0):
            x,y = temp_max_proj.shape
            #create empty array to store 4 channel in one image stack
            img_stack = np.zeros((x,y,4))
            
        ##ALARMS this is one place where the despeckle consistency matters
        if (ch==1): #despeckle vip (larger kernel bc more + more intense puncta)
            temp_max_proj = cv2.medianBlur(temp_max_proj, 5)
        if (ch==3): #despeckle sst (smaller kernel bc less nosiy) 
            temp_max_proj = cv2.medianBlur(temp_max_proj, 3)
        if (ch!=2):
            temp_rescale = exposure.rescale_intensity(temp_max_proj, in_range=(2, 98))
            temp_max_proj = temp_rescale
        #assign to channel
        img_stack[:,:,ch] = temp_max_proj
    
    if all_channels_loaded:
        fig, axs = plt.subplots(1, 3, figsize=(10.24 * 3, 10.24),dpi=dpi)
        # Overlay ch0 (blue) and ch2 (red)
        overlay_image_0_2 = np.zeros((1024, 1024, 3), dtype=np.float32)
        overlay_image_0_2[:, :, 2] = img_stack[:, :, 0]  # Blue channel (ch0)
        overlay_image_0_2[:, :, 0] = img_stack[:, :, 2]  # Red channel (ch2)
        overlay_image_0_2 = overlay_image_0_2 / overlay_image_0_2.max()
        #plot
        axs[0].imshow(overlay_image_0_2)
        axs[0].set_title(f'{animalID} Slice {slice_number}: Overlay of Channel 0 (Blue) and Channel 2 (Red)')
        axs[0].axis('off')
        
        # Plot all ROIs and add star for overlaps
        for _, row in group_df[group_df['Channel'] == 0].iterrows():
            center_x = row['X']
            center_y = row['Y']
            width = row['Width']
            height = row['Height']
            roi_oval = Ellipse((center_x, center_y), width, height, edgecolor='white', facecolor='none', linewidth=0.8)
            if row['Overlap'] == 1:
                roi_oval = Ellipse((center_x, center_y), width, height, edgecolor='yellow', facecolor='none', linewidth=0.8)
            axs[0].add_patch(roi_oval)
        
        # Overlay ch1 (green) and ch2 (red)
        overlay_image_1_2 = np.zeros((1024, 1024, 3), dtype=np.float32)
        overlay_image_1_2[:, :, 1] = img_stack[:, :, 1]  # Green channel (ch1)
        overlay_image_1_2[:, :, 0] = img_stack[:, :, 2]  # Red channel (ch2)
        overlay_image_1_2 = overlay_image_1_2 / overlay_image_1_2.max()
        #plot
        axs[1].imshow(overlay_image_1_2)
        axs[1].set_title(f'{animalID} Slice {slice_number}: Overlay of Channel 1 (Green) and Channel 2 (Red)')
        axs[1].axis('off')
        # Plot all ROIs and add star for overlaps
        for _, row in group_df[group_df['Channel'] == 1].iterrows():
            center_x = row['X']
            center_y = row['Y']
            width = row['Width']
            height = row['Height']
            roi_oval = Ellipse((center_x, center_y), width, height, edgecolor='white', facecolor='none', linewidth=0.8)
            if row['Overlap'] == 1:
                roi_oval = Ellipse((center_x, center_y), width, height, edgecolor='yellow', facecolor='none', linewidth=0.8)
            axs[1].add_patch(roi_oval)
            
        # Overlay ch3 (violet) and ch2 (red)
        overlay_image_3_2 = np.zeros((1024, 1024, 3), dtype=np.float32)
        overlay_image_3_2[:, :, 0] = img_stack[:, :, 2]  # Red channel (ch2)
        overlay_image_3_2[:, :, 1] = img_stack[:, :, 3]  # Blue channel (ch3)
        overlay_image_3_2[:, :, 2] = img_stack[:, :, 3]  # Blue channel (ch3)
        overlay_image_3_2 = overlay_image_3_2 / overlay_image_3_2.max()
        #plot
        axs[2].imshow(overlay_image_3_2)
        axs[2].set_title(f'{animalID} Slice {slice_number}: Overlay of Channel 3 (Violet) and Channel 2 (Red)')
        axs[2].axis('off')
        # Plot all ROIs and add star for overlaps
        for _, row in group_df[group_df['Channel'] == 3].iterrows():
            center_x = row['X']
            center_y = row['Y']
            width = row['Width']
            height = row['Height']
            roi_oval = Ellipse((center_x, center_y), width, height, edgecolor='white', facecolor='none', linewidth=0.8)
            if row['Overlap'] == 1:
                roi_oval = Ellipse((center_x, center_y), width, height, edgecolor='yellow', facecolor='none', linewidth=0.8)
            axs[2].add_patch(roi_oval)
                
        plt.figtext(0.5, 0.005, f'Z-score Threshold: {z_threshold}', ha='center', fontsize=12)
        plt.tight_layout()
        plt.show()
    else:
        print(f"Not all required channels were found for AnimalID: {animalID}, Slice: {slice_number}") 
    if save_path:
        save_overlay = os.path.join(save_path, f'{animalID}_S{slice_number}_overlay_z{z_threshold}.png')
        fig.savefig(save_overlay,bbox_inches='tight')
        print(f'Saved overlay plots to {save_overlay}')
    return

#simplified version of above
def plot_overlay_overlap_allROIs(parent_tiff_path, animalID, slice_number, group_df, z_threshold, dpi, save_path=None):
    '''Customized to our imaging scheme with 4 channels and vip/sst/pv/atlas'''
    
    channels = [0, 1, 2, 3]
    all_channels_loaded = True
    img_stack = None

    for ch in channels:
        # Construct the correct TIFF path for each channel
        tiff_name = f"{animalID}_S{slice_number}_ch{ch}.tif"
        tiff_path = os.path.join(parent_tiff_path, animalID, tiff_name)
        
        if not os.path.exists(tiff_path):
            print(f"TIFF file not found: {tiff_path}")
            all_channels_loaded = False
            break
        
        # Read the TIFF image
        temp_image = imread(tiff_path)
        temp_max_proj = np.max(temp_image, axis=0)
        
        if img_stack is None:
            x, y = temp_max_proj.shape
            img_stack = np.zeros((x, y, 4), dtype=np.float32)
        
        if ch == 1:
            temp_max_proj = cv2.medianBlur(temp_max_proj, 5)
        elif ch == 3:
            temp_max_proj = cv2.medianBlur(temp_max_proj, 3)
        
        if ch != 2:
            temp_max_proj = exposure.rescale_intensity(temp_max_proj, in_range=(0.5, 99.5))
        
        img_stack[:, :, ch] = temp_max_proj

    if all_channels_loaded:
        fig, axs = plt.subplots(1, 3, figsize=(10.24 * 3, 10.24), dpi=dpi)

        overlay_configs = [
            (0, 'Blue', axs[0], '0 (Blue)'),
            (1, 'Green', axs[1], '1 (Green)'),
            (3, 'Violet', axs[2], '3 (Violet)')
        ]

        for ch, color_name, ax, title_suffix in overlay_configs:
            overlay_image = np.zeros((x, y, 3), dtype=np.float32)
            overlay_image[:, :, {'Blue': 2, 'Green': 1, 'Violet': 1}[color_name]] = img_stack[:, :, ch]
            overlay_image[:, :, {'Blue': 2, 'Green': 1, 'Violet': 2}[color_name]] = img_stack[:, :, ch]
            overlay_image[:, :, 0] = img_stack[:, :, 2]
            overlay_image = overlay_image / overlay_image.max()
            ax.imshow(overlay_image)
            ax.set_title(f'{animalID} Slice {slice_number}: Overlay of Channel {title_suffix} and Channel 2 (Red)')
            ax.axis('off')

            for _, row in group_df[group_df['Channel'] == ch].iterrows():
                center_x = row['X']
                center_y = row['Y']
                width = row['Width']
                height = row['Height']
                roi_oval = Ellipse((center_x, center_y), width, height, edgecolor='yellow' if row['Overlap'] == 1 else 'white', facecolor='none', linewidth=0.8)
                ax.add_patch(roi_oval)
                if row['Overlap'] == 1:
                    ax.plot(center_x + width / 2 + 2, center_y + height / 2 + 2, '*', color='yellow', markersize=1)

        fig.text(0.5, 0.005, f'Z-score Threshold: {z_threshold}', ha='center', fontsize=12)
        plt.tight_layout()
        plt.show()

        if save_path:
            save_overlay = os.path.join(save_path, f'{animalID}_S{slice_number}_overlay_z{z_threshold}.png')
            fig.savefig(save_overlay, bbox_inches='tight')
            print(f'Saved overlay plots to {save_overlay}')
        return fig
    else:
        print(f"Not all required channels were found for AnimalID: {animalID}, Slice: {slice_number}")
        return None

#%% * CREATES OVERLAY IAMGES WITH OVERLAP ROIS LABELED
#ugh i need to plot all slices for the new animals bc the overlaps arent showing in subsample. 
parent_tiff_path = '/Users/emmaodom/Documents/ATLAS_dualSiteInj_VISp_MOp_0724/TIFFS'
save_path = '/Users/emmaodom/Documents/ATLAS_dualSiteInj_VISp_MOp_0724/Results/visualizations/Overlay_ROI_Overlap'
grouped_anID_S = pooled_ROI_measurements.groupby(['AnimalID','Slice'])
#dont redefine z_threshold, use last saved. 
dpi = 300
test = 0
for (animalID, slice_number), group_df in grouped_anID_S:
    #print(animal_id, slice_number)
    if slice_number%3==0: #subsample every 3rd slice bc we wont look at each anyways
        plot_overlay_overlap_allROIs(parent_tiff_path,animalID,slice_number,group_df, z_threshold, dpi, save_path)
    #test+=1
    #if test>=3:
        #break
#%% original create overlay pre-function, can probably delete
#should i draw all rois and star the overlap ones? to see what im missing?
parent_TIFFS = '/Users/emmaodom/Documents/ATLAS_dualSiteInj_VISp_MOp_0724/TIFFS'
save_path = '/Users/emmaodom/Documents/ATLAS_dualSiteInj_VISp_MOp_0724/Results/visualizations/Overlay_ROI_Overlap'
#get tiff paths to ^ construct an rbg stack
grouped_anID_S = pooled_ROI_measurements.groupby(['AnimalID','Slice'])
test=0
for (animalID, slice_number), group in grouped_anID_S:
    #print(animal_id, slice_number)
    # Loop through channels ch0, ch1, ch2, ch3 for the given animal_slice
    channels = [0, 1, 2, 3]
    #create empty array to store 4 channel in one image stack
    img_stack = np.zeros((1024,1024,4)) #better if pixel_num/resolution is determined from sample image
    all_channels_loaded = True
    for ch in channels:
        # Construct the correct TIFF path for each channel
        tiff_name = f"{animalID}_S{slice_number}_ch{ch}.tif"
        tiff_path = os.path.join(parent_TIFFS,animalID, tiff_name)
        if not os.path.exists(tiff_path):
            print(f"TIFF file not found: {tiff_path}")
            all_channels_loaded = False
            continue
        
        # Read the TIFF image
        temp_image = imread(tiff_path)
        temp_max_proj = np.max(temp_image, axis=0)
        ##ALARMS this is one place where the despeckle consistency matters
        if (ch==1): #despeckle vip (larger kernel bc more + more intense puncta)
            temp_max_proj = cv2.medianBlur(temp_max_proj, 5)
        if (ch==3): #despeckle sst (smaller kernel bc less nosiy) 
            temp_max_proj = cv2.medianBlur(temp_max_proj, 3)
        if (ch!=2):
            temp_rescale = exposure.rescale_intensity(temp_max_proj, in_range=(2, 98))
            temp_max_proj = temp_rescale
        #assign to channel
        img_stack[:,:,ch] = temp_max_proj
    
    if all_channels_loaded:
        fig, axs = plt.subplots(1, 3, figsize=(10.24 * 3, 10.24),dpi=300)
        # Overlay ch0 (blue) and ch2 (red)
        overlay_image_0_2 = np.zeros((1024, 1024, 3), dtype=np.float32)
        overlay_image_0_2[:, :, 2] = img_stack[:, :, 0]  # Blue channel (ch0)
        overlay_image_0_2[:, :, 0] = img_stack[:, :, 2]  # Red channel (ch2)
        overlay_image_0_2 = overlay_image_0_2 / overlay_image_0_2.max()
        #plot
        axs[0].imshow(overlay_image_0_2)
        axs[0].set_title(f'{animalID} Slice {slice_number}: Overlay of Channel 0 (Blue) and Channel 2 (Red)')
        axs[0].axis('off')
        #subindex df for ch0 rois that have overlap
        ch0_overlap = group[(group['Channel']==0) & (group['Overlap']==1)]
        for _, row in ch0_overlap.iterrows():
            center_x = row['X']
            center_y = row['Y']
            width = row['Width']
            height = row['Height']
            roi_oval = Ellipse((center_x, center_y), width, height, edgecolor='white', facecolor='none', linewidth=0.8)
            axs[0].add_patch(roi_oval)
            
        # Overlay ch1 (green) and ch2 (red)
        overlay_image_1_2 = np.zeros((1024, 1024, 3), dtype=np.float32)
        overlay_image_1_2[:, :, 1] = img_stack[:, :, 1]  # Green channel (ch1)
        overlay_image_1_2[:, :, 0] = img_stack[:, :, 2]  # Red channel (ch2)
        overlay_image_1_2 = overlay_image_1_2 / overlay_image_1_2.max()
        #plot
        axs[1].imshow(overlay_image_1_2)
        axs[1].set_title(f'{animalID} Slice {slice_number}: Overlay of Channel 1 (Green) and Channel 2 (Red)')
        axs[1].axis('off')
        #subindex df for ch0 rois that have overlap
        ch1_overlap = group[(group['Channel']==1) & (group['Overlap']==1)]
        for _, row in ch1_overlap.iterrows():
            center_x = row['X']
            center_y = row['Y']
            width = row['Width']
            height = row['Height']
            roi_oval = Ellipse((center_x, center_y), width, height, edgecolor='white', facecolor='none', linewidth=0.8)
            axs[1].add_patch(roi_oval)
            
        # Overlay ch3 (violet) and ch2 (red)
        overlay_image_3_2 = np.zeros((1024, 1024, 3), dtype=np.float32)
        overlay_image_3_2[:, :, 0] = img_stack[:, :, 2]  # Red channel (ch2)
        overlay_image_3_2[:, :, 1] = img_stack[:, :, 3]  # Blue channel (ch3)
        overlay_image_3_2[:, :, 2] = img_stack[:, :, 3]  # Blue channel (ch3)
        overlay_image_3_2 = overlay_image_3_2 / overlay_image_3_2.max()
        #plot
        axs[2].imshow(overlay_image_3_2)
        axs[2].set_title(f'{animalID} Slice {slice_number}: Overlay of Channel 3 (Violet) and Channel 2 (Red)')
        axs[2].axis('off')
        #subindex df for ch3 rois that have overlap
        ch3_overlap = group[(group['Channel']==3) & (group['Overlap']==1)]
        for _, row in ch3_overlap.iterrows():
            center_x = row['X']
            center_y = row['Y']
            width = row['Width']
            height = row['Height']
            roi_oval = Ellipse((center_x, center_y), width, height, edgecolor='white', facecolor='none', linewidth=0.8)
            axs[2].add_patch(roi_oval)
        plt.tight_layout()
        plt.show()
    else:
        print(f"Not all required channels were found for AnimalID: {animalID}, Slice: {slice_number}") 
    test+=1
    if test>3:
        break

#%% despeckle
def make_green(image):
    # Create a green version of the image
    green_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=image.dtype)
    green_image[:, :, 1] = image  # Set the green channel to the image
    return green_image

def make_cyan(image):
    # Create a green version of the image
    cyan_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=image.dtype)
    cyan_image[:, :, 1] = image  # Set the green channel to the image
    cyan_image[:, :, 2] = image
    return cyan_image


def despeckle_and_show(tiff_path,animalID,slice_number,ch,kernel=3,save_path=None):
    '''
    probably best to set kernel = 3 for SST ch3
    and kernel = 5 for VIP ch 1
    '''
    image = imread(tiff_path)
    temp_max_proj = np.max(image, axis=0)
    # Apply a median filter to remove speckles
    median_filtered = cv2.medianBlur(temp_max_proj, kernel)
    #make green!
    green_original = make_green(temp_max_proj)
    green_filtered = make_green(median_filtered)
    #adjust contrast
    green_original = exposure.rescale_intensity(green_original, in_range=(0.5, 99.5))
    green_filtered = exposure.rescale_intensity(green_filtered, in_range=(0.5, 99.5))
    # Display the original and processed images
    fig, axs = plt.subplots(1, 2, figsize=(20, 10),dpi=100)
    axs[0].imshow(green_original)
    axs[0].set_title('Original Image')
    axs[0].axis('off')

    axs[1].imshow(green_filtered)
    axs[1].set_title(f'Despeckled Image k:{kernel}')
    axs[1].axis('off')
    
    #print animalID ans slice number somewhere?
    plt.tight_layout()
    plt.show()
    
    # Save the despeckled image if save_path is provided
    if save_path:
        save_despeckle= os.path.join(save_path, f'{animalID}_S{slice_number}_despeckled_ch{ch}.png')
        cv2.imwrite(save_despeckle, cv2.cvtColor(green_image, cv2.COLOR_RGB2BGR))
        print(f'Saved despeckled image to {output_path}')
    
tiff_path = '/Users/emmaodom/Documents/ATLAS_dualSiteInj_VISp_MOp_0724/TIFFS/exp89.animal2.MOp.ACA/exp89.animal2.MOp.ACA_S6_ch1.tif' 
despeckle_and_show(tiff_path,'exp88.animal1.MOp.ORB',1,1,kernel=5)
#get list of files with ch1 and ch3 in the name, get anmalid and slice number, apply kernel accordingly (ch1=k5, ch3=k3)

#%% other example to look at 
pattern = re.compile(r"S(\d+)_z(\d+)_ch(\d+).tif")
label_pattern = re.compile(r'([^_]+)')
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
#%% * def pool surface results
def pool_surface_results(input_dir,save_path,pattern):
    '''update to save csv
    This function pools results.csv/measurement information across the entire dataset!
    '''
    all_results = pd.DataFrame()
    # Traverse the directory and find all CSV files
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith("_results.csv"):
                # Read the CSV and concatenate to the DataFrame
                csv_path = os.path.join(root, file)
                df = pd.read_csv(csv_path)
                #get animalID and slice
                match = pattern.search(file)
                if match:
                    animalID=match.group(1)
                    slice_number=match.group(2)
                    df['AnimalID'] = animalID
                    df['Slice'] = slice_number
                all_results = pd.concat([all_results, df], ignore_index=True)
    all_results['Region'] = all_results['AnimalID'].apply(lambda x: 'MOp' if 'mop' in x.lower() else 'VISp')
    all_results['Origin'] = all_results['AnimalID'].apply(lambda x: 'ACA' if 'aca' in x.lower() else 'ORB')
    all_results['Slice'] = all_results['Slice'].astype(int)
    output_csv_path = os.path.join(save_path, "pooled_surface_points.csv")
    all_results.to_csv(output_csv_path, index=False)
    print(f"Saved results to {output_csv_path}")
    return all_results

#%% * pool brain surface measurements
input_dir = '/Users/emmaodom/Documents/ATLAS_dualSiteInj_VISp_MOp_0724/Results/surface_points'
save_path = '/Users/emmaodom/Documents/ATLAS_dualSiteInj_VISp_MOp_0724/Results'
pattern = re.compile(r'(exp\d+\.\w+\.\w+\.\w+)_S(\d+)_ch2')
pooled_surface_points = pool_surface_results(input_dir,save_path,pattern)
#%% * MAY not be necessary, since matching uo to roi_sumamry_df will be by animalID. assign correct labels! make sure to use the origin column for future splits.
pooled_surface_points.loc[pooled_surface_points['AnimalID'] == 'exp88.animal1.MOp.ORB', 'Origin'] = 'ACA'
pooled_surface_points.loc[pooled_surface_points['AnimalID'] == 'exp88.animal1.VISp.ORB', 'Origin'] = 'ACA'
pooled_surface_points.loc[pooled_surface_points['AnimalID'] == 'exp88.animal2.MOp.ACA', 'Origin'] = 'ORB'
pooled_surface_points.loc[pooled_surface_points['AnimalID'] == 'exp88.animal2.VISp.ACA', 'Origin'] = 'ORB'
results_dir = '/Users/emmaodom/Documents/ATLAS_dualSiteInj_VISp_MOp_0724/Results'
pooled_surface_points.to_csv(os.path.join(save_path, 'pooled_surface_points.csv'), index=False) #save to make sure updates are included in future steps
#%% * def functions for depth calculation 

def linear_regression(x, y):
    m, b, r_value, p_value, std_err = stats.linregress(x, y)
    return m, b

def find_theta(m):
    #theta = np.arctan2(y[-1] - y[0], x[-1] - x[0])
    #for deltaX = 1, deltaY = m 
    theta = np.arctan2(m, 1)*180/np.pi 
    return theta

def rotate_points(x, y, n):
    theta = np.radians(n) 
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c,-s), (s,c)))
    XY_R = np.dot(R, np.array([x,y]))
    xR = XY_R[0,:]
    yR = XY_R[1,:]
    return xR, yR

def calculate_perpendicular_distance(pchip, roi_x, roi_y):
    # Find the nearest x on the brain surface
    nearest_x = x_interp[np.abs(x_interp - roi_x).argmin()]
    
    # Calculate the slope of the tangent at nearest_x
    tangent_slope = pchip.derivative()(nearest_x)
    
    # Calculate the slope of the perpendicular line
    perpendicular_slope = -1 / tangent_slope
    
    # Calculate the y-intercept of the perpendicular line
    brain_surface_y_at_nearest_x = pchip(nearest_x)
    y_intercept = roi_y - perpendicular_slope * roi_x
    
    # Calculate the intersection point of the perpendicular line with the brain surface
    x_intersection = (brain_surface_y_at_nearest_x - y_intercept) / perpendicular_slope
    y_intersection = perpendicular_slope * x_intersection + y_intercept
    
    # Calculate the distance between the ROI and the intersection point
    distance = np.sqrt((roi_x - x_intersection) ** 2 + (roi_y - y_intersection) ** 2)
    
    return distance, (x_intersection, y_intersection), tangent_slope, nearest_x


#%% apply depth calculation other version with brain surface tangent, erpendiular line depth etc
grouped = pooled_surface_points.groupby(['AnimalID','Slice'])

for (animalID, slice_number), group_df in grouped:
    #calculate brain surface function 
    group_df = group_df.sort_values(by='X')
    xS = group_df['X']
    yS = group_df['Y']
    pchip = PchipInterpolator(xS, yS)
    #interpolate, choose range based on x surface measurements
    x_interp = np.linspace(xS.min()-50,xS.max()+50,500) #dont know if 500 points are necessary but ok
    y_interp = pchip(x_interp)
    #index into pooled_ROI_measurements using animal id and slice #pandas indexing
    rois = pooled_ROI_measurements[(pooled_ROI_measurements['AnimalID']==animalID)]
    #function calculate distance 
    break
    rois['distance'] = rois.apply(lambda row: calculate_perpendicular_distance(pchip, row['X'], row['Y'])[0], axis=1)
    rois['intersection'] = rois.apply(lambda row: calculate_perpendicular_distance(pchip, row['X'], row['Y'])[1], axis=1)
    rois['tangent_slope'] = rois.apply(lambda row: calculate_perpendicular_distance(pchip, row['X'], row['Y'])[2], axis=1)
    rois['nearest_x'] = rois.apply(lambda row: calculate_perpendicular_distance(pchip, row['X'], row['Y'])[3], axis=1)

    
    break#for testing

#MAYBE JUST CALCULATE DISTANCE BY ROTATE AND THEN NEAREST POINT IN 

#%% FIRST BATCH OF WORKING DEPTH CODE, keep version for illustrations. 
grouped = pooled_surface_points.groupby(['AnimalID','Slice'])
test = 0
for (animalID, slice_number), group_df in grouped:
    #calculate brain surface function 
    group_df = group_df.sort_values(by='X')
    xS = group_df['X']
    yS = group_df['Y']
    #find linear equation for brain surface
    m,b = linear_regression(xS,yS)
    theta = find_theta(m)
    #rotate brain surface points to be horizontal
    #redundancy in formatting in function, array, df pls update in future.
    xSR, ySR = rotate_points(xS,yS,-theta)
    group_df = group_df.assign(x_rotated=xSR, y_rotated = ySR)
    group_df = group_df.sort_values(by='x_rotated')
    #this reassignemnt is unfortunately necessary to maintain indeces matching with the df!!
    xSR = group_df['x_rotated']
    ySR = group_df['y_rotated']
    #find brain surface function for rotated points
    pchip = PchipInterpolator(xSR, ySR)
    #interpolate, choose range based on x surface measurements
    x_interp = np.linspace(xSR.min()-50,xSR.max()+50,500) #dont know if 500 points are necessary but ok
    y_interp = pchip(x_interp)
    #index into pooled_ROI_measurements using animal id and slice #pandas indexing
    rois = pooled_ROI_measurements[(pooled_ROI_measurements['AnimalID']==animalID)&(pooled_ROI_measurements['Slice']==slice_number)]
    rois = rois.sort_values(by='X')
    xC = rois['X']
    yC = rois['Y']
    #rotate cell points to match surface point rotation
    xCR, yCR = rotate_points(rois['X'],rois['Y'],-theta)
    rois = rois.assign(x_rotated=xCR, y_rotated=yCR)
    #this reassignemnt is unfortunately necessary to maintain indeces matching with the df!!
    xCR = rois['x_rotated']
    yCR = rois['y_rotated']
    #calculate distance from roi to brain surface (depth calc)
    y_brain = pchip(xCR)
    y_depth = np.abs(y_brain-yCR)
    #debugging code
    #check surfacerotation
    plt.figure()
    plt.scatter(xS,yS, color = 'blue', label = 'original')
    plt.scatter(xC,yC, color = 'skyblue', label = 'original cells')
    plt.scatter(xSR,ySR, color = 'red', label = 'rotated')
    plt.scatter(xCR,yCR, color = 'pink', label = 'rotated cells')
    plt.plot(x_interp, y_interp, label='PCHIP Interpolator')
    # Draw lines between y_brain and y_CR
    # Draw lines between y_brain and y_CR
    for i in range(len(xCR)):
        plt.plot([xCR.iloc[i], xCR.iloc[i]], [y_brain[i], yCR.iloc[i]], color='gray', linestyle='--')
        
    plt.legend()
    print(animalID, slice_number)
    test+=1
    if test >=5:
        break

#%% ** implement calculate depth!! woohoo
def calculate_depth(pooled_surface_points, pooled_ROI_measurements, debug=False):
    grouped = pooled_surface_points.groupby(['AnimalID', 'Slice'])
    depths = []

    for (animalID, slice_number), group_df in grouped:
        # Calculate brain surface function
        group_df = group_df.sort_values(by='X')
        xS = group_df['X']
        yS = group_df['Y']
        
        # Find linear equation for brain surface
        m, b = linear_regression(xS, yS)
        theta = find_theta(m)
        
        # Rotate brain surface points to be horizontal
        xSR, ySR = rotate_points(xS, yS, -theta)
        group_df = group_df.assign(x_rotated=xSR, y_rotated=ySR)
        group_df = group_df.sort_values(by='x_rotated')
        xSR = group_df['x_rotated']
        ySR = group_df['y_rotated']
        
        # Find brain surface function for rotated points
        pchip = PchipInterpolator(xSR, ySR)
        
        # Interpolate, choose range based on x surface measurements
        x_interp = np.linspace(xSR.min() - 50, xSR.max() + 50, 500)
        y_interp = pchip(x_interp)
        
        # Index into pooled_ROI_measurements using animal id and slice
        rois = pooled_ROI_measurements[(pooled_ROI_measurements['AnimalID'] == animalID) & 
                                       (pooled_ROI_measurements['Slice'] == slice_number)]
        rois = rois.sort_values(by='X')
        xC = rois['X']
        yC = rois['Y']
        
        # Rotate cell points to match surface point rotation
        xCR, yCR = rotate_points(rois['X'], rois['Y'], -theta)
        rois = rois.assign(x_rotated=xCR, y_rotated=yCR)
        xCR = rois['x_rotated']
        yCR = rois['y_rotated']
        
        # Calculate depth from ROI to brain surface
        y_brain = pchip(xCR)
        y_depth = np.abs(y_brain - yCR)
        
        # Append depth information
        rois['Depth'] = y_depth
        depths.append(rois[['AnimalID', 'Slice', 'X', 'Y', 'Depth']])
        
        # Plotting for debugging
        if debug:
            plt.figure()
            plt.scatter(xS, yS, color='blue', label='original')
            plt.scatter(xC, yC, color='skyblue', label='original cells')
            plt.scatter(xSR, ySR, color='red', label='rotated')
            plt.scatter(xCR, yCR, color='pink', label='rotated cells')
            plt.plot(x_interp, y_interp, label='PCHIP Interpolator')
            
            # Draw lines between y_brain and y_CR
            for i in range(len(xCR)):
                plt.plot([xCR.iloc[i], xCR.iloc[i]], [y_brain[i], yCR.iloc[i]], color='gray', linestyle='--')
                
            plt.legend()
            plt.title(f'AnimalID: {animalID}, Slice: {slice_number}')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.show()

    # Combine all depths into a single DataFrame
    depths_df = pd.concat(depths)
    
    # Merge the depths back into the original DataFrame
    pooled_ROI_measurements = pd.merge(pooled_ROI_measurements, 
                                       depths_df[['AnimalID', 'Slice', 'X', 'Y', 'Depth']], 
                                       on=['AnimalID', 'Slice', 'X', 'Y'], 
                                       how='left')
    #multiply depth column by scaling factor 
    pooled_ROI_measurements['Depth_um'] = pooled_ROI_measurements['Depth']*1.515 
    pooled_ROI_measurements = pooled_ROI_measurements.drop_duplicates()
    return pooled_ROI_measurements

#if error bc depth already added to df 
    #pooled_ROI_measurements = pooled_ROI_measurements.drop(columns=['Depth_um','Depth'])
# call it
pooled_ROI_measurements = calculate_depth(pooled_surface_points, pooled_ROI_measurements, debug=True)

#%% visualization histogram of depth values split by MOp VISp
#split mop and visp , hue aca orb 
MOp_roi = pooled_ROI_measurements[pooled_ROI_measurements['Region']=='MOp']
VISp_roi = pooled_ROI_measurements[pooled_ROI_measurements['Region']=='VISp']
sns.displot(data = MOp_roi, x='Depth', hue='Origin')
sns.displot(data = VISp_roi, x='Depth', hue='Origin')
#%% visualization density plot, customizable subplots
from matplotlib.gridspec import GridSpec
#common norm ensures that density is normalized per independent group

#extract data for scatter plots
overlaps_df = pooled_ROI_measurements[(pooled_ROI_measurements['Channel']!=2) & (pooled_ROI_measurements['Overlap']==1)]
overlaps_df['x_scatter'] = np.random.rand(overlaps_df.shape[0])

fig = plt.figure(figsize=(12, 8))
fig.suptitle('Visual cortex PFC projections')
# Define a GridSpec with 1 row and 2 columns, where the first column is wider
gs = GridSpec(1, 2, width_ratios=[1,2])
# Create subplots with the specified GridSpec
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])
sns.kdeplot(data = VISp_roi, y='Depth', hue='Origin', common_norm=False, ax=ax1)
sns.scatterplot(data = overlaps_df[overlaps_df['Region']=='VISp'],x='x_scatter',y='Depth',hue='Channel',ax=ax2)


#%% visualization density plot, customizable subplots
#common norm ensures that density is normalized per independent group
pooled_ROI_measurements = pooled_ROI_measurements.drop_duplicates()
#extract data for scatter plots
overlaps_df = pooled_ROI_measurements[(pooled_ROI_measurements['Channel']!=2) & (pooled_ROI_measurements['Overlap']==1)]
overlaps_df['x_scatter'] = np.random.rand(overlaps_df.shape[0])

#VISUAL PLOT
plt.rcParams['pdf.use14corefonts']=True
fig = plt.figure(figsize=(6, 4))
fig.suptitle('Visual cortex PFC projections')
# Define a GridSpec with 1 row and 2 columns, where the first column is wider
gs = GridSpec(1, 3, width_ratios=[2,1,1])
# Create subplots with the specified GridSpec
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])
ax3 = fig.add_subplot(gs[2])
#custom hue mapping
hue_order = [0, 1, 3]
hue_labels = ['PV', 'VIP', 'SST']
custom_palette = {0: 'blue', 1: 'green', 3: 'red'}

ax1.set_title('Density plot for PFC->VISp')
sns.kdeplot(data = VISp_roi, y='Depth', hue='Origin', common_norm=False, ax=ax1)
ax2.set_title('Interneuron Dist. ACA')
sns.scatterplot(data = overlaps_df[(overlaps_df['Region']=='VISp') & (overlaps_df['Origin']=='ACA')],x='x_scatter',y='Depth',hue='Channel',hue_order=hue_order,palette=custom_palette,ax=ax2)
#ax2.legend(title='Channel', labels=hue_labels)
ax3.set_title('Interneuron Dist. ORB')
sns.scatterplot(data = overlaps_df[(overlaps_df['Region']=='VISp') & (overlaps_df['Origin']=='ORB')],x='x_scatter',y='Depth',hue='Channel',hue_order=hue_order,palette=custom_palette,ax=ax3)
#ax3.legend(title='Channel', labels=hue_labels)

# Set the same y-axis limits for all subplots
y_min = -100
y_max = 800
ax1.set_ylim(y_min, y_max)
ax2.set_ylim(y_min, y_max)
ax3.set_ylim(y_min, y_max)
# Invert y-axis for all subplots
ax1.invert_yaxis()
ax2.invert_yaxis()
ax3.invert_yaxis()
#legend
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=custom_palette[k], markersize=10) for k in hue_order]
ax2.legend(handles=handles, title='Channel', labels=hue_labels)
ax3.legend(handles=handles, title='Channel', labels=hue_labels)

plt.tight_layout()

#update to os.join when made into function
save_path = '/Users/emmaodom/Documents/ATLAS_dualSiteInj_VISp_MOp_0724/Results'
Region = 'VISp'
save_svg = save_path + '/svg/' + f'{Region}_PFC_projections.pdf'
plt.savefig(save_svg, format = 'pdf')
#plt show must come after savefig
plt.show()
#%%
#MOTOR PLOT
plt.rcParams['pdf.use14corefonts']=True
fig = plt.figure(figsize=(6, 4))
fig.suptitle('Motor cortex PFC projections')
# Define a GridSpec with 1 row and 2 columns, where the first column is wider
gs = GridSpec(1, 3, width_ratios=[2,1,1])
# Create subplots with the specified GridSpec
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])
ax3 = fig.add_subplot(gs[2])
#custom hue mapping
hue_order = [0, 1, 3]
hue_labels = ['PV', 'VIP', 'SST']
custom_palette = {0: 'blue', 1: 'green', 3: 'red'}

ax1.set_title('Density plot for PFC->MOp')
sns.kdeplot(data = MOp_roi, y='Depth', hue='Origin', common_norm=False, ax=ax1)
ax2.set_title('Interneuron Dist. ACA')
sns.scatterplot(data = overlaps_df[(overlaps_df['Region']=='MOp') & (overlaps_df['Origin']=='ACA')],
                x='x_scatter',y='Depth',hue='Channel',hue_order=hue_order,palette=custom_palette,ax=ax2)
#ax2.legend(title='Channel', labels=hue_labels)
ax3.set_title('Interneuron Dist. ORB')
sns.scatterplot(data = overlaps_df[(overlaps_df['Region']=='MOp') & (overlaps_df['Origin']=='ORB')],
                x='x_scatter',y='Depth',hue='Channel',hue_order=hue_order,palette=custom_palette,ax=ax3)
#ax3.legend(title='Channel', labels=hue_labels)

# Set the same y-axis limits for all subplots
y_min = -100
y_max = 1250
ax1.set_ylim(y_min, y_max)
ax2.set_ylim(y_min, y_max)
ax3.set_ylim(y_min, y_max)
# Invert y-axis for all subplots
ax1.invert_yaxis()
ax2.invert_yaxis()
ax3.invert_yaxis()
#legend
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=custom_palette[k], markersize=10, linestyle='None') for k in hue_order]
ax2.legend(handles=handles, title='Channel', labels=hue_labels)
ax3.legend(handles=handles, title='Channel', labels=hue_labels)

plt.tight_layout()
#update to os.join when made into function
save_path = '/Users/emmaodom/Documents/ATLAS_dualSiteInj_VISp_MOp_0724/Results'
Region = 'MOp'
save_svg = save_path + '/svg/' + f'{Region}_PFC_projections.pdf'
plt.savefig(save_svg, format = 'pdf')
#plt show must come after savefig

plt.show()
#%%
#oh i can make a density plot for all overlaps
#%% traditional subplots with seaborn integration
fig, axs = plt.subplots(1,2, figsize=(12,6), sharey = True)
fig.suptitle('Motor cortex PFC projections')
sns.kdeplot(data = MOp_roi, y='Depth', hue='Origin', common_norm=False,ax=axs[0])
sns.scatterplot(data = overlaps_df[overlaps_df['Region']=='MOp'],x='x_scatter',y='Depth',hue='Channel')
plt.ylim(-100,2000)
#%% visualization scatter plot colored by VIP/SST/PV 
overlaps_df = pooled_ROI_measurements[(pooled_ROI_measurements['Channel']!=2) & (pooled_ROI_measurements['Overlap']==1)]
overlaps_df['temp'] = np.random.rand(overlaps_df.shape[0])#rescale *0.1
plt.figure()
sns.scatterplot(data=overlaps_df,x='temp',y='Depth',hue='Channel')

#hue = channel 
#exclude ch2 
#select overlap ==1 only
#set x = constant (array of length df)
#use depth as y value
#scatter , next to density plot 
#%% make violin plots for density distribution 


#%%
# Group by 'AnimalID' and 'Region', then calculate unique slices per group
unique_slices = pooled_ROI_measurements.groupby(['AnimalID', 'Origin', 'Region'])['Slice'].nunique().reset_index()

# Rename the column for clarity
unique_slices.rename(columns={'Slice': 'UniqueSlices'}, inplace=True)

# Display the result
print(unique_slices)

mean_unique_slices = unique_slices.groupby(['Origin', 'Region'])['UniqueSlices'].mean().reset_index()
print(mean_unique_slices)

std_unique_slices = unique_slices.groupby(['Origin', 'Region'])['UniqueSlices'].std().reset_index()
print(std_unique_slices)

#%%

