#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 11:05:10 2024

@author: emmaodom
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage import io, exposure
from matplotlib.patches import Ellipse

all_roi_path = '/Volumes/T7/Motor_Spines_Pilot_Data/python_dataframes/master_roi_info.csv'
all_roi = pd.read_csv(all_roi_path)

# Step 1: Load your image
tiff_path = '/Volumes/T7/Motor_Spines_Pilot_Data/289N/231105/289N_231105_Cell2_dend2_920nm_20x_10xd_Tseries_81um_512px_0Avg-074/289N_231105_Cell2_dend2_920nm_20x_10xd_Tseries_81um_512px_0Avg-074-Ch2.tif_post_correction_z_projection.tif'
image = imread(tiff_path)

#filter all roi df 
directory = os.path.dirname(tiff_path)
sess_roi = all_roi[all_roi['directory']==directory]
sess_roi = sess_roi[::2] #take only even rois, which are spines. odd rois are background
fig, ax = plt.subplots()
ax.imshow(image, cmap='gray')  # Image in grayscale
# Calculate centers, width, and height for ovals
centers = np.column_stack((sess_roi['left'] + sess_roi['width'] / 2, 
                           sess_roi['top'] + sess_roi['height'] / 2))
widths = sess_roi['width']
heights = sess_roi['height']

for center, width, height in zip(centers, widths, heights):
    roi_oval = Ellipse(center, width, height, edgecolor='red', facecolor='none', linewidth=0.8)
    ax.add_patch(roi_oval)
plt.show() 

###CONTRAST ADJUSTED IMAGE W ROI OVERLAY and original for loop to plot roi.
image = io.imread(tiff_path)
# Adjust the contrast using Contrast Stretching
p2, p98 = np.percentile(image, (2, 98))
img_rescale = exposure.rescale_intensity(image, in_range=(p2, p98))

# Display the original and contrast-adjusted images
fig, ax = plt.subplots()
ax.imshow(img_rescale, cmap='gray') 
# Iterate over the DataFrame and add each ROI as an oval
for index, row in sess_roi.iterrows():
    # Extract ROI data
    print("row",row)
    center_x = row['left'] + row['width'] / 2
    center_y = row['top'] + row['height'] / 2
    width = row['width']
    height = row['height']
    # Create an Ellipse patch
    roi_oval = Ellipse((center_x, center_y), width, height, edgecolor='red', facecolor='none', linewidth=1)
    # Add the patch to the axes
    ax.add_patch(roi_oval)

plt.show()

'''
#ADD SAVE LINES 
path_parent = '/Volumes/T7/Motor_Spines_Pilot_Data/'
dir_save = 'python_dataframes'
path_save = path_parent + dir_save
# Save the dataframe to a CSV file in python_dataframes dir
sess_roi.to_csv(os.path.join(path_save, 'sess_roi.csv'), index=False)
'''