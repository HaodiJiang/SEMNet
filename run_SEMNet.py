# =========================================================================
#   (c) Copyright 2025
#   All rights reserved
#   Programs written by Haodi Jiang
#   Department of Computer Science
#   Sam Houston State University
#   Huntsville, Texas 77341, USA
#
#   Permission to use, copy, modify, and distribute this
#   software and its documentation for any purpose and without
#   fee is hereby granted, provided that this copyright
#   notice appears in all copies. Programmer(s) makes no
#   representations about the suitability of this
#   software for any purpose.  It is provided "as is" without
#   express or implied warranty.
# =========================================================================

import warnings

warnings.filterwarnings('ignore')
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import csv
from astropy.io import fits
import cv2
import sys

# Set paths
result_dir = 'results_SEM/KSO_1907_2007'
DATA_DIR = r'C:\Data\EUV_Project\test'  # Change to your actual data folder

# Ensure necessary directories exist
if not os.path.exists(result_dir):
    os.makedirs(result_dir)
if not os.path.exists(DATA_DIR):
    raise FileNotFoundError(f'DATA_DIR does not exist: {DATA_DIR}')

# Add result directories to path
sys.path.append(result_dir)

from SEMNet import *

# Load model
image_shape = (256, 256, 1)
model = SEMNet(image_shape).cnn_model()
model_name = 'Finetuned_Model_KSO_SEM_EUV.h5'
model.load_weights(model_name)

# Process data
target_size = (256, 256)
output_data = []

for filename in os.listdir(DATA_DIR):
    if filename.endswith(".fits"):
        print(filename)
        base_name = filename.split('.')[0]
        date_time = base_name.split('_')[1]
        date, time = date_time[:8], date_time[9:]
        timestamp = f"{date[:4]}-{date[4:6]}-{date[6:8]} {time[:2]}:{time[2:4]}"

        fits_path = os.path.join(DATA_DIR, filename)
        with fits.open(fits_path) as cak2_fits:
            cak2_fits.verify('fix')
            ck2_data = cak2_fits[0].data

        ck2_data = np.nan_to_num(ck2_data) / 1000
        ck2_data = cv2.resize(ck2_data, target_size, interpolation=cv2.INTER_AREA)
        input_ = np.expand_dims(np.expand_dims(ck2_data, axis=2), axis=0)

        output = model.predict(input_)
        pred_ch1 = round(output[0][0] / 1000, 8)
        pred_ch2 = round(output[0][1] / 1000, 8)

        output_data.append([timestamp, pred_ch1, pred_ch2])

# Save to CSV
csv_filename = os.path.join(result_dir, 'KSO_SEM_EUV.csv')
with open(csv_filename, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Time', 'pred_ch1n3 - SEM (26 - 34 nm)', 'pred_ch2 - SEM (0.1 - 50 nm)'])
    writer.writerows(output_data)
