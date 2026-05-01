import glob
import re
import os
import pandas as pd
import numpy as np

__author__ = "C. I. Tang"
__copyright__ = "Copyright (C) 2020 C. I. Tang"

"""
Based on work of Tang et al.: https://arxiv.org/abs/2011.11542
Contact: cit27@cl.cam.ac.uk
License: GNU General Public License v3.0

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

def process_motion_sense_accelerometer_files(accelerometer_data_folder_path):
    """
    Preprocess the accelerometer files of the MotionSense dataset into the 'user-list' format
    Data files can be found at https://github.com/mmalekzadeh/motion-sense/tree/master/data

    Parameters:

        accelerometer_data_folder_path (str):
            the path to the folder containing the data files (unzipped)
            the trial folders should be directly inside it

    Return:
        
        user_datsets (dict of {user_id: [(sensor_values, activity_labels)]})
            the processed dataset in a dictionary, of type {user_id: [(sensor_values, activity_labels)]}
            the keys of the dictionary is the user_id (participant id)
            the values of the dictionary are lists of (sensor_values, activity_labels) pairs
                sensor_values are 2D numpy array of shape (length, channels=3)
                activity_labels are 1D numpy array of shape (length)
                each pair corresponds to a separate trial 
                    (i.e. time is not contiguous between pairs, which is useful for making sliding windows, where it is easy to separate trials)
    """

    # label_set = {}
    user_datasets = {}
    all_trials_folders = sorted(glob.glob(accelerometer_data_folder_path + "/*"))

    # Loop through every trial folder
    for trial_folder in all_trials_folders:
        trial_name = os.path.split(trial_folder)[-1]

        # label of the trial is given in the folder name, separated by underscore
        label = trial_name.split("_")[0]
        # label_set[label] = True
        print(trial_folder)
        
        # Loop through files for every user of the trail
        for trial_user_file in sorted(glob.glob(trial_folder + "/*.csv")):

            # use regex to match the user id
            user_id_match = re.search(r'(?P<user_id>[0-9]+)\.csv', os.path.split(trial_user_file)[-1])
            if user_id_match is not None:
                user_id = int(user_id_match.group('user_id'))

                # Read file
                user_trial_dataset = pd.read_csv(trial_user_file)
                user_trial_dataset.dropna(how = "any", inplace = True)

                # Extract the x, y, z channels
                values = user_trial_dataset[["x", "y", "z"]].values

                # the label is the same during the entire trial, so it is repeated here to pad to the same length as the values
                labels = np.repeat(label, values.shape[0])

                if user_id not in user_datasets:
                    user_datasets[user_id] = []
                user_datasets[user_id].append((values, labels))
            else:
                print("[ERR] User id not found", trial_user_file)
    
    return user_datasets

def process_uci_har_accelerometer_files(accelerometer_data_folder_path):
    """
    Preprocess the raw accelerometer files of the UCI HAR dataset into the 'user-list' format

    Parameters:

        accelerometer_data_folder_path (str):
            the path to the folder containing the data files (unzipped)
            e.g. UCI_HAR_Dataset/
            the train and test folders should be directly inside it (e.g. UCI_HAR_Dataset/train/)
            the raw data files should be inside the Inertial Signals subfolder of train and test folders (e.g. UCI_HAR_Dataset/train/Inertial Signals/)
            there is one file per accelerometer axis (e.g. total_acc_x_train.txt)

    Return:
        
        user_datsets (dict of {user_id: [(features, activity_labels)]})
            the processed dataset in a dictionary, of type {user_id: [(features, activity_labels)]}
            the keys of the dictionary is the user_id (participant id)
            the values of the dictionary are lists of (features, activity_labels) pairs
    """

    user_datasets = {}
    
    # Loop through train and test folders
    for dataset_type in ["train", "test"]:

        dataset_folder = os.path.join(accelerometer_data_folder_path, dataset_type)
        inertial_signals_folder = os.path.join(dataset_folder, "Inertial Signals")

        # Read raw data files
        total_acc_x = pd.read_csv(os.path.join(inertial_signals_folder, "total_acc_x_" + dataset_type + ".txt"), sep=r'\s+', header=None).to_numpy()
        total_acc_y = pd.read_csv(os.path.join(inertial_signals_folder, "total_acc_y_" + dataset_type + ".txt"), sep=r'\s+', header=None).to_numpy()
        total_acc_z = pd.read_csv(os.path.join(inertial_signals_folder, "total_acc_z_" + dataset_type + ".txt"), sep=r'\s+', header=None).to_numpy()

        # Read activity labels
        activity_labels = pd.read_csv(os.path.join(dataset_folder, "y_" + dataset_type + ".txt"), sep=r'\s+', header=None).to_numpy().flatten()

        # Read user ids
        user_ids = pd.read_csv(os.path.join(dataset_folder, "subject_" + dataset_type + ".txt"), sep=r'\s+', header=None).to_numpy().flatten().astype(int)

        num_samples = total_acc_x.shape[0]

        # Loop through samples
        for i in range(num_samples):
            user_id = user_id = int(user_ids[i])
            features = np.stack([total_acc_x[i], total_acc_y[i], total_acc_z[i]], axis=-1) 
            label = activity_labels[i]
            label_array = np.array([label] * features.shape[0])

            if user_id not in user_datasets:
                user_datasets[user_id] = []

            user_datasets[user_id].append((features, label_array))

    for user_id in list(user_datasets.keys()):
        
        all_features = [pair[0] for pair in user_datasets[user_id]]
        all_labels = [pair[1] for pair in user_datasets[user_id]]
        
        merged_features = np.concatenate(all_features, axis=0)
        merged_labels = np.concatenate(all_labels, axis=0)
        
        user_datasets[user_id] = [(merged_features, merged_labels)]
        
    return user_datasets
    
    return user_datasets
    

def process_capture24_accelerometer_files(accelerometer_data_folder_path):

    mapping_df = pd.read_csv(os.path.join(accelerometer_data_folder_path, 'annotation-label-dictionary.csv'))
    label_map = mapping_df.set_index('annotation')['label:Walmsley2020'].to_dict()

    user_datasets = {}
    all_files = sorted(glob.glob(accelerometer_data_folder_path + "/*"))

    for file_path in all_files:
        filename = os.path.split(file_path)[-1]

        user_id_match = re.search(r'P(?P<user_id>[0-9]{3})\.csv\.gz', filename)

        if user_id_match is not None:
                
            user_id = int(user_id_match.group('user_id'))
            print(file_path)

            df = pd.read_csv(file_path, compression='gzip', engine='pyarrow', dtype={4: str})
            df.dropna(how="any", inplace=True)

            df['annotation'] = df['annotation'].map(label_map)
            df.dropna(subset=['annotation'], inplace=True)

            df = df.sort_values('time').reset_index(drop=True)

            values = df[["x", "y", "z"]].values
            labels = df['annotation'].values

            if user_id not in user_datasets:
                user_datasets[user_id] = []

            user_datasets[user_id].append((values, labels))

    return user_datasets