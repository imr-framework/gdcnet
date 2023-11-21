import os
import numpy as np
import nibabel

from scipy.ndimage import zoom

def iterate_folder_and_subfolders(input_folder):
    sub2_dirs = []

    sub_dirs = [os.path.join(input_folder, dir) for dir in os.listdir(input_folder)]
    for sub_dir in sub_dirs:
        for sub2_name in os.listdir(sub_dir):
            if os.path.isdir(os.path.join(sub_dir, sub2_name)):
                sub2_dir_path = os.path.join(sub_dir, sub2_name)
                sub2_dirs.append(sub2_dir_path)
    return sub2_dirs

def filter_folder_by_string(folder, string):
    '''
    Parameters
    ----------------
    folder: str
        Path to the folder to be filtered.
    string: str
        String to filter the folder by.
    Returns
    -------
    filtered_filename_path: string
        Path to the filtered file.
    '''
    filtered_filename_path = [os.path.join(folder, filename) for filename in os.listdir(folder) if string in filename]
    return filtered_filename_path

def generate_gdcnet_data(src_dir, dst_dirs):
    '''
    Parameters
    ----------
    src_dir: str
        Path to the folder containing the raw data
    dst_dirs: list
        List of paths to the folders where the preprocessed data will be saved.
    Returns
    -------
    None
    '''
    # Generate save_dir if it does not exist
    for dst_dir in dst_dirs:
        if not os.path.exists(dst_dir):
            os.mkdir(dst_dir)

    # Get the directories in the data_dir
    databases = os.listdir(src_dir)
    train_testID = databases[1:] # In distribution testing
    testOOD = databases[0] # Out-of-distribution testing (ds000224)

    # PE bandwidths in Hz
    BW_PE = {'ds000224': 26.483, 'ds001454': 47.384, 'ds002799': 27.902, 'ds004101': 30.637}

    # Loop over all files and subfolders in the input folder
    subjects_dirs = iterate_folder_and_subfolders(src_dir)
    subjects_traintestID = [subject_dir for subject_dir in subjects_dirs if
                      os.path.dirname(subject_dir).split('\\')[-1] in train_testID]
    subjects_testOOD = [subject_dir for subject_dir in subjects_dirs if
                     os.path.dirname(subject_dir).split('\\')[-1] in testOOD]

    # Load the data and preprocess it
    iteration = 0
    for subject_i in subjects_dirs:

        # LOADING DATA
        # Distorted EPI
        EPI_raw_path = filter_folder_by_string(os.path.join(subject_i, 'func'), 'example_func_10_brain.nii')[0]
        EPI_raw = nibabel.load(EPI_raw_path).get_fdata()

        # T1w data
        T1w_path = filter_folder_by_string(os.path.join(subject_i, 'anat'), 'rT1w_brain.nii')[0]
        T1w = nibabel.load(T1w_path).get_fdata()

        # T1w mask
        T1w_mask_path = filter_folder_by_string(os.path.join(subject_i, 'anat'), 'rT1w_brain_mask.nii')[0]
        T1w_mask = nibabel.load(T1w_mask_path).get_fdata()

        # Field map
        fmap_path = filter_folder_by_string(os.path.join(subject_i, 'fmap'), 'fmap_rads.nii')[0]
        fmap = nibabel.load(fmap_path).get_fdata() / (2 * np.pi)  # in Hz

        # Get the database number ds00xxxx
        database = os.path.dirname(subject_i).split('\\')[-1]
        VDM = fmap / BW_PE[database] # Calculate the voxel displacement map (VDM)

        # Correction benchmark (FUGUE)
        EPI_FUGUE_path = filter_folder_by_string(os.path.join(subject_i, 'func'), 'example_func_10_FUGUE_brain.nii')[0]
        EPI_FUGUE = nibabel.load(EPI_FUGUE_path).get_fdata()

        # PREPROCESSING
        # Select 5 random time frames from the EPI data
        time_frames = np.random.choice(np.arange(0, EPI_raw.shape[-1]), 5, replace=False)
        EPI_raw = EPI_raw[..., time_frames]
        EPI_FUGUE = EPI_FUGUE[..., time_frames]

        # Remove the top and bottom slices with no or low signal
        sl_sum = np.sum(T1w_mask, axis=(0, 1))
        sl0, sl1 = np.where(sl_sum > 615)[0][0], np.where(sl_sum > 615)[0][-1]
        T1w = T1w[..., sl0:sl1].astype(np.float32)
        VDM = VDM[..., sl0:sl1].astype(np.float32)
        EPI_raw = EPI_raw[..., sl0:sl1, :].astype(np.float32)
        EPI_FUGUE = EPI_FUGUE[..., sl0:sl1, :].astype(np.float32)

        # Interpolate to 32 slices
        nslices = T1w.shape[-1]
        if nslices != 32:
            scale_factor = 32 / nslices
            T1w = zoom(T1w, (1, 1, scale_factor))
            VDM = zoom(VDM, (1, 1, scale_factor))
            EPI_raw_32 = np.zeros((T1w.shape[0], T1w.shape[1], 32, EPI_raw.shape[-1]))
            EPI_FUGUE_32 = np.zeros((T1w.shape[0], T1w.shape[1], 32, EPI_FUGUE.shape[-1]))
            for tf in range(EPI_raw.shape[-1]):
                EPI_raw_32[..., tf] = zoom(EPI_raw[..., tf], (1, 1, scale_factor))
                EPI_FUGUE_32[..., tf] = zoom(EPI_FUGUE[..., tf], (1, 1, scale_factor))

        # Normalize the images to [0, 1]
        T1w = (T1w - T1w.min()) / (T1w.max() - T1w.min())
        EPI_raw = (EPI_raw_32 - EPI_raw_32.min()) / (EPI_raw_32.max() - EPI_raw_32.min())
        EPI_FUGUE = (EPI_FUGUE_32 - EPI_FUGUE_32.min()) / (EPI_FUGUE_32.max() - EPI_FUGUE_32.min())



        # Save the data
        for tf in range(EPI_raw.shape[-1]):
            data_tf = np.concatenate((np.expand_dims(EPI_raw[..., tf], 0), np.expand_dims(T1w,0), np.expand_dims(VDM, 0), np.expand_dims(EPI_FUGUE[..., tf], 0)), axis=0)

            # save data to a numpy file zero-filling the iteration and time frame numbers
            if subject_i in subjects_traintestID:
                np.save(os.path.join(dst_dirs[0], f"sub-{str(iteration+1).zfill(2)}_tf-{str(tf + 1).zfill(2)}.npy"), data_tf)
            else:
                np.save(os.path.join(dst_dirs[1], f"sub-{str(iteration+1).zfill(2)}_tf-{str(tf + 1).zfill(2)}.npy"), data_tf)

        iteration += 1

        print(f'Finished subject {iteration} of {len(subjects_dirs)}')





if __name__ == '__main__':
    raw_data_dir = r'/data/raw/60_subjects_clean'
    save_dirs = [r'D:\MMJ\githubrepo\gdcnet\data\preprocessed\train_testID', r'D:\MMJ\githubrepo\gdcnet\data\preprocessed\testOOD']
    generate_gdcnet_data(raw_data_dir, save_dirs)