import os
import numpy as np

def load_data_from_txt(data_dir, txt_file, ndims, training=True):
    '''
    Parameters
    ----------
    data_dir : str
        Directory where the pre-processed data is stored
    txt_file : str
        Text file containing the filenames of the data
    ndims : int
        Number of dimensions of the data (2 or 3)
    training : bool
        Whether to randomly permute the data for training
    Returns
    -------
    inputs : np.ndarray
        Input data (EPI and T1w images)
    outputs : np.ndarray
        Output data (VDM and benchmark EPI corrected images)
    '''
    data = []
    with open(txt_file, 'r') as f:
        for line in f:
            # Get only the filename
            data.append(line.strip())

    inputs = []
    outputs = []
    for d in data:
        # Get the pre-processed data
        d_tf = np.rot90(np.load(os.path.join(data_dir,d)), axes=[1,2])
        # Inputs and outputs
        # inputs.append(np.concatenate((np.expand_dims(d_tf[3], axis=0),np.expand_dims(d_tf[1], axis=0)), axis=0))
        # outputs.append(np.concatenate((np.expand_dims(d_tf[2], axis=0),np.expand_dims(d_tf[3], axis=0)), axis=0))
        inputs.append(d_tf[:2]) # EPI and T1w images
        outputs.append(d_tf[2:]) # VDM and benchmark EPI corrected images

    # Rearrange the data to fit input and output shapes
    if ndims == 2:
        inputs = np.swapaxes(np.concatenate(inputs, axis=-1), 0, -1)  # (N, 64, 64, 2)
        outputs = np.swapaxes(np.concatenate(outputs, axis=-1), 0, -1)  # (N, 64, 64, 2)
    elif ndims == 3:
        inputs = np.moveaxis(np.stack(inputs), 1, -1)  # (N, 64, 64, 32, 2)
        outputs = np.moveaxis(np.stack(outputs), 1, -1)  # (N, 64, 64, 32, 2)
    else:
        raise ValueError('ndims must be 2 or 3')

    # Randomly permute the data for training
    if training:
        perm = np.random.permutation(inputs.shape[0])
        inputs = inputs[perm]
        outputs = outputs[perm]

    return inputs, outputs
