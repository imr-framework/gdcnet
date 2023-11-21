import os

def data_split(src_dir, dst_dir, split_test, split_val):
    '''
    Parameters
    ----------
    src_dir : str
        Path to the directory containing the data.
    dst_dir : str
        Path to the model directory where the txt files will be saved.
    split_test : float
        Percentage of the data to be used for testing.
    split_val : float
        Percentage of the training data to be used for validation.
    '''
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    # Read the files in the data folder
    files = os.listdir(src_dir)
    sub = [*set([file_i.split('_')[0].split('/')[-1] for file_i in files])]

    # Train-test split
    train_subjects = sub[:int(len(sub) * split_test)]
    test_subjects = sub[int(len(sub) * split_test):]

    # Train-val split
    train_subjects = train_subjects[:int(len(train_subjects) * split_val)]
    val_subjects = train_subjects[int(len(train_subjects) * split_val):]

    train_files = [file_i for file_i in files if file_i.split('_')[0].split('/')[-1] in train_subjects]
    test_files = [file_i for file_i in files if file_i.split('_')[0].split('/')[-1] in test_subjects]
    val_files = [file_i for file_i in files if file_i.split('_')[0].split('/')[-1] in val_subjects]

    # Write the train, validation, and test files in the model folder in a txt file
    with open(os.path.join(dst_dir, 'train.txt'), 'w') as f:
        for file in train_files:
            f.write(file + '\n')

    with open(os.path.join(dst_dir, 'val.txt'), 'w') as f:
        for file in val_files:
            f.write(file + '\n')

    with open(os.path.join(dst_dir, 'test.txt'), 'w') as f:
        for file in test_files:
            f.write(file + '\n')


if __name__ == '__main__':
    path_data = r'D:\MMJ\githubrepo\gdcnet\data\preprocessed\train_testID'
    path_model = r'D:\MMJ\githubrepo\gdcnet\models\20231115'
    split_test = 0.8
    split_val = 0.8
    data_split(path_data, path_model, split_test, split_val)