import os

def get_latest_epoch(folder_path):
    saved_file_list = os.listdir(folder_path)
    get_epochs = map(lambda file_path: os.path.basename(file_path).split('_')[-1], saved_file_list)
    maximum_epoch_num = max(get_epochs)
    return maximum_epoch_num