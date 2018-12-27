import os

def get_latest_idx(folder_path):
    saved_file_list = os.listdir(folder_path)
    get_idxs = map(lambda file_path: os.path.basename(file_path).split('_')[-1], saved_file_list)
    maximum_idx_num = max(get_idxs)
    return maximum_idx_num