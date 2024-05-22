import numpy as np


def generate_server_testset(test_data, test_path):

    merged_dict = {'x': [], 'y': []}

    for d in test_data:
        for key, value in d.items():
            merged_dict[key].extend(value.tolist())
    merged_dict['x'] = np.array(merged_dict['x'])
    merged_dict['y'] = np.array(merged_dict['y'])
    with open(test_path + "server_testset.npz", 'wb') as f:
        np.savez_compressed(f, data=merged_dict)
