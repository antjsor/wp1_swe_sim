import numpy as np

def Truncate_End_Threshold(data, keep=60, threshold=0.0):
    reversed_data = data[::-1]
    low_value_indices = np.where(reversed_data <= threshold)[0]

    if len(low_value_indices) == 0 or low_value_indices[0] != 0:
        return data, len(data)
    # Identify the last continuous low-value block
    last_block = np.split(low_value_indices, np.where(np.diff(low_value_indices) != 1)[0] + 1)[0]

    if len(last_block) > keep:
        keep_index = len(data) - len(last_block) + keep
        return data[:keep_index], keep_index
    else:
        return data, len(data)

