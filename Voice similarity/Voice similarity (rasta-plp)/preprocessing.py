from utils import data_utils as du
import numpy as np

data_dir = DATA_DIR

data, label = du.raw_data_processing(data_dir)
print(np.shape(data))
print(np.shape(label))

np.savez_compressed("test_plp_data_and_label.npz", plp_data=data, plp_label=label)
