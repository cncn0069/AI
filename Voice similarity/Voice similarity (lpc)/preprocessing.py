from utils import data_utils as du
import numpy as np

train_data_dir = r"C:\Users\admin\Documents\AI\data\voice_recognition\train"
test_data_dir = r"C:\Users\admin\Documents\AI\data\voice_recognition\test"

train_data, train_label = du.raw_data_processing(train_data_dir)
test_data, test_label = du.raw_data_processing(test_data_dir)

np.savez_compressed("train_lpc_data_and_label.npz", train_data=train_data, train_label=train_label)
np.savez_compressed("test_lpc_data_and_label.npz", test_data=test_data, test_label=test_label)
