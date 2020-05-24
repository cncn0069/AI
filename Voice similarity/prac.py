import numpy as np

np_handler = np.load("raw_data.npz")
raw_x, raw_y = np_handler['raw_x'], np_handler['raw_y']
print(np.shape(raw_x))

raw_x = np.expand_dims(raw_x, -1)
print(np.shape(raw_x))
