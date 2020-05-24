import numpy as np

from utils import train_utils as tu

arr_handler = np.load("raw_data.npz")

x_data, y_data = arr_handler['raw_x'], arr_handler['raw_y']

model = tu.make_model()

model.fit(

)
