from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

from utils import train_utils as tu

arr_handler = np.load("raw_data.npz")

x_data, y_data = arr_handler['raw_x'], arr_handler['raw_y']
x_data = np.expand_dims(x_data, -1)
x_train, x_valid, y_train, y_valid = train_test_split(x_data, y_data, test_size=0.3, shuffle=True)

model = tu.make_model()

model.summary()

ckpt_path = CHECKPOINT PATH
model_save_path = MODEL SAVE PATH

history = tu.fit_model(
    model=model,
    x_train=x_train,
    y_train=y_train,
    x_valid=x_valid,
    y_valid=y_valid,
    ckpt_path=ckpt_path
)

model.load_weights(ckpt_path)
model.save(model_save_path)

hist = history.history
tu.training_visualization(hist)
plt.show()
