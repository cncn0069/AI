from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import numpy as np
import shutil
import os

import utils.train_utils as tu

np_handler = np.load("plp_data_and_label.npz")
raw_x, raw_y = np_handler['plp_data'], np_handler['plp_label']


x_train, x_valid, y_train, y_valid = train_test_split(raw_x, raw_y, test_size=0.20, shuffle=True)
y_train, y_valid = to_categorical(y_train), to_categorical(y_valid)

print("shape of x_train, x_valid: ", np.shape(x_train), np.shape(x_valid))
print("shape of y_train, y_valid: ", np.shape(y_train), np.shape(y_valid))

model = tu.create_model()
model.summary()

ckpt_path = CHECKPOINT_PATH
model_path = MODEL_SAVE_PATH
log_dir = "train_log"
if os.path.exists(log_dir):
    shutil.rmtree(log_dir)
    os.mkdir(path=log_dir)

train_hist = tu.train_model(
    model=model,
    x_train=x_train,
    x_valid=x_valid,
    y_train=y_train,
    y_valid=y_valid,
    ckpt_path=ckpt_path,
    model_path=model_path,
    log_dir=log_dir
)

tu.training_visualization(train_hist.history)
