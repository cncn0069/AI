from tensorflow.keras.utils import to_categorical
from tensorflow.keras import models
import numpy as np

model = models.load_model(filepath=MODEL_SAVE_PATH)

model.summary()

np_handler = np.load("test_lpc_data_and_label.npz")

for key in np_handler.keys():
    print(key)

test_data, test_label = np_handler['test_data'], np_handler['test_label']

print("shape of test_data", np.shape(test_data))
print("shape of test_label", np.shape(test_label))

test_label = to_categorical(test_label)

loss, acc = model.evaluate(test_data, test_label)
print("복원된 모델의 정확도: {:5.2f}%".format(100 * acc))
