from tensorflow.keras import models, layers, activations, optimizers, losses, callbacks
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K


def make_model():
    model = models.Sequential([
        layers.MaxPooling2D(input_shape=(128, 1299, 1)),
        layers.Conv2D(filters=256, kernel_size=(3, 3), activation=activations.relu, kernel_initializer="he_normal"),
        layers.MaxPooling2D(),
        layers.Conv2D(filters=128, kernel_size=(3, 3), kernel_initializer="he_normal"),
        layers.MaxPooling2D(),
        layers.Conv2D(filters=128, kernel_size=(3, 3), kernel_initializer="he_normal"),
        layers.MaxPooling2D(),
        layers.Dropout(rate=0.5),

        layers.Flatten(),

        layers.Dense(512, activation=activations.relu, kernel_initializer="he_normal"),
        layers.Dense(512, activation=activations.relu, kernel_initializer="he_normal"),
        layers.Dropout(rate=0.5),
        layers.Dense(256, activation=activations.relu, kernel_initializer="he_normal"),
        layers.Dense(256, activation=activations.relu, kernel_initializer="he_normal"),
        layers.Dropout(rate=0.5),

        layers.Dense(1, activation=activations.sigmoid, kernel_initializer="he_normal")
    ])

    model.compile(
        optimizer=optimizers.Adam(),
        loss=losses.binary_crossentropy,
        metrics=['acc']
    )

    return model


def fit_model(model, x_train, y_train, x_valid, y_valid, ckpt_path):
    monitor = "val_loss"
    K.clear_session()

    history = model.fit(
        x=x_train, y=y_train,
        batch_size=16,
        epochs=50,
        verbose=1,
        callbacks=[
            callbacks.ModelCheckpoint(
                filepath=ckpt_path,
                monitor=monitor,
                verbose=2,
                save_best_only=True,
                save_weights_only=True
            ),
            callbacks.EarlyStopping(
                monitor=monitor,
                min_delta=1e-4,
                patience=25,
                verbose=2,
            ),
            callbacks.ReduceLROnPlateau(
                monitor=monitor,
                factor=0.8,
                patience=3,
                verbose=2,
                min_lr=1e-4
            )
        ],
        validation_data=(x_valid, y_valid)
    )

    return history


def training_visualization(hist):
    plt.title(label="blue is training, green is validation")
    plt.subplot(2, 1, 1)
    plt.plot(hist['acc'], 'b')
    plt.plot(hist['val_acc'], 'g')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracies")

    plt.subplot(2, 1, 2)
    plt.plot(hist['loss'], 'b')
    plt.plot(hist['val_loss'], 'g')
    plt.xlabel("Epochs")
    plt.ylabel("Losses")
