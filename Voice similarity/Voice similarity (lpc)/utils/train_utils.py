from tensorflow.keras import models, layers, activations, optimizers, losses, callbacks
import matplotlib.pyplot as plt


def create_model():
    model = models.Sequential([
        layers.Dense(2048, activation=activations.relu, kernel_initializer='he_normal', input_dim=101),
        layers.Dense(2048, activation=activations.relu, kernel_initializer='he_normal'),
        layers.Dropout(rate=0.5),
        layers.Dense(1024, activation=activations.relu, kernel_initializer='he_normal'),
        layers.Dense(1024, activation=activations.relu, kernel_initializer='he_normal'),
        layers.Dropout(rate=0.5),
        layers.Dense(512, activation=activations.relu, kernel_initializer='he_normal'),
        layers.Dense(512, activation=activations.relu, kernel_initializer='he_normal'),
        layers.Dropout(rate=0.5),
        layers.Dense(256, activation=activations.relu, kernel_initializer='he_normal'),
        layers.Dense(256, activation=activations.relu, kernel_initializer='he_normal'),

        layers.Dense(4, activation=activations.softmax)
    ])

    model.compile(
        optimizer=optimizers.Adam(),
        loss=losses.categorical_crossentropy,
        metrics=['acc']
    )

    return model


def train_model(model, x_train, x_valid, y_train, y_valid, ckpt_path, model_path, log_dir):
    MONITOR = 'val_loss'

    history = model.fit(
        x=x_train, y=y_train,
        batch_size=16,
        epochs=500,
        callbacks=[
            callbacks.EarlyStopping(
                monitor=MONITOR,
                min_delta=1e-4,
                patience=10,
                verbose=2
            ),
            callbacks.ReduceLROnPlateau(
                monitor=MONITOR,
                factor=0.8,
                patience=5,
                verbose=2,
                min_lr=1e-4
            ),
            callbacks.ModelCheckpoint(
                filepath=ckpt_path,
                monitor=MONITOR,
                verbose=2,
                save_best_only=True,
                save_weights_only=True
            ),
            callbacks.TensorBoard(
                log_dir=log_dir
            )
        ],
        validation_data=(x_valid, y_valid)
    )

    model.save(filepath=model_path)

    return history


def training_visualization(hist):
    plt.subplot(2, 1, 1)
    plt.plot(hist['acc'], 'b')
    plt.plot(hist['val_acc'], 'g')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracies (train: Blue, valid: Green)")

    plt.subplot(2, 1, 2)
    plt.plot(hist['loss'], 'b')
    plt.plot(hist['val_loss'], 'g')
    plt.xlabel("Epochs")
    plt.ylabel("Losses (train: Blue, valid: Green)")

    plt.show()
