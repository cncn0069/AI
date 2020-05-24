from tensorflow.keras import models, layers, activations, optimizers, losses, callbacks


def make_model():
    model = models.Sequential([
        layers.Dense(4096, activation=activations.relu, input_shape=(128, 5195)),
        layers.Dense(4096, activation=activations.relu),
        layers.Dense(4096, activation=activations.relu),
        layers.Dropout(rate=0.5),
        layers.Dense(2048, activation=activations.relu),
        layers.Dense(2048, activation=activations.relu),
        layers.Dense(2048, activation=activations.relu),
        layers.Dropout(rate=0.5),
        layers.Dense(1024, activation=activations.relu),
        layers.Dense(1024, activation=activations.relu),
        layers.Dropout(rate=0.5),
        layers.Dense(512, activation=activations.relu),
        layers.Dense(512, activation=activations.relu),
        layers.Dropout(rate=0.5),

        layers.Dense(2, activation=activations.sigmoid)
    ])

    model.compile(
        optimizer=optimizers.Adam(),
        loss=losses.binary_crossentropy,
        metrics=['acc']
    )

    return model
