import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.metrics import Precision, Recall

def create_model(input_shape=(224, 224, 3), l2_factor=0.0001, dropout_rate=0.3):
    """
    Creates a VGG16 transfer learning model with a custom classifier head.
    Optimized for chest X-ray classification.
    """
    # Load VGG16 without top layers
    base_model = tf.keras.applications.VGG16(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False  # Freeze base initially

    # Input
    inputs = layers.Input(shape=input_shape)
    x = tf.keras.applications.vgg16.preprocess_input(inputs)

    # Base model
    x = base_model(x, training=False)
    x = layers.Flatten()(x)

    # Custom classifier head
    x = layers.Dense(512, activation="relu", kernel_regularizer=regularizers.l2(l2_factor))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(l2_factor))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs, outputs)
    model.base_model = base_model  # store base for fine-tuning

    # Compile with multiple metrics for evaluation
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            Precision(name="precision"),
            Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc")
        ]
    )

    return model


def train_model(model, train_generator, validation_generator, epochs, model_path, fine_tune=False):
    """
    Train the model with early stopping and optional fine-tuning.
    """
    early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    checkpoint = ModelCheckpoint(filepath=model_path, monitor="val_loss", save_best_only=True)

    # Initial training
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=epochs,
        callbacks=[early_stop, checkpoint]
    )

    # Optional fine-tuning of last conv block
    if fine_tune:
        print("Starting fine-tuning...")
        base_model = model.base_model
        base_model.trainable = True

        # Freeze all layers except last 4 conv layers
        for layer in base_model.layers[:-4]:
            layer.trainable = False

        model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="binary_crossentropy",
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc")
    ]
)


        history_fine = model.fit(
            train_generator,
            validation_data=validation_generator,
            epochs=epochs // 2,
            callbacks=[early_stop, checkpoint]
        )

        # Merge histories
        for key in history.history.keys():
            history.history[key] += history_fine.history[key]

    return history


def save_model(model, path):
    model.save(path)
    print("Model saved at", path)


def load_trained_model(path):
    model = tf.keras.models.load_model(path)
    print("Model loaded from", path)
    return model
