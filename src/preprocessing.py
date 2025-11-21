import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input


def create_data_generators(
    train_dir,
    test_dir,
    img_size=(224, 224),
    batch_size=32,
    validation_split=0.2
):
    """
    Creates train, validation and test data generators.
    Uses VGG16 preprocessing and standard augmentation.
    """

    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=15,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest",
        validation_split=validation_split
    )

    test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="binary",
        subset="training",
        shuffle=True
    )

    validation_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="binary",
        subset="validation",
        shuffle=False
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="binary",
        shuffle=False
    )

    return train_generator, validation_generator, test_generator


def create_new_data_generator(
    new_data_dir,
    img_size=(224, 224),
    batch_size=32
):
    """
    Creates a generator for newly uploaded data that will be used in retraining.
    This folder must contain subfolders for each class.
    """

    new_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )

    new_data_generator = new_datagen.flow_from_directory(
        new_data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="binary",
        shuffle=True
    )

    return new_data_generator
