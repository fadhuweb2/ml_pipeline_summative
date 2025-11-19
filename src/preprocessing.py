import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input  # updated

def create_data_generators(
    train_dir,
    test_dir,
    img_size=(224, 224),
    batch_size=32,
    validation_split=0.2
):
    """
    Creates train, validation, and test data generators for image classification.
    Updated for VGG16 preprocessing and stable validation.

    Args:
        train_dir (str): Path to training data folder.
        test_dir (str): Path to testing data folder.
        img_size (tuple): Target size for image resizing.
        batch_size (int): Number of images per batch.
        validation_split (float): Fraction of training data for validation.

    Returns:
        train_generator, validation_generator, test_generator: Keras ImageDataGenerator objects.
    """

    # Training data generator with augmentation and VGG16 preprocessing
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

    # Test data generator (only preprocessing)
    test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )

    # Train generator
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="binary",
        subset="training",
        shuffle=True
    )

    # Validation generator (shuffle=False for stable evaluation)
    validation_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="binary",
        subset="validation",
        shuffle=False
    )

    # Test generator
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="binary",
        shuffle=False
    )

    return train_generator, validation_generator, test_generator
