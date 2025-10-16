import os
import zipfile
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from . import config

def extract_zip(src_path, targ_path):
    """Extracts a zip file to a target path."""
    if not os.path.exists(targ_path):
        os.makedirs(targ_path)
    with zipfile.ZipFile(src_path, "r") as zip_ref:
        zip_ref.extractall(targ_path)

def get_data_generators():
    """
    Prepares data generators for training and testing.
    """
    extract_zip(config.SRC_TRAIN_TEST_PATH, config.DATASET_PATH)

    train_datagen = ImageDataGenerator(rescale=1.0/255)
    test_datagen = ImageDataGenerator(rescale=1.0/255)

    train_generator = train_datagen.flow_from_directory(
        config.TRAIN_PATH,
        target_size=(config.HEIGHT, config.WIDTH),
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
    )

    test_generator = test_datagen.flow_from_directory(
        config.TEST_PATH,
        target_size=(config.HEIGHT, config.WIDTH),
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
    )

    return train_generator, test_generator
