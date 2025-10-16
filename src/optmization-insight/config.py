# Configuration for the optimization insight analysis

# Data
BATCH_SIZE = 1024
HEIGHT = 32
WIDTH = 32
DATASET_PATH = '/content/dataset'
TRAIN_PATH = f'{DATASET_PATH}/train'
TEST_PATH = f'{DATASET_PATH}/test'
# Path to the zip file of the dataset
# Replace with your actual path
SRC_TRAIN_TEST_PATH = '/content/drive/MyDrive/representation_learning/pure_dataset/datasets/cifar10.zip'


# Model
INPUT_SHAPE = (HEIGHT, WIDTH, 3)
NUM_CLASSES = 10

# Training
EPOCHS = 100
PROG_NAME = 'GradientHistAnalysis_V0'

# Callback
BIN_COUNT = 1000
CLIP = (-1e9, 1e9)
SAVE_CHECK_PERIODICALLY = True
PERIODIC_SAVE_DIR_PATH = f"/content/drive/MyDrive/representation_learning/LSMDWB_DataSet/checkpoints/{PROG_NAME}"
PERIOD = 5
LOAD_MODEL_PATH = f"{PERIODIC_SAVE_DIR_PATH}/{PROG_NAME}.keras"
LOAD_INTERIM_RESULTS_PATH = f"{PERIODIC_SAVE_DIR_PATH}/{PROG_NAME}.json"
LOGS_DIR = "/content/logs/"
