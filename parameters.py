########################################################################################################################
####            This file will take all parameters we would use for training and testing                            ####
########################################################################################################################

import os
import numpy as np
import glob

# model parameter: train or test
# If you wish to train the model:
# TRAINING: True, TESTING: True for automatically testing, false for not, REAL: False
# If you wish to test with pretrained model:
# TRAINING: False, TESTING: True, REAL: False
# If you wish to test real images:
# TRAINING/TESTING: doesn't matter, you can set it to be either True or False, REAL: True
TRAINING = True
TESTING = True
REAL = False

Scale = 4

# path parameter: path for training and testing data
# change the path to where you place the downloaded data
TRAINING_DATA_PATH = '../Dataset/Training'
TESTING_DATA_PATH = '../Dataset/Testing'
REAL_DATA_PATH = '../Dataset/Real'
SUBFOLDER_TRAININGDATA = 'TrainingSet'
SUBFOLDER_GROUNDTRUTH = 'GT'
SUBFOLDER_ISP = 'ISP'
RESULT_PATH = '../test_result'
LOG_DIR = '../log_dir'
FILENAME_REPORT = 'record.txt'

# model parameter: other parameter required no matter for training or testing
FIRST_STAGE = 'ours'
SECOND_STAGE = 'ours'
R = 4
EPS = 1e-2
# assert (SECOND_STAGE not in ['Ours', 'ours', 'OURS'] and (R is None or EPS is None)), 'Not ours but no r and eps defined'
FF = True
# CROP_SIZE is set to be 256 for Scale 2 and 128 for Scale 4
CROP_SIZE = 128
EPOCH_NUM = 0
BATCH_SIZE = 6
MAX_EPOCH = 40
PRETRAINED = False
# TEST_IMAGE_FOLDER = 'test' if not REAL else 'real'
TEST_IMAGE_FOLDER = 'test_with_overlap' if not REAL else 'real'
STEP_FILE = 'step.npy'
TRAINING_TRAIN_FILE = [] 
#if TRAINING and not REAL:
TRAINING_TRAIN_FILE = os.listdir(os.path.join(TRAINING_DATA_PATH, SUBFOLDER_TRAININGDATA)) 
TRAINING_CAPACITY = len(TRAINING_TRAIN_FILE)
BATCH_PER_EPOCH = np.ceil(float(TRAINING_CAPACITY)/BATCH_SIZE)

# parameters for training and testing
LEARNING_RATE = 2e-4
SWITCH_LEARNING_RATE = 1e-5
SWITCH_EPOCH = 20
SAVE_FREQ = 10
TEST_RATIO = 10
GROWTH_RATE = 16
KERNEL_SIZE_DENSE = 3
KERNEL_SIZE_NORMAL = 3
KERNEL_SIZE_POOLING = 2
BOTTLE_OUTPUT = 256
LAYER_PER_BLOCK_DENSE = 8
DECAY_COEF = 0.96
# First time forget to overlap
TEST_STEP = CROP_SIZE//2
SAVE_RAW = False
TESTING_TRAIN_FILE = [] 
if TESTING and not REAL:
    TESTING_TRAIN_FILE = os.listdir(os.path.join(TESTING_DATA_PATH, SUBFOLDER_TRAININGDATA))
TESTING_CAPACITY = len(TESTING_TRAIN_FILE)


# change settings for training or testing
if REAL or not TRAINING:
    MAX_EPOCH = EPOCH_NUM
    PRETRAINED = True
if REAL:
    #path_ = glob.glob(os.path.join(REAL_DATA_PATH, '*2759*'))[0]
    #TESTING_TRAIN_FILE = [path_]
    TESTING_TRAIN_FILE = os.listdir(REAL_DATA_PATH)
    TESTING_CAPACITY = len(TESTING_TRAIN_FILE)


# To make sure folders for containing results exist
def check_and_make_folder(path):
    if not os.path.isdir(path):
        os.mkdir(path)


check_and_make_folder(RESULT_PATH)
check_and_make_folder(LOG_DIR)
