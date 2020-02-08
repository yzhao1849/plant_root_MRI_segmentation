import numpy as np
from os.path import join,expanduser

DATALOADER_NUM_WORKERS = 16

ROTATIONS = [0, 60, 120]
X_FLIPS = [0, 1]
Y_FLIPS = [0, 1]
X_Y_SWAPS = [0, 1]

MAX_ARTIFACT_THICKNESS = 5
PLANE_ARTIFACTS = ['xy_plane', 'yz_plane', 'xz_plane']
STRIDE_WEIGHT_MATRIX = 1 # stride used for calculating the weight matrix

SNR_THRESHOLDS = np.array([0, 10 ** 0.5, 10, 1000 ** 0.5, 100])


SOIL_DATA_FOR_VALIDATION = 'sand_d2_70x1x256x256_uint8.npy'

TO_DATA_TYPE_NAME_DICT = {'LupineApril2015': 'lupine_april_2015', 'Lupine_22august': 'lupine_22',
                          'gtk': 'gtk', 'lupine_small_xml': 'lupine_small', 'I_Soil_1W_DAP7': 'i_soil_1w_dap7',
                          'I_Soil_4D_DAP7': 'i_soil_4d_dap7', 'I_Sand_3D_DAP5': 'i_sand_3d_dap5'}


CHANCE_MIMIC_ROOT_VOID = 0.3

LOW_THRESHOLD_OF_GRADIENT = 1e-10


IF_IMITATE_POT = False  # True
IF_AUGMENT_ROOT = True  # True
IF_IMITATE_ALIASING = False  # True
IF_AUGMENT_SOIL = True
IF_ADD_VOIDS = False
IF_ROT_SOIL = False  # if to rotate soil when augmenting soil. Warning: slow!!!

ROOT_NOISE_LOWER = 0.9  # 0.8
ROOT_NOISE_UPPER = 1.1  # 1.2

APPLY_SOIL_GRADIENT_PROBABILITY = 0  # 0.2
GRADIENT_LOWER = 0.5  # 0.7
GRADIENT_UPPER = 1

RANDOM_SCALING_LOW = 0.2
RANDOM_SCALING_HIGH = 1.1

MIN_ROOT_WEIGHT = 1  #10

DIFF_TRAINING_CROPS_EACH_EPOCH = False  # if False, fix the training crops in each epoch
COPY_EVERYTHING_FROM_NILS = False  # if use the same data and network and training process (MSE loss)
MASK_SIDE = 5  # mask side: the outer side of the output will not be used in loss calculation or crop assembling

DATA_Z_FLIP = ['Lupine_22august', 'gtk', 'I_Soil_1W_DAP7', 'I_Soil_4D_DAP7']  # the data_names in Oguz' dataset to be flipped on the z-axis

# gradient diff loss
GDL_WEIGHT = 100

# if using the Laplace of gaussian filter for gradient diff loss
EDGE_LOSS_WEIGHT = 1