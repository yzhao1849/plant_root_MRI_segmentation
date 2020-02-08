import os
import sys
from copy import deepcopy
from os import path, makedirs
from os.path import exists
# import matplotlib
# matplotlib.use('TkAgg')
from time import time

import torch
from scipy import ndimage
from scipy.ndimage.morphology import binary_dilation
from torch import multinomial
from torch.nn import functional as F
from torch.utils.data import Dataset

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from Utils.constants import *
from Utils.errors import *
from Utils import root_types as rt
import math
import random
from skimage.measure import block_reduce


class Concat_datasets(Dataset):
    def __init__(self, datasets, fixed_mix_map=False):
        self.datasets = []
        for ds in datasets:  # remove empty datasets
            if len(ds.data_list) != 0:
                self.datasets.append(ds)

        len_list = [len(d) for d in self.datasets]
        self.concat_positions = np.concatenate(([0], np.cumsum(len_list)))
        if fixed_mix_map:  # for mixing up the different datasets randomly
            shuffled_indices = np.arange(0, self.concat_positions[-1])
            np.random.shuffle(shuffled_indices)
            indices = np.arange(0, self.concat_positions[-1])
            self.fixed_mix_map_dict = dict(zip(indices, shuffled_indices))
        else:
            self.fixed_mix_map_dict = None

    def __len__(self):
        return self.concat_positions[-1]

    def __getitem__(self, index):
        if self.fixed_mix_map_dict is not None:
            # mix the data from different datasets up using a fixed mixing mapping
            index = self.fixed_mix_map_dict[index]
        for i in range(len(self.concat_positions) - 1):
            d_start = self.concat_positions[i]
            d_end = self.concat_positions[i + 1]
            if d_start <= index < d_end:
                return self.datasets[i][index - d_start]


class DatasetManager_combining(Dataset):
    def __init__(self, data_list, data_dir, width=None, height=None, depth=None, super_resolution_factor=1.,
                 add_plane_artifacts=False, plane_artifacts_frequency=0.1, normalize=False,
                 is_training_set=True, crop_weight_offset=0, importance_sampling_root_perc=False,
                 training_data_padding=0, soil_data_list=None, soil_scaling_factor=None,
                 random_scaling=False, length=50000, diff_times=False, for_val=False, dont_care=False,
                 weight_base_dir='', use_depth=False, use_dist_to_center=False):
        super(DatasetManager_combining, self).__init__()

        self.width = width
        self.height = height
        self.depth = depth
        self.super_res_factor = int(super_resolution_factor)
        self.add_plane_artifacts = add_plane_artifacts
        self.plane_artifacts_frequency = plane_artifacts_frequency
        self.normalize = normalize
        self.is_training_set = is_training_set
        self.crop_weight_offset = float(crop_weight_offset)
        self.data_list = data_list
        self.soil_data_list = soil_data_list
        self.importance_sampling_root_perc = importance_sampling_root_perc
        self.training_data_padding = training_data_padding
        self.soil_scaling_factor = soil_scaling_factor  # only used for non-training
        self.combined_noisy_image = None
        self.random_scaling = random_scaling
        self.length = length
        self.diff_times = diff_times
        self.for_val = for_val  # if for validation, only one type of soil augmentation is applied;
        # if training, then the other 7 types are applied
        self.dont_care = dont_care
        self.data_dir = data_dir
        if importance_sampling_root_perc: assert weight_base_dir!=''
        self.weight_base_dir = weight_base_dir
        self.use_depth = use_depth
        self.use_dist_to_center = use_dist_to_center

    def __len__(self):
        return self.length

    def __getitem__(self, x):
        if self.is_training_set:

            if self.for_val or DIFF_TRAINING_CROPS_EACH_EPOCH:  # val crops should always be random
                new_seed = np.random.get_state()[1][1] - x
                if new_seed > 2 ** 32 - 1:  # will result in error if not in the range of (0, 2**32-1)
                    new_seed = 2 ** 32 - 1
                elif new_seed < 0:
                    new_seed = 0
                np.random.seed(new_seed)
                # not np.random.get_state()[1][0] ! otherwise np.random.get_state() does not change!!
                # also plus x! or will still be the same for each epoch!
                # because the init_func starts the worker with a fixed seed!
            else:  # if (not DIFF_TRAINING_CROPS_EACH_EPOCH) and (not self.for_val)
                np.random.seed(x)  # to make sure to generate the same crop when x is the same

            # randomly generate a index for getting a data
            idx = np.random.randint(low=0, high=self.data_list.shape[0])
            real_data = self.data_list[idx, -1]
        else:
            idx = x
            real_data = self.data_list[idx, 11]

        if not self.diff_times:
            ret, _ = load_crop_from_DatasetManager_combining(idx, self, self.data_dir, load_later_time=False,
                                                             for_val=self.for_val, weight_base_dir=self.weight_base_dir,
                                                             use_depth=self.use_depth,
                                                             use_dist_to_center=self.use_dist_to_center)
            ret['input_later_time'] = []
        elif real_data == 'True':  # if data is real, just copy the earlier time step as the later time step
            ret, _ = load_crop_from_DatasetManager_combining(idx, self, self.data_dir, load_later_time=False,
                                                             for_val=self.for_val, weight_base_dir=self.weight_base_dir,
                                                             use_depth=self.use_depth,
                                                             use_dist_to_center=self.use_dist_to_center)
            ret['input_later_time'] = deepcopy(ret['input'])
        else:
            # when using a second time point, do not use other input channels such as depth:
            assert self.use_depth is False
            assert self.use_dist_to_center is False
            ret, original_soil_idx = load_crop_from_DatasetManager_combining(idx, self, self.data_dir, load_later_time=False,
                                                                             for_val=self.for_val,
                                                                             weight_base_dir=self.weight_base_dir)
            start_positions = (ret['input_start_x'], ret['input_start_y'], ret['input_start_z'])
            type_of_soil = ret['noise_type']
            ret_later, _ = load_crop_from_DatasetManager_combining(idx, self, self.data_dir, load_later_time=True,
                                                                   start_positions=start_positions,
                                                                   type_of_soil=type_of_soil,
                                                                   original_soil_idx=original_soil_idx,
                                                                   for_val=self.for_val,
                                                                   weight_base_dir=self.weight_base_dir)
            ret['input_later_time'] = deepcopy(ret_later['input'])

        return ret


class DatasetManager(Dataset):
    def __init__(self, data_list, data_dir, width=None, height=None, depth=5, super_resolution_factor=1,
                 add_plane_artifacts=False, plane_artifacts_frequency=0.1,
                 normalize=False, is_training_set=True,
                 is_regression=False, crop_weight_offset=0, dont_care=False,
                 importance_sampling_root_perc=False, importance_sampling_gradient_norm=False,
                 training_data_padding=0, random_scaling=False, length=50000, diff_times=False, for_val=False,
                 test_real_data_dir=None, weight_base_dir='', use_depth=False, use_dist_to_center=False):
        """
        :param data_list: the array of virtual data: row -- image slice, column -- data feature
                               columns: [radius_multiplier, rotation, x_flip, y_flip, x_y_swap, noise_type, 
                                         noise_scale, data_name, dim_z, dim_x, dim_y, real_data]
                          the array of real data: row -- real data name (because each real 3D data should have its own name)
        :param width: the width of image slice after cropping. If None, the cropped width will equal the original image
        :param height: the height of image slice after cropping. If None, the cropped height will equal the original image
        :param depth: the number of image slice to be used in the input
        :param super_resolution_factor: if using super resolution ground truth
        :param add_plane_artifacts: if add noises that mimics the plane MRI artifacts on the fly, possible artifacts: xy_plane, yz_plane, xz_plane (randomly chosen)
        :param plane_artifacts_frequency: if add_plane_artifacts is True, the probability of adding random plane artifact to each input crop
        :param normalize: if the input crop should be normalized to standard normal distribution
        :param is_training_set: if this is a training set, then enable random cropping, otherwise do sequential slicing
        # :param root_radius_weighted: if True return a root_weight_map, calculated based on radius
        # :param histogram_bin_count: if True use histogram as one input layer
        :param is_regression: if load data for regression (the gt will be occupancy grid instead of thresholded root mask)
        :param crop_weight_offset: for training dataset, a offset is added to all possible image crops so that the weight of each will be larger than 0
        :param dont_care: if True, generate a mask for voxels close to the root surface (only useful in training and validation)
        :param importance_sampling_root_perc: if True, do importance sampling based on percentage of root voxels in each crop
        :param importance_sampling_gradient_norm: if True, do importance sampling based on the gradient norm of each crop after inputting to the network
        :param training_data_padding: the number of zero paddings to pad the whole training data before taking one random crop
        :param random_scaling: multiply the input crop with some random float between [RANDOM_SCALING_LOW, RANDOM_SCALING_HIGH]

        In order to get those cropped samples with more root voxels more often:
        Sample systematically the 3D image first, to calculate the weights (percentage of root voxels) of each sample.
        Then randomly pick one of these samples based on the weight, and uniformly randomly sample one crop from its 
        surrounding cubic region. --> is it too time-consuming? save it as a file?

        data is only loaded when the __getitem__ function is called

        one row (one 3D MRI image) of this data_list should look like:
            [radius_multiplier, rotation, x_flip, y_flip, x_y_swap, noise_type, 
            noise_scale, data_name, dim_z, dim_x, dim_y, real_data]
        """
        if len(data_list) != 0:
            min_x = data_list[:, 9].astype(int).min()
            min_y = data_list[:, 10].astype(int).min()
            min_z = data_list[:, 8].astype(int).min()

            if is_training_set:
                assert min_x >= width, 'the width after cropping should be smaller than the smallest width of all 3D images'
                assert min_y >= height, 'the height after cropping should be smaller than the smallest height of all 3D images'
                assert min_z >= depth, 'the depth after cropping should be smaller than the smallest depth of all 3D images'

        if importance_sampling_root_perc is True:  # these 2 should not be True at the same time
            assert importance_sampling_gradient_norm is False

        super(DatasetManager, self).__init__()

        self.width = width
        self.height = height
        self.depth = depth
        self.super_res_factor = int(super_resolution_factor)
        self.add_plane_artifacts = add_plane_artifacts
        self.plane_artifacts_frequency = plane_artifacts_frequency
        self.normalize = normalize
        self.is_training_set = is_training_set
        self.is_regression = is_regression
        self.crop_weight_offset = float(crop_weight_offset)
        self.dont_care = dont_care
        self.data_list = data_list
        self.importance_sampling_root_perc = importance_sampling_root_perc
        self.importance_sampling_gradient_norm = importance_sampling_gradient_norm
        self.training_data_padding = training_data_padding
        self.random_scaling = random_scaling
        self.length = length
        self.diff_times = diff_times
        self.for_val = for_val
        self.test_real_data_dir = test_real_data_dir  # only used for the visualization of test data
        self.data_dir = data_dir

        if self.importance_sampling_gradient_norm or self.importance_sampling_root_perc:
            assert weight_base_dir != ''
        self.weight_base_dir = weight_base_dir
        self.use_depth = use_depth
        self.use_dist_to_center = use_dist_to_center

    def __getitem__(self, x):
        use_loc_input_channels = self.use_depth or self.use_dist_to_center

        # Load data from file
        if self.is_training_set:

            if self.for_val or DIFF_TRAINING_CROPS_EACH_EPOCH:  # val crops should always be random
                new_seed = np.random.get_state()[1][1] - x
                if new_seed > 2 ** 32 - 1:  # will result in error if not in the range of (0, 2**32-1)
                    new_seed = 2 ** 32 - 1
                elif new_seed < 0:
                    new_seed = 0
                np.random.seed(new_seed)
                # not np.random.get_state()[1][0] ! otherwise np.random.get_state() does not change!!
                # also plus x! or will still be the same for each epoch!
                # because the init_func starts the worker with a fixed seed!
            else:  # if (not DIFF_TRAINING_CROPS_EACH_EPOCH) and (not self.for_val)
                np.random.seed(x)  # to make sure to generate the same crop when x is the same

            # randomly generate a index for getting a data
            idx = np.random.randint(low=0, high=self.data_list.shape[0])
            [radius_multiplier, rotation, x_flip, y_flip, x_y_swap, noise_type,
             noise_scale, data_name, dim_z, dim_x, dim_y, real_data] = self.data_list[idx, :]
            # Don't convert strings to numbers here --> Do it in the end!
            # or the format might cause problems in the data path

        else:
            [radius_multiplier, rotation, x_flip, y_flip, x_y_swap, noise_type,
             noise_scale, data_name, dim_z, dim_x, dim_y, real_data,
             x_start, y_start, z_start, cropsize_x, cropsize_y, cropsize_z,
             padding_x, padding_y, padding_z] = self.data_list[x, :]

            # convert string to numbers
            x_start, y_start, z_start, cropsize_x, cropsize_y, cropsize_z, padding_x, padding_y, padding_z = \
                int(float(x_start)), int(float(y_start)), int(float(z_start)), \
                int(float(cropsize_x)), int(float(cropsize_y)), int(float(cropsize_z)), \
                int(float(padding_x)), int(float(padding_y)), int(float(padding_z))

        # some variables that will be returned in the end if not changed later in the code
        base_weight_path = ""  # (in that case it will not be used anyways)
        z_i_start, y_i_start, x_i_start = -1, -1, -1
        ground_truth_path = ''
        dont_care_mask = []
        occupancy_path = ''
        snr = 0.
        root_perc = -1
        expected_weight = -1  # weight will not be used if not importance sampling
        out_start_xyz = -1
        out_start_xyz_2 = -1
        distance_to_center = []
        depth_array = []
        cropped_noisy_img_fl = []

        if data_name == 'pure_soil':  # if the data is pure soil scan
            noisy_image_name = noise_type  # if pure soil, the image file name is stored as noise type
            noisy_img_path = join(self.data_dir, data_name, noisy_image_name + '.npy')
            noisy_img = np.load(noisy_img_path, mmap_mode='r')  # (dim_z, 1, dim_x, dim_y)
            if self.super_res_factor == 1:
                ground_truth = np.zeros(noisy_img.shape).astype(np.uint8)
            else:
                target_shape = list(dim * self.super_res_factor for dim in noisy_img.shape)
                target_shape[1] = 1
                ground_truth = np.zeros(target_shape).astype(np.uint8)

            if self.is_training_set:  # when not training, only need the path to occupancy
                if self.dont_care:
                    if self.super_res_factor == 1:
                        dont_care_mask = np.zeros(noisy_img.shape).astype(np.uint8)
                    else:
                        target_shape = list(dim * self.super_res_factor for dim in noisy_img.shape)
                        target_shape[1] = 1
                        dont_care_mask = np.zeros(target_shape).astype(np.uint8)

        else:  # if data is not pure soil data

            if real_data == 'False':
                base_path = join(self.data_dir, data_name,
                                 "r_factor_{0:.2f}"
                                 "/rot_{1}"
                                 "/x_flip_{2}"
                                 "/y_flip_{3}"
                                 "/x_y_swap_{4}"
                                 "/".format(float(radius_multiplier), rotation, x_flip, y_flip, x_y_swap))
                data_path = join(base_path, "{0}x{1}x{2}".format(dim_x, dim_y, dim_z))
                noisy_img_path = join(data_path, "noisy_{0}_{1}_new_.npy".format(int(noise_scale), noise_type))
                if self.is_regression == False:
                    if self.super_res_factor == 1:
                        ground_truth_path = join(data_path, "ground_truth.npy")
                    else:
                        ground_truth_path = join(data_path, "ground_truth_res_{0}x.npy".format(self.super_res_factor))
                else:
                    # load occupancy grid data as ground truth (notice the different resolutions)
                    ground_truth_path = join(base_path,
                                             "{0:d}x{1:d}x{2:d}.npy".format(int(dim_x) * self.super_res_factor,
                                                                            int(dim_y) * self.super_res_factor,
                                                                            int(dim_z) * self.super_res_factor))

                with open(join(data_path, ".snr_{}_{}.txt".format(noise_type, noise_scale)), "r") as myfile:
                    snr = float(myfile.read().replace('\n', ''))
            else:
                if self.test_real_data_dir is None:
                    data_path = join(self.data_dir, data_name)
                    noisy_img_path = join(data_path, 'mri.npy')
                    base_path = join(self.data_dir, data_name,
                                     "r_factor_{0:.2f}"
                                     "/rot_0"
                                     "/x_flip_0"
                                     "/y_flip_0"
                                     "/x_y_swap_0"
                                     "/".format(float(radius_multiplier)))
                    if self.is_regression == False:
                        if self.super_res_factor == 1:
                            ground_truth_path = join(base_path, "{0}x{1}x{2}".format(dim_x, dim_y, dim_z),
                                                     "ground_truth.npy")
                        else:
                            ground_truth_path = join(base_path, "{0}x{1}x{2}".format(dim_x, dim_y, dim_z),
                                                     "ground_truth_res_{0}x.npy".format(self.super_res_factor))
                    else:
                        # load occupancy grid data as ground truth (notice the different resolutions)
                        ground_truth_path = join(base_path,
                                                 "{0:d}x{1:d}x{2:d}.npy".format(int(dim_x) * self.super_res_factor,
                                                                                int(dim_y) * self.super_res_factor,
                                                                                int(dim_z) * self.super_res_factor))

                    snr = 0.

                else:  # only for the visualization of test data
                    file_name = data_name + '_mri.npy'
                    noisy_img_path = join(self.test_real_data_dir, file_name)
                    if self.diff_times:
                        later_file_name = data_name[:-7] + 'later_mri.npy'  # strip('earlier')
                        later_noisy_img_path = join(self.test_real_data_dir, 'later', later_file_name)

            # load the noisy 3D MRI image along with ground truth
            noisy_img = np.load(noisy_img_path, mmap_mode='r')  # (dim_z, 1, dim_x, dim_y)
            if self.test_real_data_dir is not None and self.diff_times:
                later_noisy_img = np.load(later_noisy_img_path, mmap_mode='r')  # (dim_z, 1, dim_x, dim_y)

            if self.test_real_data_dir is None:
                if not self.is_regression:
                    if self.is_training_set:
                        ground_truth = np.load(ground_truth_path, mmap_mode='r')
                    else:  # only the path is needed when not training
                        ground_truth = []
                else:

                    ground_truth_path_npz = join('/'.join(ground_truth_path.split('/')[:-1]),
                                                 "{0:d}x{1:d}x{2:d}_occupancy.npz".format(
                                                     int(dim_x) * self.super_res_factor,
                                                     int(dim_y) * self.super_res_factor,
                                                     int(dim_z) * self.super_res_factor))
                    if self.is_training_set:
                        ground_truth = load_array_from_file(ground_truth_path, ground_truth_path_npz, return_path=False)
                    else:  # only the path is needed when not training
                        ground_truth = []
                        ground_truth_path = load_array_from_file(ground_truth_path, ground_truth_path_npz,
                                                                 return_path=True)

                    # original root_mask: (dim_z, 1, dim_x, dim_y); super-res: (dim_z*res, 1, dim_x*res, dim_y*res)
                    # original occupancy grid: (dim_x*res, dim_y*res, dim_z*res)
                    # make the occupancy grid array the same shape as the root_mask shape: (dim_z*res, 1, dim_x*res, dim_y*res)
                    ground_truth = np.moveaxis(ground_truth, 2, 0)
                    ground_truth = np.expand_dims(ground_truth, axis=1)

                # if need to calculate dont_care mask, need to load occupancy grid
                if (not self.is_regression) and self.dont_care:
                    # if doing voxel-wise classification and dont care flag is required
                    # load the occupancy grid for generating the dont care mask
                    # only possible with virtual data, if real data, need to change the path to the occupancy file

                    base_path = join(self.data_dir, data_name,
                                     "r_factor_{0:.2f}"
                                     "/rot_{1}"
                                     "/x_flip_{2}"
                                     "/y_flip_{3}"
                                     "/x_y_swap_{4}"
                                     "/".format(float(radius_multiplier), rotation, x_flip, y_flip, x_y_swap))
                    occupancy_path = join(base_path, "{0:d}x{1:d}x{2:d}.npy".format(int(dim_x) * self.super_res_factor,
                                                                                    int(dim_y) * self.super_res_factor,
                                                                                    int(dim_z) * self.super_res_factor))

                    occupancy_path_npz = join(base_path, "{0:d}x{1:d}x{2:d}_occupancy.npz".format(
                        int(dim_x) * self.super_res_factor,
                        int(dim_y) * self.super_res_factor,
                        int(dim_z) * self.super_res_factor))

                    if not self.is_training_set:  # when not training, only need the path to occupancy
                        dont_care_mask = []
                        occupancy_path = load_array_from_file(occupancy_path, occupancy_path_npz, return_path=True)

                    else:
                        occupancy = load_array_from_file(occupancy_path, occupancy_path_npz, return_path=False)
                        # transform from shape (dim_x, dim_y, dim_z) to (dim_z, 1, dim_x, dim_y), the same as ground_truth
                        occupancy = np.expand_dims(np.rollaxis(occupancy, 2, 0), axis=1)
                        occupancy_path = ""

        # Sample one crop from the 3D image
        if self.is_training_set:
            if self.training_data_padding != 0:  # pad the training data/gt/dont_care_mask before cropping

                padding = self.training_data_padding
                noisy_img = np.pad(noisy_img, ((padding // 2, padding - padding // 2),
                                               (0, 0),
                                               (padding // 2, padding - padding // 2),
                                               (padding // 2, padding - padding // 2)),
                                   'constant', constant_values=(0,))
                ground_truth = np.pad(ground_truth, (
                    (padding // 2 * self.super_res_factor, (padding - padding // 2) * self.super_res_factor),
                    (0, 0),
                    (padding // 2 * self.super_res_factor, (padding - padding // 2) * self.super_res_factor),
                    (padding // 2 * self.super_res_factor, (padding - padding // 2) * self.super_res_factor)),
                                      'constant', constant_values=(0,))
                if self.dont_care:
                    occupancy = np.pad(occupancy, (
                        (padding // 2 * self.super_res_factor, (padding - padding // 2) * self.super_res_factor),
                        (0, 0),
                        (padding // 2 * self.super_res_factor, (padding - padding // 2) * self.super_res_factor),
                        (padding // 2 * self.super_res_factor, (padding - padding // 2) * self.super_res_factor)),
                                       'constant', constant_values=(0,))

            d = self.depth
            h = self.height
            w = self.width
            if (self.importance_sampling_root_perc is True) and (data_name != 'pure_soil'):
                # only when the data is not pure soil scan, importance sampling based on root percentage is done
                # otherwise just use uniform sampling

                # ******
                # Apply importance weighted random cropping if generating Dataset for training
                # load (if exists) or calculate (if not exists) the weight matrix for this mri image
                stride_w = STRIDE_WEIGHT_MATRIX  # stride used for calculating the weight matrix, default is 1

                weight_mat_dir = join(self.weight_base_dir, 'padding{}'.format(self.training_data_padding), data_name,
                                      'r_factor_{0:.2f}'.format(float(radius_multiplier)),
                                      'rot_{}'.format(int(rotation)),
                                      'superRes_{}'.format(int(self.super_res_factor)),
                                      '{}'.format('occupancy' if self.is_regression else 'root_mask'))

                base_weight_path = join(weight_mat_dir,
                                        'weight_mat_flipSwap0_cropsize_dhw_{}*{}*{}_stride_{}.npy'.format(int(d),
                                                                                                          int(h),
                                                                                                          int(w),
                                                                                                          int(
                                                                                                              stride_w)))
                # Too error-prone to save and load again, for example should not use weight offset
                # when calculating a weight tensor which will be saved, because next time a different offset may be used
                # but if not doing this, it is too slow to run importance sampling!

                if exists(base_weight_path):
                    weight_matrix = np.load(base_weight_path)  # shape: (z,y,x)
                    # Because the matrix in weight path is by default x_flip_0/y_flip_0/x_y_swap_0
                    if int(x_flip) == 1: weight_matrix = np.flip(weight_matrix, 2).copy()
                    if int(y_flip) == 1: weight_matrix = np.flip(weight_matrix, 1).copy()
                    if int(x_y_swap) == 1: weight_matrix = np.swapaxes(weight_matrix, 1, 2).copy()
                    weight_tensor = torch.from_numpy(weight_matrix)  # a ByteTensor
                else:
                    weight_tensor = calculate_weight_mat(ground_truth, self.super_res_factor, d, h, w,
                                                         stride_w)  # shape: (z,y,x)

                    if not exists(weight_mat_dir):
                        makedirs(weight_mat_dir)
                    base_weight_mat = deepcopy(weight_tensor.numpy())

                    # careful!!! the order should be exactly the opposite to loading from disk!!!
                    # To return to the same thing
                    # otherwise x becomes y after swapping!
                    if int(x_y_swap) == 1: base_weight_mat = np.swapaxes(base_weight_mat, 1, 2)
                    if int(y_flip) == 1: base_weight_mat = np.flip(base_weight_mat, 1)
                    if int(x_flip) == 1: base_weight_mat = np.flip(base_weight_mat, 2)

                    np.save(base_weight_path, base_weight_mat)
                    del base_weight_mat

                # convert to float and add crop_weight_offset:
                weight_tensor = weight_tensor.type(torch.FloatTensor)
                weight_tensor /= 255
                weight_tensor += self.crop_weight_offset

                # take a random crop of the 3D image according to the weight
                _z, _y, _x = weight_tensor.shape
                index = multinomial(weight_tensor.flatten(), 1, replacement=True)  # choose one crop center
                expected_weight = (weight_tensor / weight_tensor.sum()).flatten()[
                    index]  # first scale so that all probability adds to 1
                expected_weight = np.float32(expected_weight.item())
                # according to the index of the crop center, calculate the crop location
                z_i, y_i, x_i = index // (_x * _y), index % (_x * _y) // _x, index % (
                    _x * _y) % _x  # location of crop in the weight matrix
                z_i_start, y_i_start, x_i_start = z_i * stride_w, y_i * stride_w, x_i * stride_w  # location of the crop in original image
                z_i_start, y_i_start, x_i_start = int(z_i_start.item()), int(y_i_start.item()), int(x_i_start.item())

            elif self.importance_sampling_gradient_norm:

                # notice: the weight matrix in weight_path is original resolution
                if data_name != "pure_noise":
                    weight_mat_dir = join(self.weight_base_dir, 'gradient_norm_based', data_name,
                                          "r_factor_{0:.2f}"
                                          "/rot_{1}"
                                          "/x_flip_{2}"
                                          "/y_flip_{3}"
                                          "/x_y_swap_{4}"
                                          "/".format(float(radius_multiplier), rotation, x_flip, y_flip, x_y_swap))
                    makedirs(weight_mat_dir, exist_ok=True)

                    base_weight_path = join(weight_mat_dir,
                                            'weight_mat_noise_{}{}_cropsize_dhw_{}*{}*{}.npy'.format(noise_type,
                                                                                                     noise_scale,
                                                                                                     int(d),
                                                                                                     int(h),
                                                                                                     int(w)))
                else:
                    weight_mat_dir = join(self.weight_base_dir, 'gradient_norm_based', data_name)
                    makedirs(weight_mat_dir, exist_ok=True)

                    base_weight_path = join(weight_mat_dir,
                                            'weight_mat_noise_{}_cropsize_dhw_{}*{}*{}.npy'.format(noise_type,
                                                                                                   int(d),
                                                                                                   int(h),
                                                                                                   int(w)))

                if exists(base_weight_path):
                    weight_matrix = np.load(base_weight_path)  # shape: (z,y,x)
                    # Because the matrix in weight path is by default x_flip_0/y_flip_0/x_y_swap_0
                    if int(x_flip) == 1: weight_matrix = np.flip(weight_matrix, 2).copy()
                    if int(y_flip) == 1: weight_matrix = np.flip(weight_matrix, 1).copy()
                    if int(x_y_swap) == 1: weight_matrix = np.swapaxes(weight_matrix, 1, 2).copy()
                    weight_tensor = torch.from_numpy(weight_matrix)  # a FloatTensor
                else:
                    img_dim_z, _, img_dim_x, img_dim_y = noisy_img.shape
                    wt_dim_z = img_dim_z - d + 1
                    wt_dim_x = img_dim_x - w + 1
                    wt_dim_y = img_dim_y - h + 1
                    weight_tensor = torch.ones((wt_dim_z, wt_dim_y, wt_dim_x))  # shape: (dim_z, dim_y, dim_x)

                    # weight_tensor should be float tensor
                    base_weight_mat = weight_tensor.numpy().astype(np.float32)

                    # careful!!! the order should be exactly the opposite to loading from disk!!!
                    # To return to the same thing
                    # otherwise x becomes y after swapping!
                    if int(x_y_swap) == 1: base_weight_mat = np.swapaxes(base_weight_mat, 1, 2)
                    if int(y_flip) == 1: base_weight_mat = np.flip(base_weight_mat, 1)
                    if int(x_flip) == 1: base_weight_mat = np.flip(base_weight_mat, 2)
                    np.save(base_weight_path, base_weight_mat)

                # sample one crop according to the weight tensor
                # the weight_tensor is a FloatTensor (0., 1.), in order to use multinomial()

                # take a random crop of the 3D image according to the weight
                _z, _y, _x = weight_tensor.shape
                index = multinomial(weight_tensor.flatten(), 1, replacement=True)  # choose one crop center
                expected_weight = (weight_tensor / weight_tensor.sum()).flatten()[
                    index]  # first scale so that all probability adds to 1
                expected_weight = np.float32(expected_weight.item())

                # according to the index of the crop center, calculate the crop location
                z_i_start, y_i_start, x_i_start = index // (_x * _y), index % (_x * _y) // _x, index % (
                    _x * _y) % _x  # location of crop in the weight matrix

                z_i_start, y_i_start, x_i_start = int(z_i_start.item()), int(y_i_start.item()), int(x_i_start.item())

            else:  # do uniform sampling instead
                img_dim_z, _, img_dim_x, img_dim_y = noisy_img.shape
                d = self.depth
                h = self.height
                w = self.width
                z_i_start = np.random.randint(img_dim_z - d + 1, size=1)[0]
                y_i_start = np.random.randint(img_dim_y - h + 1, size=1)[0]
                x_i_start = np.random.randint(img_dim_x - w + 1, size=1)[0]

                if self.importance_sampling_root_perc and data_name == 'pure_soil':  # when the data is pure soil
                    # the probability of choosing any crop is the same
                    expected_weight = np.float32(1 / ((img_dim_z - d + 1) * (img_dim_y - h + 1) * (img_dim_x - w + 1)))

            # get the noisy image crop
            cropped_noisy_img = noisy_img[z_i_start:z_i_start + d, :, x_i_start:x_i_start + w, y_i_start:y_i_start + h]

            cropped_ground_truth = ground_truth[
                                   z_i_start * self.super_res_factor:(z_i_start + d) * self.super_res_factor,
                                   :,
                                   x_i_start * self.super_res_factor:(x_i_start + w) * self.super_res_factor,
                                   y_i_start * self.super_res_factor:(y_i_start + h) * self.super_res_factor]

            if self.dont_care:
                occupancy_crop = occupancy[
                                 z_i_start * self.super_res_factor:(z_i_start + d) * self.super_res_factor,
                                 :,
                                 x_i_start * self.super_res_factor:(x_i_start + w) * self.super_res_factor,
                                 y_i_start * self.super_res_factor:(y_i_start + h) * self.super_res_factor]

                # transform from shape (dim_z, 1, dim_x, dim_y) to (dim_x, dim_y, dim_z)
                occupancy_crop = np.rollaxis(np.squeeze(occupancy_crop), 0, 3)

                dont_care_mask = generate_dont_care_mask(cropped_ground_truth, occupancy_crop, dilation=2)

            # during training (importance sampling) calculate weight: the percentage of voxels that contain roots
            num_root_voxels = float((cropped_ground_truth == 255).sum())
            num_voxels = float(cropped_ground_truth.size)
            root_perc = num_root_voxels / num_voxels

            cropped_noisy_img = deepcopy(cropped_noisy_img)
            cropped_ground_truth = deepcopy(cropped_ground_truth)  # because the original array is not writable

            # if specified, add one plane artifact of arbitrary value with random thickness
            if self.add_plane_artifacts:  # add MRI artifact-like noises
                if np.random.uniform(low=0, high=1) <= self.plane_artifacts_frequency:
                    # add the random plane with certain probability
                    dice3 = np.random.randint(len(PLANE_ARTIFACTS), size=1)[0]
                    artifact_type = PLANE_ARTIFACTS[dice3]
                    artifact_intensity = np.random.randint(256, size=1)[0]

                    try:
                        if artifact_type == 'xy_plane':
                            artifact_z_start = np.random.randint(self.depth, size=1)[0]
                            artifact_thickness = min(np.random.randint(1, high=MAX_ARTIFACT_THICKNESS + 1, size=1)[0],
                                                     self.depth - artifact_z_start)
                            cropped_noisy_img[artifact_z_start:artifact_z_start + artifact_thickness, :, :,
                            :] = artifact_intensity
                        elif artifact_type == 'yz_plane':
                            artifact_x_start = np.random.randint(self.width, size=1)[0]
                            artifact_thickness = min(np.random.randint(1, high=MAX_ARTIFACT_THICKNESS + 1, size=1)[0],
                                                     self.width - artifact_x_start)
                            cropped_noisy_img[:, :, artifact_x_start:artifact_x_start + artifact_thickness,
                            :] = artifact_intensity
                        elif artifact_type == 'xz_plane':
                            artifact_y_start = np.random.randint(self.height, size=1)[0]
                            artifact_thickness = min(np.random.randint(1, high=MAX_ARTIFACT_THICKNESS + 1, size=1)[0],
                                                     self.height - artifact_y_start)
                            cropped_noisy_img[:, :, :,
                            artifact_y_start:artifact_y_start + artifact_thickness] = artifact_intensity
                        else:
                            raise ArtifactUndefinedError
                    except ArtifactUndefinedError:
                        print('Type of artifact undefined. Should be one of ["xy_plane", "yz_plane", "xz_plane"]\n')

        else:
            # if not for training, just get one crop of the input according to the start position
            # first pad the noisy image

            x_i_start, y_i_start, z_i_start = x_start, y_start, z_start

            noisy_img = np.pad(noisy_img, ((padding_z // 2, padding_z - padding_z // 2),
                                           (0, 0),
                                           (padding_x // 2, padding_x - padding_x // 2),
                                           (padding_y // 2, padding_y - padding_y // 2)),
                               'constant', constant_values=(0,))
            cropped_noisy_img = noisy_img[z_start:z_start + cropsize_z, :,
                                x_start:x_start + cropsize_x, y_start:y_start + cropsize_y]

            if self.test_real_data_dir is not None and self.diff_times:  # load diff times for test
                later_noisy_img = np.pad(later_noisy_img, ((padding_z // 2, padding_z - padding_z // 2),
                                                           (0, 0),
                                                           (padding_x // 2, padding_x - padding_x // 2),
                                                           (padding_y // 2, padding_y - padding_y // 2)),
                                         'constant', constant_values=(0,))
                cropped_noisy_img_fl = later_noisy_img[z_start:z_start + cropsize_z, :,
                                       x_start:x_start + cropsize_x, y_start:y_start + cropsize_y]

            out_x_start = x_start * self.super_res_factor
            out_y_start = y_start * self.super_res_factor
            out_z_start = z_start * self.super_res_factor

            cropped_ground_truth = []

            # out_start_xyz is both the output start position in the not padded gt (smaller gt)
            # and also the output start position in the padded gt (larger gt) before matching to the size of output
            out_start_xyz = np.array([out_x_start, out_y_start, out_z_start])

            # the corresponding coordinates of gt crop start position in the not padded gt (smaller gt),
            # when the crop is obtained by cropping from the padded gt (larger gt)
            out_start_xyz_2 = np.array([out_x_start - padding_x // 2 * self.super_res_factor,
                                        out_y_start - padding_y // 2 * self.super_res_factor,
                                        out_z_start - padding_z // 2 * self.super_res_factor])

        # convert int8 values to float32, value range(0,1)
        cropped_noisy_img = cropped_noisy_img.astype(np.float32) / 255.
        if self.test_real_data_dir is not None and self.diff_times:  # load diff times for test
            cropped_noisy_img_fl = cropped_noisy_img_fl.astype(np.float32) / 255.

        if self.is_training_set:
            cropped_ground_truth = cropped_ground_truth.astype(np.float32) / 255.

        original_cropped_noisy_img = cropped_noisy_img.copy()
        if self.normalize:
            cropped_noisy_img = normalize(cropped_noisy_img)
            if self.test_real_data_dir is not None and self.diff_times:  # load diff times for test
                cropped_noisy_img_fl = normalize(cropped_noisy_img_fl)

        # remove the second axis of input and gt, where the length is one: from (z,1,x,y) to (z,x,y)
        cropped_noisy_img = np.squeeze(cropped_noisy_img, axis=1)
        if self.test_real_data_dir is not None and self.diff_times:  # load diff times for test
            cropped_noisy_img_fl = np.squeeze(cropped_noisy_img_fl, axis=1)
        if self.is_training_set:
            cropped_ground_truth = np.squeeze(cropped_ground_truth, axis=1)
            if self.dont_care is True:
                dont_care_mask = np.squeeze(dont_care_mask, axis=1)
            if (data_name in DATA_Z_FLIP) and use_loc_input_channels:  # flip the z axis, if use location info channel
                cropped_ground_truth = np.flip(cropped_ground_truth, axis=0).copy()
                if self.dont_care:
                    dont_care_mask = np.flip(dont_care_mask, axis=0).copy()

        # if specified, multiply the input with some random number between (0.6, 1)
        if self.random_scaling:
            random_factor = np.random.uniform(low=RANDOM_SCALING_LOW, high=RANDOM_SCALING_HIGH)
            cropped_noisy_img *= random_factor
            if self.test_real_data_dir is not None and self.diff_times:  # load diff times for test
                # random scaling usually False during testing, so this part not used
                random_factor2 = np.random.uniform(low=RANDOM_SCALING_LOW, high=RANDOM_SCALING_HIGH)
                cropped_noisy_img_fl *= random_factor2

        # rescale the cropped_noisy_img to <=1. Should not happen to test data
        if cropped_noisy_img.max() > 1:
            cropped_noisy_img /= cropped_noisy_img.max()

        if (data_name in DATA_Z_FLIP) and use_loc_input_channels:
            assert self.diff_times is False  # do not train with loc info and multi time points together
            cropped_noisy_img = np.flip(cropped_noisy_img, axis=0).copy()  # flip the z axis

        # if importance sampling: normalize the crop weight with
        # the probability of being sampled by uniform sampling
        if self.importance_sampling_root_perc or self.importance_sampling_gradient_norm:
            img_dim_z, _, img_dim_x, img_dim_y = noisy_img.shape
            d = self.depth
            h = self.height
            w = self.width
            expected_weight /= np.float32(1 / ((img_dim_z - d + 1) * (img_dim_y - h + 1) * (img_dim_x - w + 1)))

        if use_loc_input_channels:
            # generate the location-dependent info crops
            if real_data == 'False' and float(radius_multiplier) == -1:  # virtual roots generated with Nils' method
                data_type_name = None
                data_path = join(self.data_dir, '/'.join(data_name.strip('/').split('/')[:-1]))
                # get the pot information of this data type (data_name)
                data_info_file_path = join(data_path, 'params.txt')
            else:  # virtual roots from Oguz or real roots
                data_type_name = TO_DATA_TYPE_NAME_DICT[data_name]  # todo: add the test data
                data_info_file_path = None

            if self.is_training_set:
                paddings = (self.training_data_padding, self.training_data_padding, self.training_data_padding)
                w, h, d = self.width, self.height, self.depth
            else:
                paddings = (padding_x, padding_y, padding_z)
                w, h, d = cropsize_x, cropsize_y, cropsize_z

            distance_to_center, depth_array = generate_loc_info_arrays(rotation, x_flip, y_flip, x_y_swap,
                                                                       x_i_start, y_i_start, z_i_start,
                                                                       w, h, d,
                                                                       int(dim_z),
                                                                       data_type_name=data_type_name,
                                                                       data_info_file_path=data_info_file_path,
                                                                       paddings=paddings,
                                                                       use_depth=self.use_depth,
                                                                       use_dist_to_center=self.use_dist_to_center)

        if self.diff_times:  # for already combined data in Oguz' dataset,
            if self.test_real_data_dir is None:
                # there isn't any data from a different time point,
                # so just copy the input crop itself
                cropped_noisy_img_fl = deepcopy(cropped_noisy_img)

        ret = {
            'input': cropped_noisy_img,
            'ground_truth': cropped_ground_truth,  # will be [] during non-training mode
            'ground_truth_path': ground_truth_path,  # will only be used during non-training mode
            'occupancy_path': occupancy_path,  # will only be used during non-training mode
            'weight': expected_weight,
            'radius': float(radius_multiplier),
            'rotation': int(rotation),
            'x_flip': int(x_flip),
            'y_flip': int(y_flip),
            'x_y_swap': int(x_y_swap),
            'noise_type': get_noise_type(noise_type),
            'data_index': int(noise_scale),
            'data_name': data_name,
            'slice_count': int(dim_z),
            'width': int(dim_x),
            'height': int(dim_y),
            'real_data': real_data,
            'img_path': noisy_img_path,
            'snr': snr,
            'dont_care_mask': dont_care_mask,  # will be mask array during training mode, otherwise []
            'root_perc': root_perc,
            'out_start_xyz': out_start_xyz,  # the start position of the output crop in the image
            'input_start_x': x_i_start,
            'input_start_y': y_i_start,
            'input_start_z': z_i_start,

            # the following just to make the output attributes the same as the dataManager_combining:
            'base_weight_path': '',  # the path of the weight tensor file for this data point
            'soil_data_path': '',
            'soil_scaling_factor': -1,

            'distance_to_center': distance_to_center,
            'depth_array': depth_array,
            'input_later_time': cropped_noisy_img_fl,
        }

        return ret

    def __len__(self):
        # return self.data_list.shape[0] ## does it make sense?
        return self.length


def normalize(data):  # normalize the input: to a standard normal distribution (0-centered)
    data -= data.mean(axis=(2, 3), keepdims=True)
    stds = data.std(axis=(2, 3), keepdims=True)
    stds[stds == 0.] = 1.
    data /= stds
    data[np.isnan(data)] = 0.

    return data


def normalize_to(arr, min_value, max_value):
    arr = arr.astype(np.float32)
    arr -= arr.min()
    arr /= (arr.max() - arr.min())
    arr *= (max_value - min_value)
    arr += min_value
    return arr


def generate_dont_care_mask(gt, occupancy, dilation=2):
    assert gt.shape[0] == occupancy.shape[2] and gt.shape[2] == occupancy.shape[0] \
           and gt.shape[3] == occupancy.shape[1] and len(occupancy.shape) == 3  # gt: (z,1,x,y)  occ: (x,y,z)

    # the premise is that gt and occupancy have the same shape
    dilated_occupancy = binary_dilation(occupancy, iterations=dilation).astype(occupancy.dtype)  # result is binary

    # make dilated_occupancy the same shape as the root_mask (gt) shape:
    # (dim_z*res, 1, dim_x*res, dim_y*res)
    dilated_occupancy = np.moveaxis(dilated_occupancy, 2, 0)
    dilated_occupancy = np.expand_dims(dilated_occupancy, axis=1)

    dilated_occupancy[gt != 0] = 0

    dont_care_mask = dilated_occupancy.astype('uint8')

    return dont_care_mask


def calculate_weight_mat(ground_truth, super_res_factor, d, h, w, stride_w):
    '''Use convolution to calculate the weight matrix, return as torch.ByteTensor'''
    gt_tensor = torch.from_numpy(ground_truth).permute(1, 0, 3, 2) / 255.  # (1, dim_z, dim_y, dim_x), max=1
    gt_tensor = gt_tensor.type(torch.FloatTensor)
    gt_tensor = torch.unsqueeze(gt_tensor, 0)  # (1, 1, dim_z, dim_y, dim_x)
    srf = super_res_factor
    if super_res_factor != 1:
        # compress the super resolution ground truth to the original image size by convolving
        kernel_reducing = torch.ones([1, 1, 1, 1, 1], dtype=torch.float)
        gt_tensor = F.conv3d(gt_tensor, kernel_reducing, stride=[srf, srf, srf], padding=0)

    kernel_x = torch.ones([1, 1, 1, 1, w], dtype=torch.float) / ((d * h * w) ** (1 / 3.))
    kernel_y = torch.ones([1, 1, 1, h, 1], dtype=torch.float) / ((d * h * w) ** (1 / 3.))
    kernel_z = torch.ones([1, 1, d, 1, 1], dtype=torch.float) / ((d * h * w) ** (1 / 3.))

    ### faster:
    intermediate_tensor = F.conv3d(gt_tensor, kernel_x, stride=[1, 1, stride_w], padding=0)
    intermediate_tensor = F.conv3d(intermediate_tensor, kernel_y, stride=[1, stride_w, 1], padding=0)
    weight_tensor = F.conv3d(intermediate_tensor, kernel_z, stride=[stride_w, 1, 1], padding=0).squeeze()

    weight_tensor = weight_tensor * 255
    weight_tensor = weight_tensor.type(torch.ByteTensor)  # shape: (dim_z, dim_y, dim_x)

    return weight_tensor


def load_array_from_file(npy_path, npz_path, return_path=False):
    if exists(npy_path) and npy_path.endswith(".npy"):
        if return_path:
            valid_path = npy_path
        else:
            loaded_array = np.load(npy_path, mmap_mode='r')

    else:
        if return_path:
            valid_path = npz_path
        else:
            loaded_array = np.load(npz_path, mmap_mode='r')
            for k in loaded_array:
                loaded_array = loaded_array[k]
                # de-compressed npzFile is dict-like, in this case it only contain one array
                break

    if return_path:
        return valid_path
    else:
        return loaded_array


def split_datalists(data_list, train_proportion, test_proportion, validation_proportion,
                    visualization_proportion, save_data=True, environment_path=''):
    """
    Split the data_list into training, test, validation, visualization subsets
    If specified, save the train/val/test data_list separately 

    :param data_list: - the array of virtual data: row -- image slice, column -- data feature
                         columns: [radius_multiplier, rotation, x_flip, y_flip, x_y_swap, noise_type, 
                                         noise_scale, data_name, dim_z, dim_x, dim_y]
                       - the array of real data: row -- real data name (because each real 3D data should have its own name)
    :param train_proportion: the percentage of dataset to be used for training
    :param test_proportion: the percentage of dataset to be used for testing
    :param validation_proportion: the percentage of dataset to be used for validation
    :param visualization_proportion: the percentage of dataset to be used for visualization
    :param save_data: if the splitted datasets should be stored in environment_path
    :param environment_path: the path where the model, datasets and outputs are stored

    """
    assert train_proportion + test_proportion + validation_proportion + visualization_proportion <= 1., \
        'Sum of data proportions must be equal to or smaller than 1.0'

    train_count = int(data_list.shape[0] * train_proportion)
    test_count = int(data_list.shape[0] * test_proportion)
    val_count = int(data_list.shape[0] * validation_proportion)
    visualize_count = int(data_list.shape[0] * visualization_proportion)

    training_data = data_list[0: train_count]
    test_data = data_list[train_count: train_count + test_count]
    validation_data = data_list[train_count + test_count: train_count + test_count + val_count]
    visualization_data = data_list[
                         train_count + test_count + val_count: train_count + test_count + val_count + visualize_count]

    # save the datasets as an npz file
    if save_data == True:
        np.savez(open("{0}/dataset.npz".format(environment_path), 'wb+'), training_data=training_data,
                 test_data=test_data,
                 validation_data=validation_data, visualization_data=visualization_data)
    return training_data, test_data, validation_data, visualization_data


def shuffle_data(data):
    """
    Shuffles data along first axis; somehow, numpy cannot shuffle compelx data.
    :return: sorted data
    """
    indices = list(range(data.shape[0]))
    np.random.shuffle(indices)
    return data[indices, :]


def remove_data_part_column(datalist):  # in older datalists with 13 columns, 'data_part' column index=7
    # Make compatible to the new DataManager, which accepts datalist with only 12 columns (without the 'data_part' column)
    data_part_column = datalist[:, 7]
    data_part_column = data_part_column.astype(int)

    rows_to_keep = (data_part_column == 0)  # loaded dtype is str
    columns_to_keep = np.array(list(range(13)))
    columns_to_keep = np.delete(columns_to_keep, 7, 0)  # remove the 7th column
    datalist = datalist[rows_to_keep, :]
    datalist = datalist[:, columns_to_keep]
    return datalist


def load_previous_data(data_path):
    npz_file = np.load(open("{0}/dataset.npz".format(data_path), 'rb'), allow_pickle=True)
    training_data, test_data, validation_data, visualization_data = npz_file['training_data'], \
                                                                    npz_file['test_data'], \
                                                                    npz_file['validation_data'], \
                                                                    npz_file['visualization_data']
    if len(training_data) != 0:
        if training_data.shape[
            1] == 13:  # this dataset is one of those older ones, which contain 'data_part' column (index=7)
            # Make compatible to the new DataManager: keep rows in which data_part=0, remove other rows, and then remove 'data_part' column
            training_data = remove_data_part_column(training_data)

    if len(test_data) != 0:
        if test_data.shape[1] == 13:
            test_data = remove_data_part_column(test_data)

    if len(validation_data) != 0:
        if validation_data.shape[1] == 13:
            validation_data = remove_data_part_column(validation_data)

    if len(visualization_data) != 0:
        if visualization_data.shape[1] == 13:
            visualization_data = remove_data_part_column(visualization_data)

    return training_data, test_data, validation_data, visualization_data


def get_datalists(load_previous, datalists_dir, environment_path, debug_mode):
    assert load_previous or (len(datalists_dir) != 0), 'Need to either load the datalist of from the specified model ' \
                                                       'or directly provide the directory of the datalist!'
    if load_previous and len(datalists_dir) == 0:
        # if using a previous model and the user didn't define a path to data,
        # then use the original data in the model folder
        training_data, test_data, validation_data, visualization_data = load_previous_data(environment_path)
    elif len(datalists_dir) != 0:
        # if the user defined a path to data,
        # then use the data specified by the user
        training_data, test_data, validation_data, visualization_data = load_previous_data(datalists_dir)
        if not load_previous:
            # if creating a new model and using an existing dataset, then write the dataset in the model folder
            np.savez(open("{0}/dataset.npz".format(environment_path), 'wb+'), training_data=training_data,
                     test_data=test_data,
                     validation_data=validation_data, visualization_data=visualization_data)
            with open("{0}/datasets_from.txt".format(environment_path), 'a') as f:
                f.write(datalists_dir)  # record where the dataset is from

    if debug_mode:  # use only few data points in debugging mode
        if len(training_data) > 50:
            training_data = training_data[:50]
        if len(test_data) > 2:
            test_data = test_data[:2]
        if len(validation_data) > 2:
            validation_data = validation_data[:2]
        if len(visualization_data) > 2:
            visualization_data = visualization_data[:2]

    return training_data, test_data, validation_data, visualization_data


def create_splitted_non_training_datalist(original_datalist, shape_decreases, crop_size, super_res_factor):
    splitted_datalist = None
    for data_point in original_datalist:
        splitted_data_point = cut_non_training_data(data_point, shape_decreases, crop_size,
                                                    super_res_factor=super_res_factor)
        if splitted_datalist is None:
            splitted_datalist = splitted_data_point
        else:
            splitted_datalist = np.concatenate((splitted_datalist, splitted_data_point), axis=0)

    return splitted_datalist


def cut_non_training_data(datapoint, shape_decreases, crop_size, super_res_factor=2,
                          overlaps_after_prediction=(0, 0, 0)):
    """
        Cut the 3D image into reasonable sized parts to input to the network
        :param datapoint: a list representing one data point: [radius_multiplier, rotation,
                          x_flip, y_flip, x_y_swap, noise_type, noise_scale, data_name, dim_z, dim_x, dim_y, real_data]
        :param shape_decreases: a list, the decrease of size on x,y,z dimensions after prediction (when no padding in network
        :param overlap_after_prediction: a list, the expected overlaps on each dimension between crops after prediction
        :param crop_size: a list, the x,y,z length of each crop
        :param super_res_factor: super resolution factor
        :return: a new datalist with additional columns representing the starting point and location of each crop
        """
    shape_decrease_x, shape_decrease_y, shape_decrease_z = shape_decreases
    overlap_after_x, overlap_after_y, overlap_after_z = overlaps_after_prediction
    crop_size_x, crop_size_y, crop_size_z = crop_size  # size of input crop
    _, _, _, _, _, _, _, _, dim_z, dim_x, dim_y, _ = datapoint  # length of each dimension of the input image
    dim_z, dim_x, dim_y = int(dim_z), int(dim_x), int(dim_y)

    out_crop_size_x = crop_size_x - shape_decrease_x
    out_crop_size_y = crop_size_y - shape_decrease_y
    out_crop_size_z = crop_size_z - shape_decrease_z

    # calculate the overlaps between crops before inputting to the network
    overlap_before_x = overlap_after_x / super_res_factor + shape_decrease_x \
                       + (1 - 1 / super_res_factor) * out_crop_size_x
    overlap_before_y = overlap_after_y / super_res_factor + shape_decrease_y \
                       + (1 - 1 / super_res_factor) * out_crop_size_y
    overlap_before_z = overlap_after_z / super_res_factor + shape_decrease_z \
                       + (1 - 1 / super_res_factor) * out_crop_size_z

    # determine the number of crops on each dimension by analysing the output
    num_x = calculate_num_crops(dim_x * super_res_factor, out_crop_size_x, overlap_after_x)
    num_y = calculate_num_crops(dim_y * super_res_factor, out_crop_size_y, overlap_after_y)
    num_z = calculate_num_crops(dim_z * super_res_factor, out_crop_size_z, overlap_after_z)

    # calculate the special overlaps in the output for each dimension,
    # and the special overlaps in the input based on them
    special_overlap_x_after = ((num_x - 1) * (out_crop_size_x - overlap_after_x) + overlap_after_x) + out_crop_size_x \
                              - dim_x * super_res_factor
    special_overlap_y_after = ((num_y - 1) * (out_crop_size_y - overlap_after_y) + overlap_after_y) + out_crop_size_y \
                              - dim_y * super_res_factor
    special_overlap_z_after = ((num_z - 1) * (out_crop_size_z - overlap_after_z) + overlap_after_z) + out_crop_size_z \
                              - dim_z * super_res_factor
    special_overlap_x_before = special_overlap_x_after / super_res_factor + shape_decrease_x \
                               + (1 - 1 / super_res_factor) * out_crop_size_x
    special_overlap_y_before = special_overlap_y_after / super_res_factor + shape_decrease_y \
                               + (1 - 1 / super_res_factor) * out_crop_size_y
    special_overlap_z_before = special_overlap_z_after / super_res_factor + shape_decrease_z \
                               + (1 - 1 / super_res_factor) * out_crop_size_z

    # based on the number of crops, calculate the needed length of each dimension of the input
    padded_dim_x = num_x * crop_size_x - (num_x - 2) * overlap_before_x - special_overlap_x_before
    padded_dim_y = num_y * crop_size_y - (num_y - 2) * overlap_before_y - special_overlap_y_before
    padded_dim_z = num_z * crop_size_z - (num_z - 2) * overlap_before_z - special_overlap_z_before

    # Notice!! the input to the network needs to be padded with shape_decrease_x, shape_decrease_y, shape_decrease_z
    paddings_xyz = (padded_dim_x - dim_x), (padded_dim_y - dim_y), (padded_dim_z - dim_z)

    x_starts = [i * (crop_size_x - overlap_before_x) for i in range(int(num_x - 1))] + [padded_dim_x - crop_size_x]
    y_starts = [i * (crop_size_y - overlap_before_y) for i in range(int(num_y - 1))] + [padded_dim_y - crop_size_y]
    z_starts = [i * (crop_size_z - overlap_before_z) for i in range(int(num_z - 1))] + [padded_dim_z - crop_size_z]

    mesh_z, mesh_y, mesh_x = np.meshgrid(z_starts, y_starts, x_starts, indexing='ij')
    # the order is to make x change first, and then y and last z (actually not necessary)
    mesh = np.stack((mesh_x.flatten(), mesh_y.flatten(), mesh_z.flatten()), axis=1)

    num_rows = mesh.shape[0]
    datapoint_part = np.repeat(np.expand_dims(datapoint, axis=0), num_rows, axis=0)
    cropsize_part = np.repeat(np.expand_dims(crop_size, axis=0), num_rows, axis=0)

    out_crop_size = [out_crop_size_x, out_crop_size_y, out_crop_size_z]
    out_overlaps_xyz = [overlap_after_x, overlap_after_y, overlap_after_z]
    out_special_overlaps_xyz = [special_overlap_x_after, special_overlap_y_after, special_overlap_z_after]

    paddings_xyz_part = np.repeat(np.expand_dims(paddings_xyz, axis=0), num_rows, axis=0)

    return np.concatenate((datapoint_part, mesh, cropsize_part, paddings_xyz_part), axis=1), \
           out_crop_size, out_overlaps_xyz, out_special_overlaps_xyz
    # !!!notice here after concatenation, all the data type becomes string!


def calculate_num_crops(image_side_len, cropsize, overlap):
    """
    Calculate the number of crops on one dimension
    :param image_side_len: the length of the image on that dimension
    :param cropsize: the length of the crop on that dimension
    :param overlap: the overlapping length between crops (before prediction)
    :return: the number of crops
    """
    return np.ceil((image_side_len - cropsize) / (cropsize - overlap) + 1)


def generate_pure_soil_datalists(dir_data, data_name='pure_soil', augment=False, val_ratio=0.1):
    pure_soil_train_datalist = []
    pure_soil_vis_datalist = []
    dir_pure_soil = join(dir_data, data_name)
    print('dir_pure_soil:', dir_pure_soil)
    for file in os.listdir(dir_pure_soil):
        if file.endswith('.npy'):
            rad = -1
            noise_type = file[:-4]  # file name without '.npy'
            noise_scale = -1
            data_name = data_name
            if data_name.endswith('real_soil_noise'):
                real_data = True
            else:
                real_data = False

            if real_data:
                array_shape = np.load(join(dir_pure_soil, file)).shape
                if len(array_shape) == 4:
                    dim_z, _, dim_x, dim_y = array_shape
                elif len(array_shape) == 3:
                    dim_z, dim_x, dim_y = array_shape
            else:
                dim_z, dim_x, dim_y = data_name.strip('/').split('/')[-1].split('x')
                dim_z, dim_x, dim_y = int(dim_z), int(dim_x), int(dim_y)

            if not augment:
                rotation = -1
                x_flip = -1
                y_flip = -1
                xy_swap = -1

                data_point = [rad, rotation, x_flip, y_flip, xy_swap, noise_type, noise_scale,
                              data_name, dim_z, dim_x, dim_y, real_data]
                if real_data:
                    if file == SOIL_DATA_FOR_VALIDATION:  # use one data point in validation
                        pure_soil_vis_datalist.append(data_point)
                    else:
                        pure_soil_train_datalist.append(data_point)
                else:
                    pure_soil_train_datalist.append(data_point)
            else:  # if we will apply augmentation on the soil data
                for r in ROTATIONS:
                    rotation = r
                    for xf in X_FLIPS:
                        x_flip = xf
                        for yf in Y_FLIPS:
                            y_flip = yf
                            for xys in X_Y_SWAPS:
                                xy_swap = xys
                                data_point = [rad, rotation, x_flip, y_flip, xy_swap, noise_type, noise_scale,
                                              data_name, dim_z, dim_x, dim_y, real_data]
                                if real_data:
                                    if file == SOIL_DATA_FOR_VALIDATION:  # use one data point in validation
                                        pure_soil_vis_datalist.append(data_point)
                                    else:
                                        pure_soil_train_datalist.append(data_point)
                                else:
                                    pure_soil_train_datalist.append(data_point)

    pure_soil_train_datalist = shuffle_data(np.array(pure_soil_train_datalist))
    if not real_data:
        num_data = len(pure_soil_train_datalist)
        num_val = int(num_data * val_ratio)
        pure_soil_vis_datalist = deepcopy(pure_soil_train_datalist[:num_val])
        pure_soil_train_datalist = deepcopy(pure_soil_train_datalist[num_val:])
    else:
        pure_soil_vis_datalist = np.array(pure_soil_vis_datalist)
    pure_soil_val_datalist = deepcopy(pure_soil_vis_datalist)

    return pure_soil_train_datalist, pure_soil_val_datalist, pure_soil_vis_datalist


def combine_real_and_virtual_datalists(real_datalist, virtual_datalist, real_ratio=0.4):
    # todo: implement this!
    if len(real_datalist) / len(virtual_datalist) < real_ratio:  # then decrease the number of virtual data
        pass
    return


def add_pure_soil_data(datalist_dir, pure_soil_train_datalist, pure_soil_vis_datalist, output_dir=None, freq_train=0.1):
    # add the pure soil data points to the original training datalist
    num_soil_train = len(pure_soil_train_datalist)
    training_data, test_data, validation_data, visualization_data = load_previous_data(datalist_dir)
    expected_num_soil_train = np.ceil(len(training_data) * freq_train)
    repeats = expected_num_soil_train / num_soil_train
    print('expected_num_soil_train:', expected_num_soil_train)
    print('repeats:', repeats)
    print('original len(training_data):', len(training_data))
    for r in range(int(repeats)):
        training_data = np.append(training_data, pure_soil_train_datalist, axis=0)
        training_data = shuffle_data(training_data)
    num_remainder_soil_train = int(num_soil_train * (repeats - int(repeats)))
    if num_remainder_soil_train > 0:  # if repeats is not an integer
        training_data = np.append(training_data, shuffle_data(pure_soil_train_datalist)[:num_remainder_soil_train, :],
                                  axis=0)
        training_data = shuffle_data(training_data)

    print('Index of pure soil data in training_data', np.where(training_data[:, 7] == 'pure_soil'))
    print(len(training_data))
    print('The resulting frequency of pure soil data in training data:',
          (training_data[:, 7] == 'pure_soil').sum() / len(training_data))

    # add the pure soil data points to the original visualization datalist
    visualization_data = np.append(visualization_data, pure_soil_vis_datalist, axis=0)
    print('Index of pure soil data in visualization_data', np.where(visualization_data[:, 7] == 'pure_soil'))

    if output_dir is not None:
        output_sub_dir = output_dir + '/pure_soil_frequency_{:.1f}'.format(freq_train)
        makedirs(output_sub_dir, exist_ok=True)
        np.savez(open("{0}/dataset.npz".format(output_sub_dir), 'wb+'), training_data=training_data,
                 test_data=test_data,
                 validation_data=validation_data, visualization_data=visualization_data)

    return


def combine_whole_intensity_with_soil(intensity, occupancy, pure_soil, soil_scaling_factor, out_of_pot_mask=None):
    '''

    :param intensity:
    :param occupancy:
    :param pure_soil:
    # :param soil_scaling_factor:
    :param out_of_pot_mask:
    :return:
    '''
    # the shape should be in the order of (z, _, x, y)
    assert len(intensity.shape) == 4
    assert len(pure_soil.shape) == 4
    intensity_z, _, intensity_x, intensity_y = intensity.shape
    soil_z, _, soil_x, soil_y = pure_soil.shape

    # match the shape of pure soil data to the intensity grid
    if intensity_x > soil_x:
        diff_x = intensity_x - soil_x
        # 'wrap' mode to make the padded area similar to the original noise data
        pure_soil = np.pad(pure_soil, pad_width=((0, 0), (0, 0), (diff_x // 2, diff_x - diff_x // 2), (0, 0)),
                           mode='wrap')
    elif intensity_x < soil_x:
        diff_x = soil_x - intensity_x
        pure_soil = pure_soil[:, :, diff_x // 2:-(diff_x - diff_x // 2), :]

    if intensity_y > soil_y:
        diff_y = intensity_y - soil_y
        pure_soil = np.pad(pure_soil, pad_width=((0, 0), (0, 0), (0, 0), (diff_y // 2, diff_y - diff_y // 2)),
                           mode='wrap')
    elif intensity_y < soil_y:
        diff_y = soil_y - intensity_y
        pure_soil = pure_soil[:, :, :, diff_y // 2:-(diff_y - diff_y // 2)]

    if intensity_z > soil_z:
        # extend the z dimension of pure soil data by repeating z
        num_whole_repeats = intensity_z // soil_z
        depth_last_partial_repeat = intensity_z % soil_z
        if num_whole_repeats > 1:
            pure_soil = np.concatenate([pure_soil for _ in range(num_whole_repeats)], axis=0)
        if depth_last_partial_repeat > 0:
            pure_soil = np.concatenate((pure_soil, pure_soil[:depth_last_partial_repeat, :, :]), axis=0)
    elif intensity_z < soil_z:
        diff_z = soil_z - intensity_z
        pure_soil = pure_soil[diff_z // 2:-(diff_z - diff_z // 2), :, :, :]

    # combine the intensity crop and the soil crop
    combined = np.zeros(pure_soil.shape) + pure_soil

    # to make the border between root and soil softer:
    combined = combined * (1 - occupancy / 255.)

    # add noise to root
    noised_intensity, intensity, occupancy = add_noise_to_root_crop(occupancy, intensity)

    combined += noised_intensity

    # imitate pot, set the area out of the pot mask to 0
    if out_of_pot_mask is not None:
        assert intensity.shape == out_of_pot_mask.shape
        # if some root intensity goes out of pot, then do not apply pot mask
        if (intensity > 0).sum() > 0:
            intensity_out_of_pot_perc = (intensity[out_of_pot_mask.astype(bool)] > 0).sum() \
                                        / (intensity > 0).sum()
            if intensity_out_of_pot_perc > 0:
                out_of_pot_mask *= 0
        combined[out_of_pot_mask.astype(bool)] = 0

    return combined


def augment_soil_image(soil_img, soil_rot, soil_x_flip, soil_y_flip, soil_xy_swap):
    assert len(soil_img.shape) == 4  # should be the shape of (z,1,x,y)
    if IF_ROT_SOIL:
        if soil_rot != 0:
            soil_img = ndimage.rotate(soil_img, soil_rot, axes=(2, 3), reshape=False)
            # crop after rotation because rotation introduces empty areas around the original image content!
            _, _, dim_x, dim_y = soil_img.shape
            shorter_dim = min(dim_x, dim_y)
            to_cut = int(np.ceil(shorter_dim / 2 * (1 - 1 / math.sqrt(2))))
            soil_img = soil_img[:, :, to_cut:-to_cut, to_cut:-to_cut]

    if soil_x_flip != 0:
        soil_img = np.flip(soil_img, axis=2)
    if soil_y_flip != 0:
        soil_img = np.flip(soil_img, axis=3)
    if soil_xy_swap != 0:
        soil_img = np.swapaxes(soil_img, 2, 3)
    return soil_img


def imitate_padding_in_soil_crop(cropped_soil, padding, x_i_start, y_i_start, z_i_start,
                                 dim_x, dim_y, dim_z, w, h, d):
    overlapping_z_0 = padding // 2 - z_i_start
    overlapping_z_1 = (z_i_start + d) - (dim_z + padding // 2)

    if overlapping_z_0 > 0:
        cropped_soil[:overlapping_z_0, :, :, :] = 0
    if overlapping_z_1 > 0:
        cropped_soil[-overlapping_z_1:, :, :, :] = 0  # notice it should be negative!!!!

    overlapping_x_0 = padding // 2 - x_i_start
    overlapping_x_1 = (x_i_start + w) - (dim_x + padding // 2)
    if overlapping_x_0 > 0:
        cropped_soil[:, :, :overlapping_x_0, :] = 0
    if overlapping_x_1 > 0:
        cropped_soil[:, :, -overlapping_x_1:, :] = 0

    overlapping_y_0 = padding // 2 - y_i_start
    overlapping_y_1 = (y_i_start + h) - (dim_y + padding // 2)
    if overlapping_y_0 > 0:
        cropped_soil[:, :, :, :overlapping_y_0] = 0
    if overlapping_y_1 > 0:
        cropped_soil[:, :, :, -overlapping_y_1:] = 0

    return cropped_soil


def get_pot_pos_info(data_type_name=None, data_info_file_path=None):
    # get the pot information of this data type (data_type_name)
    if data_type_name is not None:
        data_type = rt.DataGenerationInfo(data_type_name)
        dim_x, dim_y, dim_z = data_type.shape
        pot_pos_x, pot_pos_y = data_type.pot_pos
        pot_pos_x *= dim_x
        pot_pos_y *= dim_y
        pot_rad = data_type.pot_radius
        pot_rad *= dim_x

    elif data_info_file_path is not None:  # read the info from file instead
        with open(data_info_file_path, 'r') as f:
            for l in f.readlines():
                if l.startswith('pot_radius:'):
                    pot_rad = float(l.split(':')[-1].strip())
                elif l.startswith('dim_x'):
                    dim_x = int(l.split(':')[-1].strip())
                elif l.startswith('dim_y'):
                    dim_y = int(l.split(':')[-1].strip())
                elif l.startswith('dim_z'):
                    dim_z = int(l.split(':')[-1].strip())
                elif l.startswith('pot_pos_x'):
                    pot_pos_x = int(l.split(':')[-1].strip())
                elif l.startswith('pot_pos_y'):
                    pot_pos_y = int(l.split(':')[-1].strip())
            pot_rad *= dim_x

    return dim_x, dim_y, dim_z, pot_pos_x, pot_pos_y, pot_rad


def generate_out_of_pot_mask(rot, x_flip, y_flip, xy_swap, data_type_name=None, data_info_file_path=None):
    rot, x_flip, y_flip, xy_swap = int(rot), int(x_flip), int(y_flip), int(xy_swap)

    # # get the pot information of this data type (data_type_name)
    dim_x, dim_y, dim_z, pot_pos_x, pot_pos_y, pot_rad = get_pot_pos_info(data_type_name=data_type_name,
                                                                          data_info_file_path=data_info_file_path)
    if pot_rad > dim_x:
        return np.zeros((dim_z, 1, dim_x, dim_y)).astype(np.uint8)

    # transform the pot parameters according to the augmentation information
    if rot != 0:
        center_x = dim_x // 2
        center_y = dim_y // 2
        a0 = pot_pos_x - center_x
        b0 = pot_pos_y - center_y
        # negative rot because rot here is counterclockwise!
        a1 = a0 * math.cos(math.radians(-rot)) + b0 * math.sin(math.radians(-rot))
        b1 = - a0 * math.sin(math.radians(-rot)) + b0 * math.sin(math.radians(-rot))
        pot_pos_x = a1 + center_x
        pot_pos_y = b1 + center_y
    if x_flip != 0:
        pot_pos_x = dim_x - pot_pos_x
    if y_flip != 0:
        pot_pos_y = dim_y - pot_pos_y
    if xy_swap != 0:
        temp = pot_pos_x
        pot_pos_x = pot_pos_y
        pot_pos_y = temp

    x_range = np.arange(0, dim_x)
    y_range = np.arange(0, dim_y)
    x_mesh, y_mesh = np.meshgrid(x_range, y_range, indexing='ij')
    # a 2D mask of the pot section
    out_of_pot_mask = np.sqrt(np.square(x_mesh - pot_pos_x) + np.square(y_mesh - pot_pos_y)) > pot_rad
    # convert to the shape of (z,1,x,y)
    out_of_pot_mask = np.expand_dims(out_of_pot_mask, axis=0)  # (1,x,y)
    out_of_pot_mask = np.tile(out_of_pot_mask, (dim_z, 1, 1))  # (z,x,y)
    out_of_pot_mask = np.expand_dims(out_of_pot_mask, axis=1)  # (z,1,x,y)

    return out_of_pot_mask.astype(np.uint8)


def generate_loc_info_arrays(rot, x_flip, y_flip, xy_swap, x_i_start, y_i_start, z_i_start,
                             width, height, depth, img_dim_z, data_type_name=None, data_info_file_path=None,
                             paddings=(0, 0, 0), use_depth=False, use_dist_to_center=False):
    '''
    Get location-dependent information arrays as additional input channels:
    distance_to_center: distance to the center axis, normalized with the x dimension
    depth_array: depth from the top of the root image, normalized with the z dimension, has <0 value in padding area

    :param rot:
    :param x_flip:
    :param y_flip:
    :param xy_swap:
    :param x_i_start:
    :param y_i_start:
    :param z_i_start:
    :param width:
    :param height:
    :param depth:
    :param img_dim_z: original z dimension of the whole image without padding
    :param data_type_name:
    :param data_info_file_path:
    :param paddings:
    :return: distance_to_center, depth_array
    '''

    distance_to_center = []  # returns [] if IF_USE_DIS_TO_CENTER is False
    depth_array = []  # returns [] if IF_USE_DEPTH is False

    rot, x_flip, y_flip, xy_swap = int(rot), int(x_flip), int(y_flip), int(xy_swap)
    padding_x, padding_y, padding_z = paddings

    # # get the pot information of this data type (data_type_name)
    dim_x, dim_y, dim_z, pot_pos_x, pot_pos_y, _ = get_pot_pos_info(data_type_name=data_type_name,
                                                                    data_info_file_path=data_info_file_path)

    # update the pot information with the padding
    original_dim_x = dim_x
    original_dim_z = dim_z
    dim_x += padding_x
    dim_y += padding_y
    dim_z += padding_z
    pot_pos_x += padding_x // 2
    pot_pos_y += padding_y // 2

    if use_dist_to_center:
        # transform the pot parameters according to the augmentation information
        if rot != 0:
            center_x = dim_x // 2
            center_y = dim_y // 2
            a0 = pot_pos_x - center_x
            b0 = pot_pos_y - center_y
            # negative rot because rot here is counterclockwise!
            a1 = a0 * math.cos(math.radians(-rot)) + b0 * math.sin(math.radians(-rot))
            b1 = - a0 * math.sin(math.radians(-rot)) + b0 * math.sin(math.radians(-rot))
            pot_pos_x = a1 + center_x
            pot_pos_y = b1 + center_y
        if x_flip != 0:
            pot_pos_x = dim_x - pot_pos_x
        if y_flip != 0:
            pot_pos_y = dim_y - pot_pos_y
        if xy_swap != 0:
            temp = pot_pos_x
            pot_pos_x = pot_pos_y
            pot_pos_y = temp

        # generate the distance to center array
        x_range = np.arange(x_i_start, x_i_start + width)
        y_range = np.arange(y_i_start, y_i_start + height)
        x_mesh, y_mesh = np.meshgrid(x_range, y_range, indexing='ij')
        # calculate the relative distance (divided by the original non-padded dimension of x)
        distance_to_center = np.sqrt(np.square(x_mesh - pot_pos_x) + np.square(y_mesh - pot_pos_y)) / original_dim_x

        # convert to the shape of (z,x,y)
        distance_to_center = distance_to_center.astype(np.float32)
        distance_to_center = np.expand_dims(distance_to_center, axis=0)  # (1,width,height)
        distance_to_center = np.tile(distance_to_center, (depth, 1, 1))  # (depth,width,height)

    if use_depth:
        # generate the relative depth array (divided by original non-padded dim z)
        # depth_array can be negative in padding area
        depth_array = np.ones((depth, width, height)).astype(np.float32)
        data_names_z_flip = []
        for name in DATA_Z_FLIP:
            data_names_z_flip.append(TO_DATA_TYPE_NAME_DICT[name])
        if data_type_name in data_names_z_flip:  # lupine_small and gtk
            z_range = np.arange(img_dim_z + padding_z // 2 - z_i_start - depth,
                                img_dim_z + padding_z // 2 - z_i_start)  # (depth, )
        else:
            z_range = np.arange(z_i_start - padding_z // 2, z_i_start - padding_z // 2 + depth)  # (depth, )
        z_range = np.expand_dims(np.expand_dims(z_range, 1), 2)  # (depth, 1, 1)
        depth_array *= z_range
        depth_array /= original_dim_z

    return distance_to_center, depth_array


def get_random_soil_crop(soil_data_list, data_dir, soil_idx, depth, width, height, get_whole=False, for_val=False):
    '''

    :param soil_data_list:
    :param depth:
    :param width:
    :param height:
    :param get_whole: if to get the whole soil image instead of a crop
    :return:
    '''
    # one random soil data will be chosen from the soil_data_list:
    soil_data = soil_data_list[soil_idx]

    # load the pure soil data
    [_, _, _, _, _, soil_noise_type, _,
     soil_data_name, _, _, _, _] = soil_data

    # if pure soil, the image file name is stored as noise type
    soil_data_path = join(data_dir, soil_data_name, soil_noise_type + '.npy')
    soil_img = np.load(soil_data_path, mmap_mode='r')  # (dim_z, 1, dim_x, dim_y)
    soil_img = soil_img.astype(np.float32)
    # to avoid adding uint8 leading to overflow
    # although actually multiplying with the float scaling factor will already convert it to float
    # and adding float32 with uint8 will automatically result in float32

    # normalize soil data to range (0,255)
    soil_img = normalize_to(soil_img, 0, 255)

    # use soil power instead of soil scaling factor (above)
    # soil power
    if soil_data_name.split('/')[-1].startswith('real'):  # if real soil
        soil_img = normalize_to(soil_img, 0, 1)
        # soil_power = np.random.uniform(SOIL_POWER_MIN, SOIL_POWER_MAX)
        soil_power = np.exp(-0.7 + 1.4 * np.random.rand(1))
        soil_img = np.power(soil_img, soil_power)
        soil_img = normalize_to(soil_img, 0, 255)

    # augment the soil data if indicated by the parameters
    if IF_AUGMENT_SOIL:
        # default is for_val==True: only one type of soil augmentation is used in validation:
        soil_x_flip = 1
        soil_y_flip = 1
        soil_xy_swap = 1
        if not for_val:
            # the soil augmentation for training is the other 7 types
            while soil_x_flip * soil_y_flip * soil_xy_swap == 1:
                # cycle until gets a different type of augmentation
                soil_x_flip = random.choice(X_FLIPS)
                soil_y_flip = random.choice(Y_FLIPS)
                soil_xy_swap = random.choice(X_Y_SWAPS)

        soil_rot = random.choice(ROTATIONS)  # usually not used in augment_soil_image
        soil_img = augment_soil_image(soil_img, soil_rot, soil_x_flip, soil_y_flip, soil_xy_swap)

    if get_whole:
        return soil_img  # warning: no aliasing effect or gradient applied
    else:
        # sample randomly one crop from the soil data
        soil_img_dim_z, _, soil_img_dim_x, soil_img_dim_y = soil_img.shape

        if IF_IMITATE_ALIASING and soil_img_dim_z >= depth * 2:
            ds, hs, ws = depth * 2, height * 2, width * 2
        else:
            ds, hs, ws = depth, height, width
        if not soil_data_name.split('/')[-1].startswith('real'):  # if virtual soil
            soil_z_i_start = np.random.randint(soil_img_dim_z - ds + 1, size=1)[0]
            soil_y_i_start = np.random.randint(soil_img_dim_y - hs + 1, size=1)[0]
            soil_x_i_start = np.random.randint(soil_img_dim_x - ws + 1, size=1)[0]
        else:
            if IF_IMITATE_POT:
                # if real soil and mimicking pot, only sample from the area where there is soil
                soil_z_i_start = np.random.randint(soil_img_dim_z - ds + 1,
                                                   size=1)[0]
                soil_y_i_start = np.random.randint(low=int(soil_img_dim_y / 4),
                                                   high=soil_img_dim_y - int(soil_img_dim_y / 4) - hs + 1,
                                                   size=1)[0]
                soil_x_i_start = np.random.randint(low=int(soil_img_dim_x / 4),
                                                   high=soil_img_dim_x - int(soil_img_dim_x / 4) - ws + 1,
                                                   size=1)[0]
            else:  # if not mimicking pot, do the same as for virtual soil
                soil_z_i_start = np.random.randint(soil_img_dim_z - ds + 1, size=1)[0]
                soil_y_i_start = np.random.randint(soil_img_dim_y - hs + 1, size=1)[0]
                soil_x_i_start = np.random.randint(soil_img_dim_x - ws + 1, size=1)[0]

        # get the noisy soil crop
        cropped_soil = soil_img[soil_z_i_start:soil_z_i_start + ds, :, soil_x_i_start:soil_x_i_start + ws,
                       soil_y_i_start:soil_y_i_start + hs]

        if IF_IMITATE_ALIASING and soil_img_dim_z >= depth * 2:
            # shrink the crop by compressing z and reducing xy
            # 1-compress the depth dimension: take one slice from every 2 slices
            cropped_soil = cropped_soil[::2, :, :, :]

            # 2-reduce the width and height dimensions by averaging
            cropped_soil = block_reduce(cropped_soil, block_size=(1, 1, 2, 2), func=np.mean)

        # apply gradient to the soil
        if APPLY_SOIL_GRADIENT_PROBABILITY > 0:
            dice1 = np.random.uniform()
            if dice1 < APPLY_SOIL_GRADIENT_PROBABILITY:
                gradient = np.arange(cropped_soil.shape[0])
                gradient = normalize_to(gradient, GRADIENT_LOWER, GRADIENT_UPPER)
                gradient = np.expand_dims(np.expand_dims(np.expand_dims(gradient, 1), 2), 3)
                cropped_soil *= gradient  # automatically converted from uint8 to float32

        return cropped_soil, soil_data_path, (soil_x_i_start, soil_y_i_start, soil_z_i_start), \
               (soil_x_flip, soil_y_flip, soil_xy_swap)


def add_noise_to_root_crop(cropped_occupancy, cropped_intensity):
    if IF_ADD_VOIDS:
        # with some probability, clear a small part of root part to mimic the voids in images
        dice2 = np.random.uniform()
        if dice2 < CHANCE_MIMIC_ROOT_VOID:  # 50% chance
            thickness_void = np.random.randint(2, high=4)
            void_start = np.random.randint(cropped_occupancy.shape[0] - thickness_void + 1)
            cropped_occupancy[void_start:void_start + thickness_void] = 0

    # add some simple noise (random normal) to the root intensity grid
    random_noise = np.random.normal(0, scale=1, size=cropped_intensity.shape)
    random_noise = normalize_to(random_noise, ROOT_NOISE_LOWER, ROOT_NOISE_UPPER)
    cropped_intensity = cropped_intensity.astype(np.float32)
    noised_intensity = cropped_intensity * random_noise

    noised_intensity[cropped_occupancy == 0] = 0
    return noised_intensity, cropped_intensity, cropped_occupancy


def combine_intensity_with_soil_crop(cropped_soil, cropped_occupancy, cropped_intensity,
                                     noised_intensity, cropped_mask=None):
    # combine the intensity crop and the soil crop
    combined_crop = np.zeros(cropped_soil.shape) + cropped_soil

    # alpha is the factor that controls the ratio of root and soil in the overlapping area
    alpha = 0.7 + 0.3 * np.random.rand(1)  # using Nils method here
    combined_crop = combined_crop * (1 - cropped_occupancy / 255. * alpha)
    # so that the border between root and soil can be more natural

    combined_crop = combined_crop.astype(np.float32)
    root_brightness = 0.8 + 0.5 * np.random.rand(1)
    combined_crop += noised_intensity * alpha * root_brightness  # using Nils method here
    cropped_noisy_img = combined_crop

    if IF_IMITATE_POT:
        assert cropped_mask is not None
        # if the root in cropped_intensity somehow reaches outside the pot area,
        # do not apply the pot mask!!
        if (cropped_intensity > 0).sum() > 0:
            intensity_out_of_pot_perc = (cropped_intensity[cropped_mask.astype(bool)] > 0).sum() \
                                        / (cropped_intensity > 0).sum()
            if intensity_out_of_pot_perc > 0:
                cropped_mask *= 0  # make the mask useless

        cropped_noisy_img[cropped_mask.astype(bool)] = 0
        # need to convert to bool type!!
        # otherwise is interpreted as index numbers

    return cropped_noisy_img


def pad_data(data, padding, value=0, imitate_aliasing=False):
    if not imitate_aliasing:
        data = np.pad(data, ((padding // 2, padding - padding // 2),
                             (0, 0),
                             (padding // 2, padding - padding // 2),
                             (padding // 2, padding - padding // 2)),
                      'constant', constant_values=(value,))

    else:
        data = np.pad(data, ((padding // 2 * 2, (padding - padding // 2) * 2),
                             (0, 0),
                             (padding // 2 * 2, (padding - padding // 2) * 2),
                             (padding // 2 * 2, (padding - padding // 2) * 2)),
                      'constant', constant_values=(value,))
    return data


def get_crop_loc_importance_sampling_root_perc(padding, data_name, radius_multiplier, rotation, x_flip, y_flip,
                                               x_y_swap, d, h, w, ground_truth, super_res_factor,
                                               crop_weight_offset, weight_base_dir):
    # only when the data is not pure soil scan, importance sampling based on root percentage is done
    # otherwise just use uniform sampling

    # ******
    # Apply importance weighted random cropping if generating Dataset for training
    # load (if exists) or calculate (if not exists) the weight matrix for this mri image
    stride_w = STRIDE_WEIGHT_MATRIX  # stride used for calculating the weight matrix, default is 1

    weight_mat_dir = join(weight_base_dir, 'padding{}'.format(padding), data_name,
                          'r_factor_{0:.2f}'.format(float(radius_multiplier)),
                          'rot_{}'.format(int(rotation)),
                          'superRes_{}'.format(int(super_res_factor)),
                          '{}'.format('root_mask'))

    base_weight_path = join(weight_mat_dir,
                            'weight_mat_flipSwap0_cropsize_dhw_{}*{}*{}_stride_{}.npy'.format(int(d),
                                                                                              int(h),
                                                                                              int(w),
                                                                                              int(
                                                                                                  stride_w)))

    if exists(base_weight_path):
        weight_matrix = np.load(base_weight_path)  # shape: (z,y,x)
        # Because the matrix in weight path is by default x_flip_0/y_flip_0/x_y_swap_0
        # should be the same order as augment_data()
        if int(x_flip) == 1: weight_matrix = np.flip(weight_matrix, 2).copy()
        if int(y_flip) == 1: weight_matrix = np.flip(weight_matrix, 1).copy()
        if int(x_y_swap) == 1: weight_matrix = np.swapaxes(weight_matrix, 1, 2).copy()
        weight_tensor = torch.from_numpy(weight_matrix)  # a ByteTensor
    else:
        weight_tensor = calculate_weight_mat(ground_truth, super_res_factor, d, h, w, stride_w)  # shape: (z,y,x)

        if not exists(weight_mat_dir):
            makedirs(weight_mat_dir)
        base_weight_mat = deepcopy(weight_tensor.numpy())

        # careful!!! the order should be exactly the opposite to augment_data!!! To return to the same thing
        # otherwise x becomes y after swapping!
        if int(x_y_swap) == 1: base_weight_mat = np.swapaxes(base_weight_mat, 1, 2)
        if int(y_flip) == 1: base_weight_mat = np.flip(base_weight_mat, 1)
        if int(x_flip) == 1: base_weight_mat = np.flip(base_weight_mat, 2)
        np.save(base_weight_path, base_weight_mat)
        del base_weight_mat

    # convert to float and add crop_weight_offset:
    weight_tensor = weight_tensor.type(torch.FloatTensor)
    weight_tensor /= 255
    weight_tensor += crop_weight_offset

    # take a random crop of the 3D image according to the weight
    _z, _y, _x = weight_tensor.shape
    index = multinomial(weight_tensor.flatten(), 1, replacement=True)  # choose one crop center
    expected_weight = (weight_tensor / weight_tensor.sum()).flatten()[
        index]  # first scale so that all probability adds to 1
    expected_weight = np.float32(expected_weight.item())
    # according to the index of the crop center, calculate the crop location
    z_i, y_i, x_i = index // (_x * _y), index % (_x * _y) // _x, index % (
        _x * _y) % _x  # location of crop in the weight matrix
    z_i_start, y_i_start, x_i_start = z_i * stride_w, y_i * stride_w, x_i * stride_w  # location of the crop in original image
    z_i_start, y_i_start, x_i_start = int(z_i_start.item()), int(y_i_start.item()), int(x_i_start.item())

    return z_i_start, y_i_start, x_i_start, expected_weight, base_weight_path


def augment_data(data, x_flip, y_flip, x_y_swap):
    assert len(data.shape) == 4 and data.shape[1] == 1  # data should be of the shape (z,1,x,y)
    if int(x_flip) == 1:
        data = np.flip(data, axis=2)
    if int(y_flip) == 1:
        data = np.flip(data, axis=3)
    if int(x_y_swap) == 1:
        data = np.swapaxes(data, 2, 3)

    return data


def get_crop_by_location(data, x_i_start, y_i_start, z_i_start, w, h, d):
    if not IF_IMITATE_ALIASING:
        cropped_data = data[z_i_start:z_i_start + d, :, x_i_start:x_i_start + w,
                       y_i_start:y_i_start + h]
    else:
        cropped_data = data[z_i_start * 2:z_i_start * 2 + d * 2, :, x_i_start * 2:x_i_start * 2 + w * 2,
                       y_i_start * 2:y_i_start * 2 + h * 2]

        # compress the depth dimension: take one slice from every 2 slices
        cropped_data = cropped_data[::2, :, :, :]

        # reduce the width and height dimensions by averaging
        cropped_data = block_reduce(cropped_data, block_size=(1, 1, 2, 2), func=np.mean)
    return cropped_data


def add_plane_artifact(cropped_noisy_img, artifacts_frequency, width, height, depth):
    if np.random.uniform(low=0, high=1) <= artifacts_frequency:
        # add the random plane with certain probability
        dice3 = np.random.randint(len(PLANE_ARTIFACTS), size=1)[0]
        artifact_type = PLANE_ARTIFACTS[dice3]
        artifact_intensity = np.random.randint(256, size=1)[0]
        try:
            if artifact_type == 'xy_plane':
                artifact_z_start = np.random.randint(depth, size=1)[0]
                artifact_thickness = min(np.random.randint(1, high=MAX_ARTIFACT_THICKNESS + 1, size=1)[0],
                                         depth - artifact_z_start)
                cropped_noisy_img[artifact_z_start:artifact_z_start + artifact_thickness, :, :,
                :] = artifact_intensity
            elif artifact_type == 'yz_plane':
                artifact_x_start = np.random.randint(width, size=1)[0]
                artifact_thickness = min(np.random.randint(1, high=MAX_ARTIFACT_THICKNESS + 1, size=1)[0],
                                         width - artifact_x_start)
                cropped_noisy_img[:, :, artifact_x_start:artifact_x_start + artifact_thickness,
                :] = artifact_intensity
            elif artifact_type == 'xz_plane':
                artifact_y_start = np.random.randint(height, size=1)[0]
                artifact_thickness = min(np.random.randint(1, high=MAX_ARTIFACT_THICKNESS + 1, size=1)[0],
                                         height - artifact_y_start)
                cropped_noisy_img[:, :, :,
                artifact_y_start:artifact_y_start + artifact_thickness] = \
                    artifact_intensity
            else:
                raise ArtifactUndefinedError

        except ArtifactUndefinedError:
            print('Type of artifact undefined. Should be one of ["xy_plane", "yz_plane", "xz_plane"]\n')

    return cropped_noisy_img


def get_noise_type(noise_name):
    return '_'.join(noise_name.split('_')[:-1])


def load_crop_from_DatasetManager_combining(idx, dmanagerC, data_dir, load_later_time=False, start_positions=None,
                                            type_of_soil=None, original_soil_idx=None, for_val=False,
                                            weight_base_dir='', use_depth=False, use_dist_to_center=False):
    use_loc_input_channels = use_depth or use_dist_to_center

    start = time()
    # set the initial values of variables to be returned, if not changed later, these values will be returned
    base_weight_path = ""  # (in that case it will not be used anyways)
    ground_truth_path = ''
    noisy_img_path = ''
    soil_data_path = ''
    dont_care_mask = []
    occupancy_path = ''
    snr = 0.
    root_perc = -1
    expected_weight = -1  # weight will not be used if not importance sampling
    out_start_xyz = -1
    cropped_ground_truth = []
    soil_scaling_factor = -1
    soil_noise_type = 'real'
    distance_to_center = []
    depth_array = []
    cropped_noisy_img_fl = []
    soil_idx = ''  # the index of soil in the datalist, which is used to combine with virtual root intensity
    soil_crop_start_position = ()
    soil_augment_params = ()

    # Load data from file
    if dmanagerC.is_training_set:
        [radius_multiplier, rotation, x_flip, y_flip, x_y_swap, noise_type,
         noise_scale, data_name, dim_z, dim_x, dim_y, real_data] = dmanagerC.data_list[idx, :]

        # Don't convert strings to numbers here --> Do it in the end!
        # or the format might cause problems in the data path
    else:
        [radius_multiplier, rotation, x_flip, y_flip, x_y_swap, noise_type,
         noise_scale, data_name, dim_z, dim_x, dim_y, real_data,
         x_start, y_start, z_start, cropsize_x, cropsize_y, cropsize_z,
         padding_x, padding_y, padding_z] = dmanagerC.data_list[idx, :]

        # convert string to numbers
        x_start, y_start, z_start, cropsize_x, cropsize_y, cropsize_z, padding_x, padding_y, padding_z = \
            int(float(x_start)), int(float(y_start)), int(float(z_start)), \
            int(float(cropsize_x)), int(float(cropsize_y)), int(float(cropsize_z)), \
            int(float(padding_x)), int(float(padding_y)), int(float(padding_z))

    if real_data == 'False':
        if float(radius_multiplier) != -1:
            base_path = join(data_dir, data_name,
                             "r_factor_{0:.2f}"
                             "/rot_{1}"
                             "/x_flip_{2}"
                             "/y_flip_{3}"
                             "/x_y_swap_{4}"
                             "/".format(float(radius_multiplier), rotation, x_flip, y_flip, x_y_swap))
            data_path = join(base_path, "{0}x{1}x{2}".format(dim_x, dim_y, dim_z))
            if not IF_IMITATE_ALIASING:
                intensity_path = join(base_path, "{0}x{1}x{2}_intensity.npz".format(dim_x, dim_y, dim_z))
                occupancy_path = join(base_path, "{0}x{1}x{2}_occupancy.npz".format(dim_x, dim_y, dim_z))
            else:
                intensity_path = join(base_path, "{0}x{1}x{2}_intensity.npz".format(int(dim_x) * 2, int(dim_y) * 2,
                                                                                    int(dim_z) * 2))
                occupancy_path = join(base_path, "{0}x{1}x{2}_occupancy.npz".format(int(dim_x) * 2, int(dim_y) * 2,
                                                                                    int(dim_z) * 2))
            noisy_img_path = intensity_path  # will be returned and used when running visualization
            if dmanagerC.super_res_factor == 1:
                ground_truth_path = join(data_path, "ground_truth.npy")
            else:
                ground_truth_path = join(data_path, "ground_truth_res_{0}x.npy".format(dmanagerC.super_res_factor))

            if not (
                not dmanagerC.is_training_set and dmanagerC.combined_noisy_image is not None):  # because loading npz is slow
                # load the intensity grid and ground truth, both datatypes should be uint8
                intensity = np.load(intensity_path)['arr_0']
                occupancy = np.load(occupancy_path)['arr_0']

                # transform intensity grid to the same shape as ground truth
                # original intensity grid: (dim_x, dim_y, dim_z)
                # make the occupancy grid array the same shape as the root_mask shape: (dim_z, 1, dim_x, dim_y)
                intensity = np.moveaxis(intensity, 2, 0)
                intensity = np.expand_dims(intensity, axis=1)
                occupancy = np.moveaxis(occupancy, 2, 0)
                occupancy = np.expand_dims(occupancy, axis=1)

            # only need to load gt when training
            if dmanagerC.is_training_set:
                ground_truth = np.load(ground_truth_path, mmap_mode='r')

            if IF_IMITATE_POT:
                # # get the pot information of this data type (data_name)
                out_of_pot_mask = generate_out_of_pot_mask(rotation, x_flip, y_flip, x_y_swap,
                                                           data_type_name=TO_DATA_TYPE_NAME_DICT[data_name])


        else:
            if not load_later_time:
                # for these virtual root data, the data_name is the relative path
                data_path = join(data_dir, '/'.join(data_name.strip('/').split('/')[:-1]))
                intensity_path = join(data_dir, data_name)
                # todo: for one experiment, load non-aliased data instead?

            else:
                data_path = join(data_dir, '/'.join(data_name.strip('/').split('/')[:-2] + ['stopTime100.00%']))
                intensity_path = join(data_path, data_name.strip('/').split('/')[-1])

            if dmanagerC.dont_care:  # load the occ for generating dont care mask
                if dmanagerC.super_res_factor == 2:
                    occ_for_dont_care = np.load(join(data_path, "ground_truth_res_2x_notThresholded.npy"))
                elif dmanagerC.super_res_factor == 1:
                    occ_for_dont_care = np.load(join(data_path, intensity_path))

                # transform from shape (dim_z, dim_x, dim_y) to (dim_z, 1, dim_x, dim_y)
                occ_for_dont_care = np.expand_dims(occ_for_dont_care, axis=1)

                # augment the occ_for_dont_care according to the augmentation params:
                occ_for_dont_care = augment_data(occ_for_dont_care, x_flip, y_flip, x_y_swap)

            noisy_img_path = intensity_path  # not used except for debug report

            if dmanagerC.super_res_factor == 2:
                ground_truth_path = join(data_path, "ground_truth_res_2x.npy")
            else:
                if COPY_EVERYTHING_FROM_NILS:  # need to do regression in this case, so not thresholded
                    ground_truth_path = join(data_path, "occupancy_small_antialias_256x256x400_notThresholded.npy")
                else:
                    ground_truth_path = join(data_path, "occupancy_small_antialias_256x256x200.npy")
            if not (not dmanagerC.is_training_set
                    and dmanagerC.combined_noisy_image is not None):  # because loading is only necessary in this case
                intensity = np.load(intensity_path)

                # transform from shape (dim_z, dim_x, dim_y) to (dim_z, 1, dim_x, dim_y)
                intensity = np.expand_dims(intensity, axis=1)

                # augment the intensity and occ according to the augmentation params:
                intensity = augment_data(intensity, x_flip, y_flip, x_y_swap)

                occupancy = deepcopy(intensity)  # intensity here is the same as occupancy

            # only need to load gt when training
            if dmanagerC.is_training_set:
                ground_truth = np.load(ground_truth_path, mmap_mode='r')

                # transform from shape (dim_z, dim_x, dim_y) to (dim_z, 1, dim_x, dim_y)
                ground_truth = np.expand_dims(ground_truth, axis=1)

                # augment the ground_truth according to the augmentation params:
                ground_truth = augment_data(ground_truth, x_flip, y_flip, x_y_swap)

            if IF_IMITATE_POT:
                # get the pot information of this data type (data_name)
                info_file_path = join(data_path, 'params.txt')
                out_of_pot_mask = generate_out_of_pot_mask(rotation, x_flip, y_flip, x_y_swap,
                                                           data_info_file_path=info_file_path)
                # print('**************time used for part 1:', time() - start)

        # generate dont care mask if is training loader, only applicable to virtual data
        # too slow to generate for the whole image, do it on the crop; see later
        if dmanagerC.is_training_set:
            if dmanagerC.dont_care:
                occupancy_path = ""

    else:
        data_path = join(data_dir, data_name)
        noisy_img_path = join(data_path, 'mri.npy')
        base_path = join(data_dir, data_name,
                         "r_factor_{0:.2f}"
                         "/rot_0"
                         "/x_flip_0"
                         "/y_flip_0"
                         "/x_y_swap_0"
                         "/".format(float(radius_multiplier)))
        if dmanagerC.super_res_factor == 1:
            ground_truth_path = join(base_path, "{0}x{1}x{2}".format(dim_x, dim_y, dim_z), "ground_truth.npy")
        else:
            ground_truth_path = join(base_path, "{0}x{1}x{2}".format(dim_x, dim_y, dim_z),
                                     "ground_truth_res_{0}x.npy".format(dmanagerC.super_res_factor))

        # load the noisy 3D MRI image along with ground truth
        noisy_img = np.load(noisy_img_path, mmap_mode='r')  # (dim_z, 1, dim_x, dim_y)

        # only need to load gt when training
        if dmanagerC.is_training_set:
            ground_truth = np.load(ground_truth_path, mmap_mode='r')

    # pad the data when specified
    if dmanagerC.is_training_set:  # pad the training data/gt before cropping
        if dmanagerC.training_data_padding != 0:
            padding = dmanagerC.training_data_padding
            if real_data == 'False':
                intensity = pad_data(intensity, padding, value=0, imitate_aliasing=IF_IMITATE_ALIASING)
                occupancy = pad_data(occupancy, padding, value=0, imitate_aliasing=IF_IMITATE_ALIASING)

                if IF_IMITATE_POT:
                    out_of_pot_mask = pad_data(out_of_pot_mask, padding, value=True, imitate_aliasing=False)
            else:
                noisy_img = pad_data(noisy_img, padding, value=0, imitate_aliasing=False)

            # because padding the big gt is slow, only do it when necessary: namely in importance sampling...
            # but need to change the way the cropped gt is got
            if dmanagerC.importance_sampling_root_perc:
                ground_truth = np.pad(ground_truth,
                                      ((padding // 2 * dmanagerC.super_res_factor, (padding - padding // 2)
                                        * dmanagerC.super_res_factor),
                                       (0, 0),
                                       (padding // 2 * dmanagerC.super_res_factor, (padding - padding // 2)
                                        * dmanagerC.super_res_factor),
                                       (padding // 2 * dmanagerC.super_res_factor, (padding - padding // 2)
                                        * dmanagerC.super_res_factor)),
                                      'constant', constant_values=(0,))
                if dmanagerC.dont_care:
                    occ_for_dont_care = np.pad(occ_for_dont_care,
                                               ((padding // 2 * dmanagerC.super_res_factor, (padding - padding // 2)
                                                 * dmanagerC.super_res_factor),
                                                (0, 0),
                                                (padding // 2 * dmanagerC.super_res_factor, (padding - padding // 2)
                                                 * dmanagerC.super_res_factor),
                                                (padding // 2 * dmanagerC.super_res_factor, (padding - padding // 2)
                                                 * dmanagerC.super_res_factor)),
                                               'constant', constant_values=(0,))

        # get one random crop from the intensity grid
        d, h, w = dmanagerC.depth, dmanagerC.height, dmanagerC.width
        if start_positions is not None:
            x_i_start, y_i_start, z_i_start = start_positions
        else:
            if dmanagerC.importance_sampling_root_perc:  # only during training
                # only when the data is not pure soil scan, importance sampling based on root percentage is done
                # otherwise just use uniform sampling
                z_i_start, y_i_start, x_i_start, \
                expected_weight, base_weight_path = get_crop_loc_importance_sampling_root_perc(
                    dmanagerC.training_data_padding,
                    data_name,
                    radius_multiplier, rotation, x_flip, y_flip, x_y_swap,
                    d, h, w, ground_truth, dmanagerC.super_res_factor,
                    dmanagerC.crop_weight_offset, weight_base_dir)

            else:  # do uniform sampling
                if real_data == 'False':
                    img_dim_z, _, img_dim_x, img_dim_y = intensity.shape
                else:
                    img_dim_z, _, img_dim_x, img_dim_y = noisy_img.shape

                if IF_IMITATE_ALIASING:
                    img_dim_z /= 2
                    img_dim_x /= 2
                    img_dim_y /= 2

                z_i_start = np.random.randint(img_dim_z - d + 1, size=1)[0]
                y_i_start = np.random.randint(img_dim_y - h + 1, size=1)[0]
                x_i_start = np.random.randint(img_dim_x - w + 1, size=1)[0]

        # get the noisy image crop
        if real_data == 'False':
            cropped_intensity = get_crop_by_location(intensity, x_i_start, y_i_start, z_i_start, w, h, d)
            cropped_occupancy = get_crop_by_location(occupancy, x_i_start, y_i_start, z_i_start, w, h, d)

            if IF_IMITATE_POT:
                cropped_mask = out_of_pot_mask[z_i_start:z_i_start + d, :, x_i_start:x_i_start + w,
                               y_i_start:y_i_start + h]

            if load_later_time:
                # get a soil crop for the full length root
                # load a different soil data but with the same noise type as cropped_soil
                idx_list_same_type = []
                _soil_i = 0
                for data in dmanagerC.soil_data_list:
                    if get_noise_type(
                            data[5]) == type_of_soil:  # and _soil_i != original_soil_idx:  # avoid the same soil data

                        idx_list_same_type.append(_soil_i)
                    _soil_i += 1
                soil_idx = np.random.choice(idx_list_same_type)
                soil_noise_type = type_of_soil
            else:
                soil_idx = np.random.randint(len(dmanagerC.soil_data_list), size=1)[0]
                soil_noise_type = dmanagerC.soil_data_list[soil_idx, 5]
                soil_noise_type = get_noise_type(soil_noise_type)

            #########################################
            cropped_soil, soil_data_path, soil_crop_start_position, \
            soil_augment_params = get_random_soil_crop(dmanagerC.soil_data_list, data_dir, soil_idx,
                                                       dmanagerC.depth, dmanagerC.width, dmanagerC.height,
                                                       for_val=for_val)

            # imitate the padding in soil crop, if the root crop overlaps with the padding area
            cropped_soil = imitate_padding_in_soil_crop(deepcopy(cropped_soil), dmanagerC.training_data_padding,
                                                        x_i_start, y_i_start, z_i_start,
                                                        int(dim_x), int(dim_y), int(dim_z),
                                                        dmanagerC.width, dmanagerC.height, dmanagerC.depth)

            ### root augmentation part!
            # add more augmentations: discontinuous slices
            if IF_AUGMENT_ROOT:
                noised_intensity, cropped_intensity, cropped_occupancy = add_noise_to_root_crop(cropped_occupancy,
                                                                                                cropped_intensity)
            else:
                noised_intensity = cropped_intensity

            # combine the intensity crop and the soil crop
            if IF_IMITATE_POT:
                # todo: add transparency factor alpha and root scaling factor here
                cropped_noisy_img = combine_intensity_with_soil_crop(cropped_soil, cropped_occupancy,
                                                                     cropped_intensity, noised_intensity, cropped_mask)
            else:
                cropped_noisy_img = combine_intensity_with_soil_crop(cropped_soil, cropped_occupancy,
                                                                     cropped_intensity, noised_intensity)
            #########################################

        else:
            cropped_noisy_img = noisy_img[z_i_start:z_i_start + d, :, x_i_start:x_i_start + w,
                                y_i_start:y_i_start + h]

        if dmanagerC.importance_sampling_root_perc:
            # the cropping should be different because only when importance sampling, the gt is padded
            cropped_ground_truth = ground_truth[
                                   z_i_start * dmanagerC.super_res_factor:(z_i_start + d) * dmanagerC.super_res_factor,
                                   :,
                                   x_i_start * dmanagerC.super_res_factor:(x_i_start + w) * dmanagerC.super_res_factor,
                                   y_i_start * dmanagerC.super_res_factor:(y_i_start + h) * dmanagerC.super_res_factor]
            if dmanagerC.dont_care:
                occ_dc_crop = occ_for_dont_care[
                              z_i_start * dmanagerC.super_res_factor:(z_i_start + d) * dmanagerC.super_res_factor,
                              :,
                              x_i_start * dmanagerC.super_res_factor:(x_i_start + w) * dmanagerC.super_res_factor,
                              y_i_start * dmanagerC.super_res_factor:(y_i_start + h) * dmanagerC.super_res_factor]

                # transform from shape (dim_z, 1, dim_x, dim_y) to (dim_x, dim_y, dim_z)
                occ_dc_crop = np.rollaxis(np.squeeze(occ_dc_crop), 0, 3)

                dont_care_mask = generate_dont_care_mask(cropped_ground_truth, occ_dc_crop, dilation=2)
        else:
            # crop and pad
            gt_dim_z, _, gt_dim_x, gt_dim_y = ground_truth.shape
            gt_padding_left = padding // 2 * dmanagerC.super_res_factor

            # usually should be the second value of the comparison
            gt_z_i_start = max(0, z_i_start * dmanagerC.super_res_factor - gt_padding_left)
            gt_z_i_end = min(gt_dim_z, (z_i_start + d) * dmanagerC.super_res_factor - gt_padding_left)
            crop_pad_z_left = max(0, 0 - (z_i_start * dmanagerC.super_res_factor - gt_padding_left))
            crop_pad_z_right = max(0, ((z_i_start + d) * dmanagerC.super_res_factor - gt_padding_left) - gt_dim_z)

            gt_x_i_start = max(0, x_i_start * dmanagerC.super_res_factor - gt_padding_left)
            gt_x_i_end = min(gt_dim_x, (x_i_start + w) * dmanagerC.super_res_factor - gt_padding_left)
            crop_pad_x_left = max(0, 0 - (x_i_start * dmanagerC.super_res_factor - gt_padding_left))
            crop_pad_x_right = max(0, ((x_i_start + w) * dmanagerC.super_res_factor - gt_padding_left) - gt_dim_x)

            gt_y_i_start = max(0, y_i_start * dmanagerC.super_res_factor - gt_padding_left)
            gt_y_i_end = min(gt_dim_y, (y_i_start + h) * dmanagerC.super_res_factor - gt_padding_left)
            crop_pad_y_left = max(0, 0 - (y_i_start * dmanagerC.super_res_factor - gt_padding_left))
            crop_pad_y_right = max(0, ((y_i_start + h) * dmanagerC.super_res_factor - gt_padding_left) - gt_dim_y)

            cropped_ground_truth = ground_truth[gt_z_i_start:gt_z_i_end,
                                   :,
                                   gt_x_i_start:gt_x_i_end,
                                   gt_y_i_start:gt_y_i_end]
            cropped_ground_truth = np.pad(cropped_ground_truth, ((crop_pad_z_left, crop_pad_z_right),
                                                                 (0, 0),
                                                                 (crop_pad_x_left, crop_pad_x_right),
                                                                 (crop_pad_y_left, crop_pad_y_right)),
                                          'constant', constant_values=(0,))
            if dmanagerC.dont_care:
                occ_dc_crop = occ_for_dont_care[gt_z_i_start:gt_z_i_end,
                              :,
                              gt_x_i_start:gt_x_i_end,
                              gt_y_i_start:gt_y_i_end]
                occ_dc_crop = np.pad(occ_dc_crop, ((crop_pad_z_left, crop_pad_z_right),
                                                   (0, 0),
                                                   (crop_pad_x_left, crop_pad_x_right),
                                                   (crop_pad_y_left, crop_pad_y_right)),
                                     'constant', constant_values=(0,))

                # transform from shape (dim_z, 1, dim_x, dim_y) to (dim_x, dim_y, dim_z)
                occ_dc_crop = np.rollaxis(np.squeeze(occ_dc_crop), 0, 3)

                dont_care_mask = generate_dont_care_mask(cropped_ground_truth, occ_dc_crop, dilation=2)

        # during training (importance sampling) calculate weight: the percentage of voxels that contain roots
        num_root_voxels = float((cropped_ground_truth == 255).sum())
        num_voxels = float(cropped_ground_truth.size)
        root_perc = num_root_voxels / num_voxels

        # if specified, add one plane artifact of arbitrary value with random thickness
        if dmanagerC.add_plane_artifacts:  # add MRI artifact-like noises
            cropped_noisy_img = add_plane_artifact(cropped_noisy_img,
                                                   dmanagerC.plane_artifacts_frequency,
                                                   dmanagerC.width, dmanagerC.height, dmanagerC.depth)

    else:  # non-training
        # if not real data: get one soil data
        if real_data == 'False':
            if ((dmanagerC.combined_noisy_image is None) and (not load_later_time)) \
                    or ((dmanagerC.combined_noisy_image_full_length is None) and load_later_time):
                # because each validation DataManager_combining is only for one single validation data
                # load one pure soil data

                if load_later_time:
                    # get a soil crop for the full length root
                    # load a different soil data but with the same noise type as cropped_soil
                    idx_list_same_type = []
                    _soil_i = 0
                    for data in dmanagerC.soil_data_list:
                        if get_noise_type(data[
                                              5]) == type_of_soil:  # and _soil_i != original_soil_idx:  # avoid the same soil data
                            idx_list_same_type.append(_soil_i)
                        _soil_i += 1
                    soil_idx = np.random.choice(idx_list_same_type)
                    soil_noise_type = type_of_soil
                else:
                    soil_idx = idx % len(dmanagerC.soil_data_list)
                    soil_noise_type = dmanagerC.soil_data_list[soil_idx, 5]
                    soil_noise_type = get_noise_type(soil_noise_type)

                soil_img, soil_data_path, soil_crop_start_position, \
                soil_augment_params = get_random_soil_crop(dmanagerC.soil_data_list, data_dir, soil_idx,
                                                           None, None, None, get_whole=True, for_val=True)

                # if not real data: combine the intensity and soil
                if IF_IMITATE_POT:
                    noisy_img = combine_whole_intensity_with_soil(intensity, occupancy, soil_img,
                                                                  None,
                                                                  out_of_pot_mask=out_of_pot_mask)
                else:
                    noisy_img = combine_whole_intensity_with_soil(intensity, occupancy, soil_img, None)
                # use a class data member to store the combined image!
                if not load_later_time:
                    dmanagerC.combined_noisy_image = noisy_img
                else:
                    dmanagerC.combined_noisy_image_full_length = noisy_img

            else:
                if not load_later_time:
                    noisy_img = dmanagerC.combined_noisy_image
                else:
                    noisy_img = dmanagerC.combined_noisy_image_full_length

                    # only update the class member for non-training,
                    # because one dataManager_combining corresponds to one whole non-training data,
                    # and for it the soil_scaling_factor should be the same!!

        # pad the noisy image and get the crop
        x_i_start, y_i_start, z_i_start = x_start, y_start, z_start

        noisy_img = np.pad(noisy_img, ((padding_z // 2, padding_z - padding_z // 2),
                                       (0, 0),
                                       (padding_x // 2, padding_x - padding_x // 2),
                                       (padding_y // 2, padding_y - padding_y // 2)),
                           'constant', constant_values=(0,))
        cropped_noisy_img = noisy_img[z_i_start:z_i_start + cropsize_z, :,
                            x_i_start:x_i_start + cropsize_x, y_i_start:y_i_start + cropsize_y]

        out_x_start = x_i_start * dmanagerC.super_res_factor
        out_y_start = y_i_start * dmanagerC.super_res_factor
        out_z_start = z_i_start * dmanagerC.super_res_factor

        out_start_xyz = np.array([out_x_start, out_y_start, out_z_start])

    # convert int8 values to float32, value range(0,1)
    cropped_noisy_img = cropped_noisy_img.astype(np.float32) / 255.
    if dmanagerC.is_training_set:
        cropped_ground_truth = cropped_ground_truth.astype(np.float32) / 255.

    if dmanagerC.normalize:
        cropped_noisy_img = normalize(cropped_noisy_img)

    # remove the second axis of input and gt, where the length is one: from (z,1,x,y) to (z,x,y)
    cropped_noisy_img = np.squeeze(cropped_noisy_img, axis=1)
    if dmanagerC.is_training_set:
        cropped_ground_truth = np.squeeze(cropped_ground_truth, axis=1)
        if dmanagerC.dont_care:
            dont_care_mask = np.squeeze(dont_care_mask, axis=1)
        if (data_name in DATA_Z_FLIP) and use_loc_input_channels:
            cropped_ground_truth = np.flip(cropped_ground_truth, axis=0).copy()  # flip the z axis
            if dmanagerC.dont_care:
                dont_care_mask = np.flip(dont_care_mask, axis=0).copy()  # flip the z axis, if use location info channel

    # if specified, multiply the input with some random number between (0.6, 1)
    if dmanagerC.random_scaling:
        random_factor = 0.2 + 1.2 * np.random.rand(1)  # using Nils method here
        cropped_noisy_img *= random_factor

    # rescale the cropped_noisy_img to <=1
    if cropped_noisy_img.max() > 1:
        cropped_noisy_img /= cropped_noisy_img.max()
    if (data_name in DATA_Z_FLIP) and use_loc_input_channels:
        cropped_noisy_img = np.flip(cropped_noisy_img, axis=0).copy()  # flip the z axis

    # if importance sampling: normalize the crop weight
    # with the probability of being sampled by uniform sampling
    if dmanagerC.importance_sampling_root_perc:
        img_dim_z, _, img_dim_x, img_dim_y = ground_truth.shape
        img_dim_z /= dmanagerC.super_res_factor
        img_dim_x /= dmanagerC.super_res_factor
        img_dim_y /= dmanagerC.super_res_factor
        d = dmanagerC.depth
        h = dmanagerC.height
        w = dmanagerC.width
        expected_weight /= np.float32(1 / ((img_dim_z - d + 1) * (img_dim_y - h + 1) * (img_dim_x - w + 1)))

    if use_loc_input_channels:  # todo: incorporate z_padding
        # generate the location-dependent info crops
        if real_data == 'False' and float(radius_multiplier) == -1:  # virtual roots generated with Nils' method
            data_type_name = None
            data_path = join(data_dir, '/'.join(data_name.strip('/').split('/')[:-1]))
            # get the pot information of this data type (data_name)
            data_info_file_path = join(data_path, 'params.txt')
        else:  # virtual roots from Oguz or real roots
            data_type_name = TO_DATA_TYPE_NAME_DICT[data_name]
            data_info_file_path = None

        if dmanagerC.is_training_set:
            paddings = (
            dmanagerC.training_data_padding, dmanagerC.training_data_padding, dmanagerC.training_data_padding)
            w, h, d = dmanagerC.width, dmanagerC.height, dmanagerC.depth
        else:
            paddings = (padding_x, padding_y, padding_z)
            w, h, d = cropsize_x, cropsize_y, cropsize_z

        distance_to_center, depth_array = generate_loc_info_arrays(rotation, x_flip, y_flip, x_y_swap,
                                                                   x_i_start, y_i_start, z_i_start,
                                                                   w, h, d,
                                                                   int(dim_z),
                                                                   data_type_name=data_type_name,
                                                                   data_info_file_path=data_info_file_path,
                                                                   paddings=paddings,
                                                                   use_depth=use_depth,
                                                                   use_dist_to_center=use_dist_to_center)

    ret = {
        'input': cropped_noisy_img,
        'ground_truth': cropped_ground_truth,  # will be [] during non-training mode
        'ground_truth_path': ground_truth_path,  # will only be used during non-training mode
        'occupancy_path': occupancy_path,  # will only be used during non-training mode
        'weight': expected_weight,

        'radius': float(radius_multiplier),
        'rotation': int(rotation),
        'x_flip': int(x_flip),
        'y_flip': int(y_flip),
        'x_y_swap': int(x_y_swap),
        'noise_type': soil_noise_type,  # noise_type does not matter here
        'data_index': int(noise_scale),
        'data_name': data_name,
        'slice_count': int(dim_z),
        'width': int(dim_x),
        'height': int(dim_y),
        'real_data': real_data,
        'img_path': noisy_img_path,
        'snr': snr,
        'dont_care_mask': dont_care_mask,  # will be mask array during training mode, otherwise []
        'root_perc': root_perc,
        'out_start_xyz': out_start_xyz,  # the start position of the output crop in the image if the network has no shape decrease
        'base_weight_path': base_weight_path,  # the path of the weight tensor file for this data point
        'soil_data_path': soil_data_path,
        'soil_scaling_factor': soil_scaling_factor,

        # the following just to make the output attributes the same as the dataManager_combining:
        'input_start_x': x_i_start,
        'input_start_y': y_i_start,
        'input_start_z': z_i_start,

        'distance_to_center': distance_to_center,
        'depth_array': depth_array,

    }

    return ret, soil_idx


if __name__ == '__main__':

    rot = 0
    x_flip = 0

    y_flip = 0
    xy_swap = 0
    x_i_start, y_i_start, z_i_start = 160, 0, 100
    # width, height, depth = 256+43, 256+43, 120+43  #276, 276, 148
    width, height, depth = 60, 60, 60  # 276, 276, 148
    distance_to_center, depth_array = \
        generate_loc_info_arrays(rot, x_flip, y_flip, xy_swap, x_i_start, y_i_start, z_i_start,
                                 width, height, depth, data_type_name='lupine_small', paddings=(43, 43, 43))

    np.save('/home/user/zhaoy/Root_MRI/temp/debug_out_dir/loc_info_test/distance_to_center.npy', distance_to_center)
    np.save('/home/user/zhaoy/Root_MRI/temp/debug_out_dir/loc_info_test/depth_array.npy', depth_array)

