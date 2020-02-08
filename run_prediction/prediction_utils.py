import argparse
import os
from os.path import join
import datetime
from time import time
import sys
import numpy as np
from torch.utils.data import Dataset


def parse_args():
    parser = argparse.ArgumentParser(description='3D Prediction Runner')

    parser.add_argument('-nc', '--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('-cd', '--cuda-device', type=int, default=[0], metavar='CD',
                        help='index of cuda device to use, default=2', nargs="+")

    parser.add_argument('-idd', '--input_data_dir', type=str, default='', help="load data from this dir.")
    parser.add_argument('-od', '--output_dir', type=str, required=True, help="The output will be stored in this "
                                                                             "directory.")
    parser.add_argument('-md', '--model_dir', type=str,
                        help="the directory where the trained model is stored", required=True)

    parser.add_argument('-bs', '--batch_size', type=int, required=True,
                        help="the batch size used for inputting crops to the network (depends on GPU memory. "
                             "batch size only affects the segmentation speed, not the result)")

    args = parser.parse_args()

    return args


def print_remaining_time(loop_start_time, i_batch, data_loader):
    time_passed = time() - loop_start_time if i_batch != 0 else 0
    expected_total_time = float(len(data_loader.dataset)) / float(i_batch) * time_passed if i_batch != 0 else 0
    expected_remaining_time = expected_total_time - time_passed if i_batch != 0 else 0
    expected_remaining_time_str = str(datetime.timedelta(seconds=expected_remaining_time)).split('.')[0]
    time_passed_str = str(datetime.timedelta(seconds=time_passed)).split('.')[0]

    sys.stdout.write("\rMini-batch {0:02d}/{1:02d}, Elapsed time: {2}, "
                     "Remaining time: {3}".format(i_batch,
                                                  len(data_loader)-1,
                                                  time_passed_str,
                                                  expected_remaining_time_str))
    sys.stdout.flush()


def cut_overlappings(out_crop, start_position_xyz, whole_out_dim_zxy, overlaps_xyz, special_overlaps_xyz):
    # dimensions of out_crop is (z,x,y)
    # if overlap value is odd, then split into overlap//2 and overlap-overlap//2, each provided by the crop on each side

    start_x, start_y, start_z = start_position_xyz  # start point of the crop in the whole image

    dim_z, dim_x, dim_y = whole_out_dim_zxy
    overlap_x, overlap_y, overlap_z = overlaps_xyz
    special_overlap_x, special_overlap_y, special_overlap_z = special_overlaps_xyz

    crop_dim_z, crop_dim_x, crop_dim_y = out_crop.size()
    # the positions to cut the original crop
    crop_x_start, crop_x_end = 0, crop_dim_x
    crop_y_start, crop_y_end = 0, crop_dim_y
    crop_z_start, crop_z_end = 0, crop_dim_z

    # there are 4 different situations in terms of each dimension
    # todo: make it work when overlap is 0
    # dimension x
    if start_x == 0:  # the first crop
        if start_x + out_crop.size()[1] * 2 - special_overlap_x == dim_x:  # also the second last crop
            if special_overlap_x > overlap_x:
                crop_x_end = -(special_overlap_x - special_overlap_x // 2)
        else:
            crop_x_end = -(overlap_x - overlap_x // 2)

    elif start_x + out_crop.size()[1] == dim_x:  # the last crop
        crop_x_start = special_overlap_x // 2

    elif start_x + out_crop.size()[1] * 2 - special_overlap_x == dim_x:  # the second last crop
        crop_x_start = overlap_x // 2
        crop_x_end = -(special_overlap_x - special_overlap_x // 2)

    else:  # any other crop positions
        crop_x_start = overlap_x // 2
        crop_x_end = -(overlap_x - overlap_x // 2)

    if crop_x_end == 0:
        crop_x_end = crop_dim_x
    crop_x_start = int(crop_x_start)
    crop_x_end = int(crop_x_end)
    out_crop = out_crop[:, crop_x_start:crop_x_end, :]

    # dimension y
    if start_y == 0:  # the first crop
        if start_y + out_crop.size()[2] * 2 - special_overlap_y == dim_y:  # also the second last crop
            if special_overlap_y > overlap_y:
                crop_y_end = -(special_overlap_y - special_overlap_y // 2)
        else:
            crop_y_end = -(overlap_y - overlap_y // 2)

    elif start_y + out_crop.size()[2] == dim_y:  # the last crop
        crop_y_start = special_overlap_y // 2

    elif start_y + out_crop.size()[2] * 2 - special_overlap_y == dim_y:  # the second last crop
        crop_y_start = overlap_y // 2
        crop_y_end = -(special_overlap_y - special_overlap_y // 2)

    else:  # any other crop positions
        crop_y_start = overlap_y // 2
        crop_y_end = -(overlap_y - overlap_y // 2)

    if crop_y_end == 0:
        crop_y_end = crop_dim_y
    crop_y_start = int(crop_y_start)
    crop_y_end = int(crop_y_end)
    out_crop = out_crop[:, :, crop_y_start:crop_y_end]

    # dimension z
    if start_z == 0:  # the first crop
        if start_z + out_crop.size()[0] * 2 - special_overlap_z == dim_z:  # also the second last crop
            if special_overlap_z > overlap_z:
                crop_z_end = -(special_overlap_z - special_overlap_z // 2)
        else:
            crop_z_end = -(overlap_z - overlap_z // 2)

    elif start_z + out_crop.size()[0] == dim_z:  # the last crop
        crop_z_start = special_overlap_z // 2

    elif start_z + out_crop.size()[0] * 2 - special_overlap_z == dim_z:  # the second last crop
        crop_z_start = overlap_z // 2
        crop_z_end = -(special_overlap_z - special_overlap_z // 2)

    else:  # any other crop positions
        crop_z_start = overlap_z // 2
        crop_z_end = -(overlap_z - overlap_z // 2)

    if crop_z_end == 0:
        crop_z_end = crop_dim_z
    crop_z_start = int(crop_z_start)
    crop_z_end = int(crop_z_end)
    out_crop = out_crop[crop_z_start:crop_z_end, :, :]

    return out_crop, crop_x_start, crop_x_end, crop_y_start, crop_y_end, crop_z_start, crop_z_end


def add_one_crop_to_whole_prediction(outcrop, whole_prediction, start_position_xyz):
    assert len(outcrop.size()) == 3  # assert the outcrop is a 3D tensor
    start_x, start_y, start_z = start_position_xyz
    shape_z, shape_x, shape_y = outcrop.size()
    whole_prediction[start_z:start_z + shape_z, start_x:start_x + shape_x, start_y:start_y + shape_y] += outcrop
    return whole_prediction


def generate_datalist_real(input_data_dir, one_data=False, loaded_data=None):
    """
    Generate the data list given the location of real data
    :return:
    """
    # if user specified the datalist to use
    # assert that the data is real data
    # datasets=[[] for _ in os.listdir(input_data_dir)]

    if input_data_dir=='':
        assert loaded_data is not None
        assert one_data is True

    if loaded_data is None:
        if one_data:  # just one data as input
            assert os.path.isfile(input_data_dir)
            filename_list = [input_data_dir.strip('/').split('/')[-1]]  # the file name of this data
            input_data_dir = '/'+'/'.join(input_data_dir.strip('/').split('/')[:-1])  # the directory of this data
        else:
            filename_list = os.listdir(input_data_dir)
        first_npy_file = True
        for i in range(len(filename_list)):

            if filename_list[i].split('.')[-1] == 'npy':  # the data has to be npy
                data_name = filename_list[i]
                data_i = np.load(os.path.join(input_data_dir, data_name))  # load the npy files
                dim_z, _, dim_x, dim_y = data_i.shape
                visualization_data_i = [[data_name, dim_z, dim_x, dim_y]]

                if first_npy_file is True:
                    visualization_data = visualization_data_i
                    first_npy_file = False
                else:
                    visualization_data = np.concatenate((visualization_data, visualization_data_i), axis=0)
    else:
        dim_z, _, dim_x, dim_y = loaded_data.shape
        assert (len(loaded_data.shape) == 4 and loaded_data.shape[1] == 1), \
            'input data should be of the shape (dim_z, 1, dim_x, dim_y)'
        visualization_data = [['', dim_z, dim_x, dim_y]]
        input_data_dir = ''

    # print("User specified visualization_data:", visualization_data)
    return visualization_data, input_data_dir


def calculate_num_crops(image_side_len, cropsize, overlap):
    """
    Calculate the number of crops on one dimension
    :param image_side_len: the length of the image on that dimension
    :param cropsize: the length of the crop on that dimension
    :param overlap: the overlapping length between crops (before prediction)
    :return: the number of crops
    """
    return np.ceil((image_side_len - cropsize) / (cropsize - overlap) + 1)


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
    _, dim_z, dim_x, dim_y = datapoint  # length of each dimension of the input image
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


def normalize_to(arr, min_value, max_value):
    arr = arr.astype(np.float32)
    arr -= arr.min()
    arr /= (arr.max() - arr.min())
    arr *= (max_value - min_value)
    arr += min_value
    return arr


def normalize(data):  # normalize the input: to a standard normal distribution (0-centered)
    data -= data.mean(axis=(2, 3), keepdims=True)
    stds = data.std(axis=(2, 3), keepdims=True)
    stds[stds == 0.] = 1.
    data /= stds
    data[np.isnan(data)] = 0.

    return data


class DatasetManager(Dataset):
    def __init__(self, data_list, super_resolution_factor=2, normalize=False,
                 diff_times=False, test_data_dir=None, loaded_data=None):
        """
        :param data_list: the array of virtual data: row -- image slice, column -- data feature
                               columns: [radius_multiplier, rotation, x_flip, y_flip, x_y_swap, noise_type,
                                         noise_scale, data_name, dim_z, dim_x, dim_y, real_data]
                          the array of real data: row -- real data name (because each real 3D data should have its own name)
        :param normalize: if the input crop should be normalized to standard normal distribution
        :param diff_times: if also load the root image at a later time point
        :param test_data_dir: the directory of the test data
        """
        super(DatasetManager, self).__init__()

        self.super_res_factor = super_resolution_factor
        self.normalize = normalize
        self.data_list = data_list
        self.diff_times = diff_times
        self.test_data_dir = test_data_dir  # only used for the visualization of test data
        self.loaded_data = loaded_data  # a loaded 3D image can be directly provided

    def __getitem__(self, x):

        # Load data from file
        # todo: eliminate the unnecessary parameters
        [data_name, dim_z, dim_x, dim_y,
         x_start, y_start, z_start, cropsize_x, cropsize_y, cropsize_z,
         padding_x, padding_y, padding_z] = self.data_list[x, :]

        # convert string to numbers
        x_start, y_start, z_start, cropsize_x, cropsize_y, cropsize_z, padding_x, padding_y, padding_z = \
            int(float(x_start)), int(float(y_start)), int(float(z_start)), \
            int(float(cropsize_x)), int(float(cropsize_y)), int(float(cropsize_z)), \
            int(float(padding_x)), int(float(padding_y)), int(float(padding_z))

        # some variables that will be returned in the end if not changed later in the code
        cropped_noisy_img_fl = []

        # only for the visualization of test data
        file_name = data_name
        noisy_img_path = join(self.test_data_dir, file_name)
        if self.diff_times:
            later_file_name = data_name[:-11] + 'later.npy'  # strip('earlier.npy')
            later_noisy_img_path = join(self.test_data_dir, 'later', later_file_name)

        # load the noisy 3D MRI image along with ground truth
        if self.loaded_data is not None:
            noisy_img = self.loaded_data
        else:
            noisy_img = np.load(noisy_img_path, mmap_mode='r')  # (dim_z, 1, dim_x, dim_y)
            self.loaded_data = noisy_img
        assert noisy_img.dtype == np.uint8, 'the dtype of the input data should be uint8'
        assert (len(noisy_img.shape) == 4 and noisy_img.shape[1] == 1), \
            'input data should be of the shape (dim_z, 1, dim_x, dim_y)'
        # normalize to (0, 255)
        noisy_img = normalize_to(noisy_img, 0, 255).astype(np.uint8)
        if self.test_data_dir is not None and self.diff_times:
            later_noisy_img = np.load(later_noisy_img_path, mmap_mode='r')  # (dim_z, 1, dim_x, dim_y)
            assert later_noisy_img.dtype == np.uint8
            # normalize to (0, 255)
            later_noisy_img = normalize_to(later_noisy_img, 0, 255).astype(np.uint8)

        # Sample one crop from the 3D image
        # if not for training, just get one crop of the input according to the start position
        # first pad the noisy image
        noisy_img = np.pad(noisy_img, ((padding_z // 2, padding_z - padding_z // 2),
                                       (0, 0),
                                       (padding_x // 2, padding_x - padding_x // 2),
                                       (padding_y // 2, padding_y - padding_y // 2)),
                           'constant', constant_values=(0,))
        cropped_noisy_img = noisy_img[z_start:z_start + cropsize_z, :,
                                      x_start:x_start + cropsize_x, y_start:y_start + cropsize_y]

        if self.test_data_dir is not None and self.diff_times:  # load diff times for test
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

        # out_start_xyz is both the output start position in the not padded gt (smaller gt)
        # and also the output start position in the padded gt (larger gt) before matching to the size of output
        out_start_xyz = np.array([out_x_start, out_y_start, out_z_start])

        # convert int8 values to float32, value range(0,1)
        cropped_noisy_img = cropped_noisy_img.astype(np.float32) / 255.
        if self.test_data_dir is not None and self.diff_times:  # load diff times for test
            cropped_noisy_img_fl = cropped_noisy_img_fl.astype(np.float32) / 255.

        if self.normalize:
            cropped_noisy_img = normalize(cropped_noisy_img)
            if self.test_data_dir is not None and self.diff_times:  # load diff times for test
                cropped_noisy_img_fl = normalize(cropped_noisy_img_fl)

        # remove the second axis of input and gt, where the length is one: from (z,1,x,y) to (z,x,y)
        cropped_noisy_img = np.squeeze(cropped_noisy_img, axis=1)
        if self.test_data_dir is not None and self.diff_times:  # load diff times for test
            cropped_noisy_img_fl = np.squeeze(cropped_noisy_img_fl, axis=1)

        # rescale the cropped_noisy_img to <=1. Should not happen to test data
        if cropped_noisy_img.max()>1:
            cropped_noisy_img /= cropped_noisy_img.max()

        # print('non_combining &&&&&&&&&&&& unique values of dont_care_mask:', np.unique(dont_care_mask))  # debugging
        ret = {
            'input': cropped_noisy_img,
            'out_start_xyz': out_start_xyz,  # the start position of the output crop in the image
            'input_later_time': cropped_noisy_img_fl,
        }

        return ret

    def __len__(self):
        # return self.data_list.shape[0] ## does it make sense?
        return len(self.data_list)
