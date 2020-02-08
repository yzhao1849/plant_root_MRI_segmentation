import argparse
import datetime
import os
import sys
from os import path

import torch as t

sys.path.append(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
print("Name of base dir", path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
from Utils.constants import *
from time import time


def current_datetime():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def symlink_force(target, link_name):
    try:
        os.symlink(target, link_name)
    except:
        os.remove(link_name)
        os.symlink(target, link_name)


def parse_args():
    parser = argparse.ArgumentParser(description='2D Network Runner')

    parser.add_argument('-lri', '--learning-rate-initial', type=float, default=0.0006, metavar='Initial_LR',
                        help='learning rate, default=0.0006')
    parser.add_argument('-nc', '--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('-cd', '--cuda-device', type=int, default=[0], metavar='CD',
                        help='index of cuda device to use, default=2', nargs="+")

    parser.add_argument('-lt', '--load-trained', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('-lti', '--load-trained-index', type=int, default=5, metavar='LTI',
                        help='The index to load as pretrained model.')

    parser.add_argument('-rw', '--root-weight', type=float, default=1., metavar='RW',
                        help='A single root pixel has x RW times of a soil pixel ')

    parser.add_argument('-wd', '--weight-decay', type=float, default=0.0, metavar='WD',
                        help='L2 loss parameter, default=0')

    parser.add_argument('-n', '--net', type=str, default='Network', metavar='N',
                        help='Specify the name of the Network to use.', required=True)

    parser.add_argument('-bs', '--batch-size', type=int, default=20, help='Batch size, default=20')

    parser.add_argument('-tts', '--total-training-steps',  type=int, default=2000, metavar='TTS',
                        help='Total number of steps, default=2000', required=False)

    parser.add_argument('-si', '--save-interval', type=int, default=1, metavar='N',
                        help='After every nth epoch model will be saved, default=1', required=False)

    parser.add_argument('-ti', '--test-interval', type=int, default=1, metavar='N',
                        help='After every nth epoch model will be tested, default=1')

    parser.add_argument('-vi', '--visualization-interval', type=int, default=1, metavar='VI',
                        help='After every nth batch model will save outputs of some data including the Real MRI, '
                             'default=1.', required=False)

    parser.add_argument('-nps', '--numpy-seed', type=int, default=0,
                        help='Seed for Numpy\'s Pseudo-Random Number Generator, default=0')

    parser.add_argument('-ts', '--pytorch-seed', type=int, default=0,
                        help='Seed for Pytorch\'s Pseudo-Random Generator, default=0')

    parser.add_argument('-d', '--debug', action='store_true', default=False, help='Enables debugging; may slow down.')

    parser.add_argument('-cp', '--confidence_penalty', type=float, default=0., help='Confidence Penalty, default=0')

    parser.add_argument('-sf', '--super_resolution_factor', type=int, default=2, help='Super resolution factor, '
                                                                                       'default=2')
    parser.add_argument('-loss', '--loss-type', type=str, default="nll",
                        help='Loss type, default=\'nll\', Possible values: \'nll\', \'iou\', \'mse\' default=\'nll\'')
    parser.add_argument('-id', '--identification', type=str, default="", help="If not empty, a field with is going to "
                                                                              "be appended to the environment name.")

    parser.add_argument('-gi', '--gradient-application-interval', type=int, default=1, help="After each backward pass "
                                                                                            "of a minibatch, the "
                                                                                            "gradients are applied "                                   
                                                                                            "every GIth minibatch pass,"
                                                                                            " default=1")

    parser.add_argument('-mg', '--max-gradient-norm', type=float, default=0, help="Gradient clipping, if not 0, "
                                                                                  "gradient will be clipped to norm MG."
                                                                                  "default=0")

    parser.add_argument('-nm', '--normalize_all', action='store_true', default=False)

    parser.add_argument('-mod', '--model-output-dir', type=str, required=True,
                        help="set the full path of the directory where the model outputs will be stored")
    parser.add_argument('-csl', '--crop_side_length', type=int, default=60,
                        help="the length of one size of a cubic random crop for training, default: 60")
    parser.add_argument('-apa', '--add_plane_artifacts', action='store_true', default=False,
                        help="if add noises that mimics the plane MRI artifacts on the fly, the direction is randomly chosen")

    parser.add_argument('-dd', '--datalists_dir', type=str, default='',
                        help="if not empty str, load the already existing datalist in the directory")
    parser.add_argument('-ddc', '--datalists_dir_combining', type=str, default='',
                        help="if not empty str, load the already existing datalist in the directory for combination with pure soil data")
    parser.add_argument('-rwl', '--reweight_loss', action='store_true', default=False,
                        help="if reweighting the loss with the importance of input crop, during importance sampling")
    parser.add_argument('-cwo', '--crop_weight_offset', type=float, default=1/254.,
                        help="the offset added to the crop weight, so that any image crop will have a weight larger than 0")
    parser.add_argument('-paf', '--plane_artifacts_frequency', type=float, default=0,
                        help="the probability of adding random plane artifact to each input crop")
    parser.add_argument('-dc', '--dont_care', action='store_true', default=False,
                        help="if true, generate and apply dont_care mask")
    parser.add_argument('-cgdl', '--calculate_gradient_diff_loss', action='store_true', default=False,
                        help="if true, calculate gradient difference loss")

    parser.add_argument('-ntcs', '--non_training_crop_size', type=int, nargs='+', default=[80,80,80],
                        help="specify the crop size on dimension xyz of the input image")
    parser.add_argument('-oap', '--overlaps_after_prediction', type=int, nargs='+', default=[0, 0, 0],
                        help="specify the overlaps on dimension xyz between the output crops")
    parser.add_argument('-ntbs', '--non_training_batch_size', type=int, default=4,
                        help="specify the batch size during validation/visualization/testing")
    parser.add_argument('-stvi', '--save_train_vis_interval', type=int, default=0, metavar='N',
                        help='After every nth batch the training results will be saved, '
                             'default=0, in which case no training result will be saved')
    parser.add_argument('-isrp', '--importance_sampling_root_perc', action='store_true', default=False,
                        help='If true, apply importance sampling based on root voxel percentage of each image crop')
    parser.add_argument('-isgn', '--importance_sampling_gradient_norm', action='store_true', default=False,
                        help='If true, apply importance sampling based on gradient norm of each image crop')

    parser.add_argument('-vwrc', '--val_with_random_crop', action='store_true', default=False,
                        help='if True, do validation on one random crop of each validation data')
    parser.add_argument('-sdd', '--soil_datalist_dir', type=str, default=None,
                        help="the directory of the pure soil datalist")

    parser.add_argument('-rs', '--random_scaling', action='store_true', default=False,
                        help='if True, when loading data, multiply the input crop with some random float between '
                             '[RANDOM_SCALING_LOW, RANDOM_SCALING_HIGH]')

    parser.add_argument('-tl', '--train_length', type=int, default=50000,
                        help='Number of samples in the training dataloader, default=50000')
    parser.add_argument('-vl', '--val_length', type=int, default=2000,
                        help='Number of samples in the validation dataloader, default=50000')
    parser.add_argument('-tcl', '--train_combining_length', type=int, default=50000,
                        help='Number of samples in the training dataloader_combining, default=50000')
    parser.add_argument('-vcl', '--val_combining_length', type=int, default=2000,
                        help='Number of samples in the validation dataloader_combining, default=2000')
    parser.add_argument('-tlri', '--train_loss_reporting_interval', type=int, default=1,
                        help='interval (number of batches) of train loss reporting, default is every batch')
    parser.add_argument('-vai', '--val_interval', type=int, default=60,
                        help='interval (number of training batches) of val loss reporting, default=60', required=False)
    parser.add_argument('-rwdi', '--root_weight_decrease_interval', type=int, default=0,
                        help='interval of root weight decrease, default=60')

    parser.add_argument('-datad', '--data_dir', type=str, required=True,
                        help='the full path of the directory where the data is located')
    parser.add_argument('-wbd', '--weight_base_dir', type=str, default='',
                        help='the full path of the directory where the crop weight matrices should be stored')
    parser.add_argument('-ud', '--use_depth', action='store_true', default=False,
                        help='if True, use depth information as additional input channel')
    parser.add_argument('-udtc', '--use_dist_to_center', action='store_true', default=False,
                        help='if True, use distance to the pot central axis information as additional input channel')
    parser.add_argument('-ult', '--use_later_time', action='store_true', default=False,
                        help='if True, use the image of the same root at a later time point as the second input channel')

    args = parser.parse_args()

    return args


class AutoWrapper(object):
    def __init__(self, *wrapped_objects):
        self.wrapped_objects = wrapped_objects
        first_element = wrapped_objects[0]


def print_and_highlight(info):
    print('*' * 100)
    print("info!", info)
    print('*' * 100)



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
        # print("start crop y")
        if start_y + out_crop.size()[2] * 2 - special_overlap_y == dim_y:  # also the second last crop
            if special_overlap_y > overlap_y:
                crop_y_end = -(special_overlap_y - special_overlap_y // 2)
        else:
            crop_y_end = -(overlap_y - overlap_y // 2)

    elif start_y + out_crop.size()[2] == dim_y:  # the last crop
        crop_y_start = special_overlap_y // 2

    elif start_y + out_crop.size()[2] * 2 - special_overlap_y == dim_y:  # the second last crop
        # print("second last crop y")
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


if __name__ == '__main__':
    out_crop = t.ones((34,34,34))
    start_position_xyz = (0,0,111*2)

    whole_out_dim_zxy = (256, 512, 512)
    overlaps_xyz = (0,0,0)
    special_overlaps_xyz = (32, 32, 16)
    special_overlaps_xyz = (32, 32, 16)
    out_crop, crop_x_start, crop_x_end, crop_y_start, crop_y_end, crop_z_start, crop_z_end = \
        cut_overlappings(out_crop, start_position_xyz, whole_out_dim_zxy, overlaps_xyz, special_overlaps_xyz)
    print('out_crop shape:',out_crop.size())
    print('crop_x_start:',crop_x_start)
    print('crop_x_end:',crop_x_end)
    print('crop_y_start:',crop_y_start)
    print('crop_y_end:',crop_y_end)
    print('crop_z_start:',crop_z_start)
    print('crop_z_end:',crop_z_end)
