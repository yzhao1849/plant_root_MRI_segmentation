import importlib
import json
import shutil

import torch as t
import torch.optim as optim
from tensorboardX import SummaryWriter  # using for visualization of loss and metrics over time
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from DataLoader import *
from Loss import Loss2D as Loss_classification
from Loss_regression import Loss2D as Loss_regression
from Utils import Utils, LogUtil
from Utils.Utils import print_remaining_time, cut_overlappings, add_one_crop_to_whole_prediction
from Utils.Utils import symlink_force, current_datetime
from Utils.constants import *
from Utils.errors import *
import models

start = time()
t.multiprocessing.set_sharing_strategy('file_system')  # in case the limit for num of open file descriptors is low

"""
The script to train any network. For parameters, please see Utils/Utils.py
This script uses tensorboard to visualize the variables
"""

args = Utils.parse_args()
total_training_steps = args.total_training_steps
mini_batch_size = args.batch_size
net_name = args.net
super_resolution_factor = args.super_resolution_factor
weight_decay = args.weight_decay
cuda = not args.no_cuda
load_trained_model = args.load_trained
datalists_dir = args.datalists_dir  # new
datalists_dir_combining = args.datalists_dir_combining  # new
reweight_loss = args.reweight_loss  # new
crop_weight_offset = args.crop_weight_offset  # new
plane_artifacts_frequency = args.plane_artifacts_frequency  # new
dont_care = args.dont_care  # new
calculate_gradient_diff_loss = args.calculate_gradient_diff_loss  # new
non_training_crop_size = args.non_training_crop_size  # new
overlaps_after_prediction = args.overlaps_after_prediction  # new
non_training_batch_size = args.non_training_batch_size  # new
importance_sampling_root_perc = args.importance_sampling_root_perc  # new
importance_sampling_gradient_norm = args.importance_sampling_gradient_norm  # new
val_with_random_crop = args.val_with_random_crop  # new

soil_datalist_dir = args.soil_datalist_dir
soil_datalist_path = join(soil_datalist_dir, 'dataset.npz')
if soil_datalist_path is not None:
    soil_data_lists = np.load(soil_datalist_path)
    soil_data_train_list = soil_data_lists['training_data']
    soil_data_val_list = soil_data_lists['validation_data']
    soil_data_vis_list = soil_data_lists['visualization_data']

random_scaling = args.random_scaling  # new
train_length = args.train_length  # new
val_length = args.val_length  # new
train_combining_length = args.train_combining_length  # new
val_combining_length = args.val_combining_length  # new
train_loss_reporting_interval = args.train_loss_reporting_interval  # new
val_interval = args.val_interval  # new
root_weight_decrease_interval = args.root_weight_decrease_interval  # new
weight_base_dir = args.weight_base_dir
use_loc_input_channels = args.use_depth or args.use_dist_to_center

if importance_sampling_root_perc is True:
    print("******* Applying importance sampling based on root voxel percentage of each image crop *******")
elif importance_sampling_gradient_norm is True:
    print("******* Applying importance sampling based on gradient norm of each image crop *******")
else:
    print("******* Sampling uniformly one crop from each input image *******")

trained_model_index = args.load_trained_index
# index of training batch: add one after each training batch (still increasing in next epoch)
if load_trained_model:
    idx_train_batch = (train_length + train_combining_length) / mini_batch_size * (trained_model_index + 1)
else:
    idx_train_batch = 0  # start with 0

cuda_ind = args.cuda_device
np_seed = args.numpy_seed
torch_seed = args.pytorch_seed
root_weight = args.root_weight
learning_rate_init = args.learning_rate_initial
test_interval = args.test_interval
save_interval = args.save_interval
vis_interval = args.visualization_interval
save_train_vis_interval = args.save_train_vis_interval

if val_with_random_crop:
    # assert vis_interval % test_interval == 0
    # because validation crop visualization is done in the same elif clause as validation.
    # If not True, only the visualization frequency of vis data is specified by vis_interval
    if vis_interval % val_interval != 0:
        print('Warning: vis_interval cannot be divided by val_interval, so val crop results will not be saved!')

loss_type = args.loss_type
confidence_penalty = args.confidence_penalty
debug = args.debug
if debug:
    print("*" * 100)
    print("Debugging mode: some printing, use a few data points only, and assemble whole_debugging_tensor and store")

gradient_application_interval = args.gradient_application_interval
# assert that the total number of training data is divisible by mini_batch_size and gradient_application_interval
assert (train_length + train_combining_length) % mini_batch_size == 0
assert (train_length + train_combining_length) / mini_batch_size % gradient_application_interval == 0

max_gradient_norm = args.max_gradient_norm
print('#'*100+'max_gradient_norm:', max_gradient_norm)

model_outputs_dir = args.model_output_dir  # name of output directory
add_plane_artifacts = args.add_plane_artifacts

crop_side_length = args.crop_side_length  # new
assert crop_side_length == non_training_crop_size[0]
assert crop_side_length == non_training_crop_size[1]
assert crop_side_length == non_training_crop_size[2]

# Remove py extension from Net file
net_name = net_name.replace('.py', '')
# Import the class 'Net' from specified package name
print('***********************net_name:', net_name)
Net = importlib.import_module('.{}'.format(net_name), package='models').Net

np.random.seed(np_seed)
t.manual_seed(torch_seed)

device = t.device("cuda:{}".format(cuda_ind[0]) if cuda else "cpu")

if loss_type != 'mse':  # classification
    Loss = Loss_classification
else:
    Loss = Loss_regression
    print('$$$$$$$$$$$$$ Using MSE as regression loss!')

model = Net()
# Get shape_decreases from the model 
shape_decreases = list(model.calculate_shape_decreases_3D_Net(non_training_crop_size))
if COPY_EVERYTHING_FROM_NILS:
    assert net_name=='UNet_3D_noPadding_Nils'
if net_name=='UNet_3D_noPadding_Nils':
    # mask side: the outer side of the output will not be used in loss calculation or crop assembling
    print('%%%%%%%%%% masking the outer side of the output')
    for i in range(len(shape_decreases)):
        shape_decreases[i] += MASK_SIDE*2
print("$$$$$$$$$$$$$$$$ shape_decreases:", shape_decreases)

loss_calculator = Loss(confidence_penalty=confidence_penalty,
                       super_resolution_factor=super_resolution_factor)

normalize_input_tensor = args.normalize_all  # default is False
# no need to normalize when using pretrained ResNet, because the input image is no longer RGB

environment_name = "{}_rw:x{}_lr:{}_tts:{}_bs:{}".format(
    model.model_type(), root_weight, learning_rate_init, total_training_steps, mini_batch_size)
environment_name = environment_name + '_{}'.format(loss_type) if loss_type is not 'nll' else environment_name
environment_name = environment_name + ('_gi_{}'.format(gradient_application_interval) if
                                       gradient_application_interval != 1 else "")
environment_name = environment_name + ('_mg_{:.2f}'.format(max_gradient_norm) if max_gradient_norm != 0. else "")
environment_name = environment_name + ('_na' if args.normalize_all else "")
environment_name = environment_name + ('_csl_{}'.format(crop_side_length))  #
environment_name = environment_name + ('_apa' if add_plane_artifacts else "")  #
environment_name = environment_name + ('_id_{}'.format(args.identification) if args.identification != "" else "")
environment_name = environment_name + ('_debugging' if args.debug else '')

data_dir = args.data_dir

env_dir = join(model_outputs_dir, environment_name)
os.makedirs(env_dir, exist_ok=True)
logfile = open("{0}/console.txt".format(env_dir), 'a+')  # append plus read
sys.stdout = LogUtil.StreamTee(sys.stdout, logfile)

print("Environment: {0}".format(environment_name))
print("Using device: '{}'".format(device))

models_dir = join(env_dir, 'saved_models', 'trained_models')
weights_viz = join(env_dir, 'Weights')
outputs_dir = join(env_dir, 'denoised_images')
test_outputs_dir = join(env_dir, 'test', 'denoised_images')
train_outputs_dir = join(env_dir, 'denoised_images', 'training_results')
os.makedirs(models_dir, exist_ok=True)
os.makedirs(outputs_dir, exist_ok=True)
os.makedirs(train_outputs_dir, exist_ok=True)

# after deciding env_dir, generate a SummaryWriter for visualization using tensorboardX
writer = SummaryWriter(join(env_dir, 'runs'))

# write model arguments
# json.dumps to convert dict to dict string, json.load to convert opened file to dict
model_arguments_path = join(env_dir, 'model_arguments.txt')
arguments_text = json.dumps(vars(args), indent=4, sort_keys=True)  # .replace("\n", "<br>")
with open(model_arguments_path, 'w') as f:
    f.write(arguments_text)
print(arguments_text)  # .replace("<br>", "\n"))

start = time()

training_datalist = training_datalist_combining = None

if datalists_dir != '':
    print('loading datalist...')
    training_datalist, test_datalist, validation_datalist, visualization_datalist = \
        get_datalists(load_previous=load_trained_model, datalists_dir=datalists_dir, environment_path=env_dir, debug_mode=debug)

if datalists_dir_combining != '':
    print('loading datalist combining...')
    training_datalist_combining, test_datalist_combining, validation_datalist_combining, visualization_datalist_combining = \
        get_datalists(load_previous=load_trained_model, datalists_dir=datalists_dir_combining, environment_path=env_dir,
                      debug_mode=debug)

if datalists_dir == '' and datalists_dir_combining == '':
    print('no given datalist directories...')
    # both empty means loading data from existing model dir or generate new datasets
    training_datalist, test_datalist, validation_datalist, visualization_datalist = \
        get_datalists(load_previous=load_trained_model, datalists_dir=datalists_dir, environment_path=env_dir,
                      debug_mode=debug)

# training_data_padding
if training_datalist is not None:
    val = validation_datalist
else:
    val = validation_datalist_combining

for data in val:
    splitted_datalist, out_cropsize_xyz, _, _ = cut_non_training_data(data, shape_decreases,
                                                                      non_training_crop_size,
                                                                      super_res_factor=super_resolution_factor,
                                                                      overlaps_after_prediction=overlaps_after_prediction)
    training_data_padding = splitted_datalist[0, -1]
    training_data_padding = int(float(training_data_padding))
    break

width, height, depth = crop_side_length, crop_side_length, crop_side_length
# if not combine_soil_with_intensity:
if training_datalist is not None:  # non-combining datalist is given as arg
    training_dataManager1 = DatasetManager(training_datalist, data_dir, width=width, height=height, depth=depth,
                                          super_resolution_factor=super_resolution_factor,
                                          add_plane_artifacts=add_plane_artifacts,
                                          plane_artifacts_frequency=plane_artifacts_frequency,
                                          normalize=normalize_input_tensor,
                                          is_training_set=True,
                                          is_regression=(loss_type == 'mse'),
                                          crop_weight_offset=crop_weight_offset,
                                          dont_care=dont_care,
                                          importance_sampling_root_perc=importance_sampling_root_perc,
                                          importance_sampling_gradient_norm=importance_sampling_gradient_norm,
                                          training_data_padding=training_data_padding,
                                           random_scaling=random_scaling,
                                           length=train_length,
                                           diff_times=args.use_later_time,
                                           for_val=False,
                                           weight_base_dir=weight_base_dir,
                                           use_depth=args.use_depth,
                                           use_dist_to_center=args.use_dist_to_center)
    validation_dataManager1 = DatasetManager(validation_datalist, data_dir, width=crop_side_length, height=crop_side_length,
                                             depth=crop_side_length,
                                             super_resolution_factor=super_resolution_factor,
                                             # add_plane_artifacts=False,  ##
                                             add_plane_artifacts=add_plane_artifacts,
                                             plane_artifacts_frequency=plane_artifacts_frequency,
                                             normalize=normalize_input_tensor,
                                             is_training_set=True,
                                             # not training set, but use the same way of getting random crop from the whole image
                                             is_regression=(loss_type == 'mse'),
                                             # crop_weight_offset=crop_weight_offset,  # no importance sampling for val data
                                             dont_care=dont_care,  # right ???
                                             importance_sampling_root_perc=False,
                                             # always use uniform sampling for validation data
                                             importance_sampling_gradient_norm=False,  ##
                                             training_data_padding=training_data_padding,
                                             ## use the same padding as the training data
                                             # random_scaling=False, # no random scaling during validation
                                             random_scaling=random_scaling,
                                             length=val_length,
                                             diff_times=args.use_later_time,
                                             for_val=True,  # for making the validation crop random
                                             weight_base_dir=weight_base_dir,
                                             use_depth=args.use_depth,
                                             use_dist_to_center=args.use_dist_to_center
                                             )

if training_datalist_combining is not None:  # combining datalist is given as arg
    training_dataManager2 = DatasetManager_combining(training_datalist_combining, data_dir,
                                                     width=width, height=height, depth=depth,
                                                    super_resolution_factor=super_resolution_factor,
                                                    add_plane_artifacts=add_plane_artifacts,
                                                    plane_artifacts_frequency=plane_artifacts_frequency,
                                                    normalize=normalize_input_tensor,
                                                    is_training_set=True,
                                                    crop_weight_offset=crop_weight_offset,
                                                    importance_sampling_root_perc=importance_sampling_root_perc,
                                                    training_data_padding=training_data_padding,
                                                    soil_data_list = soil_data_train_list,
                                                     random_scaling=random_scaling,
                                                     length=train_combining_length,
                                                     diff_times=args.use_later_time,
                                                     for_val=False,
                                                     dont_care=dont_care,
                                                     weight_base_dir=weight_base_dir,
                                                     use_depth=args.use_depth,
                                                     use_dist_to_center=args.use_dist_to_center)
    validation_dataManager2 = DatasetManager_combining(validation_datalist_combining, data_dir,
                                                       width=crop_side_length,
                                                       height=crop_side_length, depth=crop_side_length,
                                                       super_resolution_factor=super_resolution_factor,
                                                       add_plane_artifacts=add_plane_artifacts,
                                                       plane_artifacts_frequency=plane_artifacts_frequency,
                                                       normalize=normalize_input_tensor,
                                                       is_training_set=True,
                                                       importance_sampling_root_perc=False,
                                                       training_data_padding=training_data_padding,
                                                       soil_data_list=soil_data_val_list,
                                                       random_scaling=random_scaling,
                                                       length=val_combining_length,
                                                       diff_times=args.use_later_time,
                                                       for_val=True,
                                                       dont_care=dont_care,
                                                       weight_base_dir=weight_base_dir,
                                                       use_depth=args.use_depth,
                                                       use_dist_to_center=args.use_dist_to_center)

if (training_datalist is not None) and (training_datalist_combining is not None):
    training_dataManager = Concat_datasets((training_dataManager1, training_dataManager2), fixed_mix_map=False)
    # no need to set fixed_mix_map to true, can already mix well when shuffle is true in dataloader

    validation_dataManager = Concat_datasets((validation_dataManager1, validation_dataManager2), fixed_mix_map=False)

elif training_datalist is not None:
    training_dataManager = training_dataManager1
    validation_dataManager = validation_dataManager1
else:
    training_dataManager = training_dataManager2
    validation_dataManager = validation_dataManager2

worker_ind = 0
training_data_loader = DataLoader(training_dataManager, batch_size=mini_batch_size, num_workers=DATALOADER_NUM_WORKERS,
                                  shuffle=True,
                                  worker_init_fn=lambda x: np.random.seed(x + np_seed + worker_ind))

worker_ind = 32
val_dataloader = DataLoader(validation_dataManager, batch_size=non_training_batch_size, shuffle=True,
                            num_workers=DATALOADER_NUM_WORKERS,
                            worker_init_fn=lambda x: np.random.seed(x + np_seed + worker_ind))

# Approximate accuracy parameters
threshold_begin = 15.
threshold_end = 240.
threshold_count = 3
threshold_step_size = (threshold_end - threshold_begin) / (threshold_count - 1)
thresholds = [str(threshold_begin + threshold_index * threshold_step_size) for threshold_index in
              range(threshold_count)]
dilations = list(range(0, 9, 2))

adam_optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), learning_rate_init,
                            weight_decay=weight_decay)


def load_model(epoch):
    # load state dict immediately loads to GPU so we first load to CPU; then move the GPU
    state_dict = t.load('{0}/model_{1}.tm'.format(models_dir, epoch), map_location='cpu')
    model.load_state_dict(state_dict['net'])
    adam_optimizer.load_state_dict(state_dict['optimizer'])
    if cuda:
        for state in adam_optimizer.state.values():
            for k, v in state.items():
                if t.is_tensor(v):
                    state[k] = v.to(device)


if load_trained_model:
    load_model(epoch=trained_model_index)
    print('**************** Loading trained model: model_{}.tm'.format(trained_model_index))

if cuda and len(cuda_ind) > 1:
    model = t.nn.DataParallel(model, device_ids=cuda_ind)

model = model.to(device)

training_start_step = 0 if not load_trained_model else trained_model_index + 1

vis_dir = join(model_outputs_dir, environment_name)

mini_loss_ctr = 0

start = time()


def remove_first_axis(tensor):
    return tensor[0]


def clip_gradient():
    if max_gradient_norm != 0:
        return clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_gradient_norm)
    return 0


def run_network(step, mode):
    """

    :param step: the 'step'th ep och
    :param mode: 'Training', 'Test', 'Validation', 'Visualization'
    :return:
    """
    global root_weight

    training = mode == 'Training'
    test = mode == 'Test'
    validation = mode == 'Validation'
    visualization = mode == 'Visualization'

    print("\n\n{0}, {1}".format(mode, current_datetime()))

    if training:
        model.train()
        data_loader = training_data_loader
        optimizer = adam_optimizer  # if step < sgd_start_epoch or sgd_start_epoch < 0 else sgd_optimizer
        print("Using optimizer {} with learning rate {}".format(type(optimizer).__name__,
                                                                optimizer.param_groups[0]['lr']))

        sum_total_loss = 0
        sum_soil_loss = 0
        sum_root_loss = 0
        total_weight_root_pixels = 0
        total_weight_soil_pixels = 0

        if importance_sampling_gradient_norm:
            # record the training sampling distribution when doing importance sampling based on gradient norm
            # the 4 training data for which the sampling distribution will be recorded
            train_data_list = [[1.0, 120, 1, 1, 0, 'd', 1, 'lupine_small_xml', 128, 256, 256, False],
                               [0.7071067811865475, 60, 1, 1, 0, 'd', 4, 'lupine_small_xml', 128, 256, 256, False],
                               [1.0, 60, 1, 0, 0, 'd', 4, 'Lupine_22august', 120, 256, 256, False],
                               [1.0, 0, 1, 1, 1, 'd', 1, 'Lupine_22august', 120, 256, 256, False]]

            output_dir_sample_distribution = join(env_dir, 'train_data_sampling_distributions')
            os.makedirs(output_dir_sample_distribution, exist_ok=True)

            sampling_distribution_matrix_list = []
            # try to load the crop weight matrices from folder
            for i in range(len(train_data_list)):
                radius_i, rot_i, xflip_i, yflip_i, xy_swap_i, noise_type_i, noise_scale_i, data_name_i = \
                    train_data_list[i][:8]
                file_path = '{0}/{1}_rad{2:.2f}_rot{3}_xflip{4}_yflip{5}' \
                            '_x_y_swap{6}_noise_{7}{8}.npz'.format(
                    output_dir_sample_distribution, data_name_i, radius_i,
                    rot_i, xflip_i, yflip_i, xy_swap_i,
                    noise_type_i, noise_scale_i)
                if exists(file_path):
                    npz_file = np.load(file_path)
                    for k in npz_file:
                        key_val = k
                        break
                    sampling_distribution_matrix_list.append(npz_file[key_val])
                else:  # initial sampling distribution tensors are empty, shape (z,x,y)
                    train_data = train_data_list[i]
                    sampling_distribution_matrix_list.append(np.zeros((train_data[8], train_data[9],
                                                                       train_data[10])).astype(np.uint16))

        loop_start_time = time()

        for i_batch, sample_batched in enumerate(data_loader):
            print_remaining_time(loop_start_time, i_batch, data_loader)

            batch_ground_truth = sample_batched['ground_truth']
            batch_data = sample_batched['input']  # should not remove_first_axis
            batch_size = batch_data.size()[0]
            importance_weights = sample_batched['weight']
            assert len(importance_weights) == batch_size

            dont_care_mask = sample_batched['dont_care_mask']

            # # record the root percentages of the samples during training
            # root_perc = sample_batched['root_perc']
            # root_perc_list.extend(root_perc)

            if importance_sampling_gradient_norm:
                radius_list = sample_batched['radius']
                rot_list = sample_batched['rotation']
                x_flip_list = sample_batched['x_flip']
                y_flip_list = sample_batched['y_flip']
                x_y_swap_list = sample_batched['x_y_swap']
                noise_type_list = sample_batched['noise_type']
                noise_scale_list = sample_batched['data_index']
                data_name_list = sample_batched['data_name']

                input_position_x_list = sample_batched['input_start_x']
                input_position_y_list = sample_batched['input_start_y']
                input_position_z_list = sample_batched['input_start_z']

                # save the sampling distribution matrix if updated
                for i in range(batch_size):
                    data_features = [radius_list[i], rot_list[i], x_flip_list[i], y_flip_list[i], x_y_swap_list[i],
                                     noise_type_list[i], noise_scale_list[i], data_name_list[i]]
                    for j in range(len(train_data_list)):
                        current_train_data = train_data_list[j]
                        if current_train_data[
                           :8] == data_features:  # the current crop is from one of the training data to record

                            start_x = input_position_x_list[i]
                            start_y = input_position_y_list[i]
                            start_z = input_position_z_list[i]
                            assert start_x != -1

                            in_d_z, in_d_x, in_d_y = batch_data[i].size()
                            sampling_distribution_matrix_list[j][start_z:start_z + in_d_z,
                            start_x:start_x + in_d_x,
                            start_y:start_y + in_d_y] += np.ones((in_d_z,
                                                                  in_d_x,
                                                                  in_d_y)).astype(np.uint16)

                            # store the sampling_distribution_matrices
                            path_to_save = '{0}/{1}_rad{2:.2f}_rot{3}_xflip{4}_yflip{5}' \
                                           '_x_y_swap{6}_noise_{7}{8}.npz'.format(
                                output_dir_sample_distribution, data_name_list[i], radius_list[i],
                                rot_list[i], x_flip_list[i], y_flip_list[i], x_y_swap_list[i],
                                noise_type_list[i], noise_scale_list[i])

                            with open(path_to_save, 'wb+') as file_to_write:
                                np.savez_compressed(file_to_write, sampling_distribution_matrix_list[j])

            if use_loc_input_channels:  # first add distance_to_center, if exists, then depth_array
                batch_input = t.unsqueeze(batch_data, 1)  # shape of (batch_size, 1, z, x, y)
                if args.use_dist_to_center:
                    batch_distance_to_center = sample_batched['distance_to_center']  # shape of (batch_size, z, x, y)
                    batch_distance_to_center = t.unsqueeze(batch_distance_to_center,
                                                           1)  # shape of (batch_size, 1, z, x, y)
                    # concatenate as a new channel
                    batch_input = t.cat((batch_input, batch_distance_to_center),
                                        dim=1)  # shape of (batch_size, 2, z, x, y)
                if args.use_depth:
                    batch_depth_array = sample_batched['depth_array']
                    batch_depth_array = t.unsqueeze(batch_depth_array, 1)
                    # concatenate as a new channel
                    batch_input = t.cat((batch_input, batch_depth_array), dim=1)
            elif args.use_later_time:
                batch_input = t.unsqueeze(batch_data, 1)  # shape of (batch_size, 1, z, x, y)
                batch_data_later_time = sample_batched['input_later_time']
                batch_data_later_time = t.unsqueeze(batch_data_later_time, 1)  # shape of (batch_size, 1, z, x, y)
                batch_input = t.cat((batch_input, batch_data_later_time), dim=1)  # shape of (batch_size, 2, z, x, y)
            else:
                batch_input = batch_data

            batch_ground_truth, batch_input = batch_ground_truth.to(device), batch_input.to(
                device)

            # get the output from the model
            output, before_sigmoid = model(batch_input)

            # if the output has a smaller shape because of no padding,
            # need to process the ground_truth, dont_care_mask and root_weight_map to match the shapes of them
            batch_ground_truth = match_shape_big_to_small(batch_ground_truth, output)
            if dont_care:
                dont_care_mask = match_shape_big_to_small(dont_care_mask, output)

            # adding addtional parameters
            # todo: write a function to add additional parameters!
            additional_params = dict()

            if reweight_loss is True:
                if importance_weights[0].item() != -1:
                    # when importance_weights an array of -1 then do not use to calculate loss
                    importance_weights = importance_weights.to(device)
                    additional_params['importance_weights'] = importance_weights
            if dont_care:
                dont_care_mask = dont_care_mask.to(device)
                additional_params['dont_care_mask'] = dont_care_mask

            additional_params['calculate_gradient_diff_loss'] = calculate_gradient_diff_loss
            additional_params['device'] = device

            if loss_type == 'iou':
                additional_params['calculate_iou'] = True

            additional_params['mode'] = 'training'

            average_regularized_loss_list, total_regularized_loss_list, total_regularized_root_loss_list, \
            total_regularized_soil_loss_list, average_loss_list, total_loss_list, total_root_loss_list, \
            total_soil_loss_list, root_pixels_total_weight_list, soil_pixels_total_weight_list, \
            iou_loss, true_positive_count_list, \
            false_positive_count_list, false_negative_count_list, true_negative_count_list\
                = loss_calculator(
                output, batch_ground_truth, before_sigmoid,
                root_weight,
                **additional_params)

            # add to the loss of this epoch
            sum_total_loss += total_regularized_loss_list.detach().sum()
            sum_soil_loss += total_regularized_soil_loss_list.detach().sum()
            sum_root_loss += total_regularized_root_loss_list.detach().sum()
            total_weight_root_pixels += root_pixels_total_weight_list.detach().sum().item()
            total_weight_soil_pixels += soil_pixels_total_weight_list.detach().sum().item()

            # optimize after every batch
            if loss_type == 'nll':
                if importance_sampling_gradient_norm is True:

                    # paths of the weight tensor of each image where each input crop comes from
                    base_weight_path_list = sample_batched['base_weight_path']

                    # calculate the gradient norm for each data crop in the training batch,
                    # and update the sampling probability matrices
                    for i in range(len(average_regularized_loss_list)):
                        assert len(average_regularized_loss_list) == batch_size

                        # the loss of one data crop
                        loss_i = average_regularized_loss_list[i] / batch_size
                        loss_i.backward(retain_graph=True)

                        # from being freed, so the backward() can be done more than once
                        del loss_i

                        # get the gradient norm for this data crop
                        gradient_norm_i = get_gradient_norm(model)[0]  # todo: no longer correct function for this...

                        print('gradient_norm_i/100:', gradient_norm_i / 100)

                        # clear the gradients
                        optimizer.zero_grad()  # no longer needed

                        # load the weight tensor of the current image
                        base_weight_path = base_weight_path_list[i]
                        weight_matrix = np.load(base_weight_path, mmap_mode='r')  # shape: (z,y,x)
                        weight_tensor = t.tensor(weight_matrix)  # a FloatTensor, shape: (dim_z, dim_y, dim_x)

                        # assign a new probability of sampling of this specific crop
                        start_x = input_position_x_list[i].item()  # notice: was originally a tensor with single value!
                        start_y = input_position_y_list[i].item()
                        start_z = input_position_z_list[i].item()
                        assert start_x != -1

                        in_d_z, in_d_x, in_d_y = batch_data[i].size()

                        # value 2 is hard coded here:
                        # the crops whose weights will be affected by the current crop
                        # have at least 1/4 overlap with the current crop
                        cw_start_z = max(0,
                                         start_z - in_d_z // 2)  # if the starting index is negative, then only the last part of the index range is used
                        cw_end_z = start_z + in_d_z // 2  # the index of tensor if too big will automatically be converted to biggest possible index
                        cw_start_y = max(0, start_y - in_d_y // 2)
                        cw_end_y = start_y + in_d_y // 2
                        cw_start_x = max(0, start_x - in_d_x // 2)
                        cw_end_x = start_x + in_d_x // 2
                        crop_weight = weight_tensor[cw_start_z:cw_end_z, cw_start_y:cw_end_y, cw_start_x:cw_end_x]

                        # if part of the weight tensor crop has the original value (1),
                        # then assign the gradient norm directly to this part????

                        # todo: check if anything is wrong here
                        # divide by 100 because original gradient_norm_i is usually much larger than 1, in the range (10,200)
                        weight_tensor[cw_start_z:cw_end_z, cw_start_y:cw_end_y, cw_start_x:cw_end_x][
                            crop_weight == 1] = gradient_norm_i / 100
                        # if part of the weight tensor crop is already updated (!=1),
                        # then assign with the average between the current value and the gradient norm
                        weight_tensor[cw_start_z:cw_end_z, cw_start_y:cw_end_y, cw_start_x:cw_end_x][
                            crop_weight != 1] += gradient_norm_i / 100
                        weight_tensor[cw_start_z:cw_end_z, cw_start_y:cw_end_y, cw_start_x:cw_end_x][
                            crop_weight != 1] /= 2

                        # save the updated weight tensor
                        base_weight_mat = weight_tensor.numpy()
                        np.save(base_weight_path, base_weight_mat)

                        del weight_tensor, gradient_norm_i, crop_weight

                (average_regularized_loss_list.sum()/average_regularized_loss_list.size()[0]/gradient_application_interval).backward(retain_graph=False)

                global idx_train_batch
                # record the loss if using gdl loss (because it is different of the total loss now,
                # which is only the NLL loss)
                if calculate_gradient_diff_loss:
                    writer.add_scalar('Regularized_losses/Average training loss',
                                      average_regularized_loss_list.mean(), idx_train_batch)

                # record the gradient sum of model in every training step
                current_gradient_total_before, perc_grad_0_before = get_gradient_norm(model)

                writer.add_scalar('Gradient/Gradient L2 norm of each batch before clipping',
                                  current_gradient_total_before, idx_train_batch)

            elif loss_type == 'iou':
                # todo: get the list of iou_loss for all the data points in this batch
                iou_loss.backward()
                # the backward function accumulates gradients in the leaves (the trainable weights/biases) with respect to the loss

            elif loss_type == 'mse':
                # for the last batch which may have less samples,
                # still weight the samples with the same weight as previous batches: 1/mini_batch_size (constant)
                # then divide by gradient_application_interval for applying the optimization after certain number of batches
                (average_regularized_loss_list.sum() / average_regularized_loss_list.size()[0] / gradient_application_interval).backward(
                    retain_graph=False)

                # record the gradient sum of model in every training step
                current_gradient_total_before, perc_grad_0_before = get_gradient_norm(model)
                # global idx_train_batch
                writer.add_scalar('Gradient/Gradient L2 norm of each batch before clipping',
                                  current_gradient_total_before, idx_train_batch)

            # gradient_norms_v.add_point(step + ((i_batch+1)/len(data_loader)), norm, name="Norm")
            if debug:
                # todo: calculate the loss for each data crop in the mini-batch
                gradient_norm = get_gradient_norm(model)[0]

            if (i_batch+1)%gradient_application_interval==0 or i_batch+1==len(data_loader):
                clip_gradient()
                current_gradient_total_after, perc_grad_0_after = get_gradient_norm(model)
                writer.add_scalar('Gradient/Gradient L2 norm of after '
                                  'clipping after every {} batches'.format(gradient_application_interval),
                                  current_gradient_total_after, idx_train_batch)
                optimizer.step()
                optimizer.zero_grad()

            # every N-th batch, the training results will be saved
            if save_train_vis_interval != 0:
                if idx_train_batch % save_train_vis_interval == 0:
                    print('\nSaving training results......')
                    write_crop_prediction_results(output, sample_batched, batch_data, train_outputs_dir,
                                                  idx_train_batch,
                                                  total_regularized_soil_loss_list, total_regularized_root_loss_list,
                                                  importance_weights, reweight_loss, mode='training',
                                                  use_later_time=args.use_later_time)

            # to avoid memory increasing? delete the tensors that are no longer used
            del additional_params  # explicitly delete the dictionary?
            del output, before_sigmoid, batch_ground_truth
            del average_regularized_loss_list, total_regularized_loss_list, total_regularized_root_loss_list, \
                total_regularized_soil_loss_list, average_loss_list, total_loss_list, total_root_loss_list, \
                total_soil_loss_list, root_pixels_total_weight_list, soil_pixels_total_weight_list, \
                iou_loss, true_positive_count_list, \
                false_positive_count_list, false_negative_count_list, true_negative_count_list

            # every N-th batch, run visualization
            if idx_train_batch % vis_interval == 0:
                visualize_outputs(idx_train_batch)

            # every N-th batch, run validation
            if idx_train_batch % val_interval == 0:
                validate_network(idx_train_batch)

            if idx_train_batch % train_loss_reporting_interval == 0:
                # after certain number of batches, record the losses of this epoch: total, soil and root
                train_loss = sum_total_loss / (total_weight_root_pixels + total_weight_soil_pixels)
                if total_weight_root_pixels != 0:
                    train_root_loss = sum_root_loss / total_weight_root_pixels
                else:
                    train_root_loss = sum_root_loss
                train_soil_loss = sum_soil_loss / total_weight_soil_pixels

                writer.add_scalars('Regularized_losses/Training',
                                   {'Total loss': train_loss.item(),
                                    'Root loss': train_root_loss.item(),
                                    'Soil loss': train_soil_loss.item()},
                                   idx_train_batch)  # add to be visualized by tensorboard

                # reset the recording variables
                sum_total_loss = 0
                sum_soil_loss = 0
                sum_root_loss = 0
                total_weight_root_pixels = 0
                total_weight_soil_pixels = 0

            # schedule root_weight (change after certain number of batches)
            if root_weight_decrease_interval != 0:
                if idx_train_batch % root_weight_decrease_interval == 0:
                    root_weight = max(MIN_ROOT_WEIGHT, root_weight-1)  # lowest root weight is 1

            writer.add_scalar('Root weight', root_weight, idx_train_batch)

            idx_train_batch += 1  # add one after each training batch

            free_cached_memo_and_record_gpu_usage(idx_train_batch, 'Training')

        # every N-th epoch, save the network
        if step % save_interval == 0:
            save_network(step)

    elif validation and val_with_random_crop:  # do validation on one random crop of each validation data
        model.eval()  # change to eval mode, this might affect BN!!

        with t.no_grad():
            num_tp = 0
            num_fp = 0
            num_fn = 0
            num_tn = 0
            sum_total_loss = 0
            sum_soil_loss = 0
            sum_root_loss = 0
            total_weight_root_pixels = 0
            total_weight_soil_pixels = 0

            loss_recorder = dict()  # record the loss to be visualized in tensorboard
            loss_recorder['radius_multiplier'] = dict()
            loss_recorder['snr'] = dict()
            loss_recorder['data_name'] = dict()
            loss_recorder['noise_type_level'] = dict()

            if calculate_gradient_diff_loss:
                avg_val_loss = -1

            loop_start_time = time()

            for i_batch, sample_batched in enumerate(val_dataloader):
                print_remaining_time(loop_start_time, i_batch, val_dataloader)

                batch_ground_truth = sample_batched['ground_truth']
                batch_data = sample_batched['input']  # should not remove_first_axis
                batch_size = batch_data.size()[0]

                if use_loc_input_channels:  # first add distance_to_center, if exists, then depth_array
                    batch_input = t.unsqueeze(batch_data, 1)  # shape of (batch_size, 1, z, x, y)
                    if args.use_dist_to_center:
                        batch_distance_to_center = sample_batched[
                            'distance_to_center']  # shape of (batch_size, z, x, y)
                        batch_distance_to_center = t.unsqueeze(batch_distance_to_center,
                                                               1)  # shape of (batch_size, 1, z, x, y)
                        # concatenate as a new channel
                        batch_input = t.cat((batch_input, batch_distance_to_center),
                                            dim=1)  # shape of (batch_size, 2, z, x, y)
                    if args.use_depth:
                        batch_depth_array = sample_batched['depth_array']
                        batch_depth_array = t.unsqueeze(batch_depth_array, 1)
                        # concatenate as a new channel
                        batch_input = t.cat((batch_input, batch_depth_array), dim=1)
                elif args.use_later_time:
                    batch_input = t.unsqueeze(batch_data, 1)  # shape of (batch_size, 1, z, x, y)
                    batch_data_later_time = sample_batched['input_later_time']
                    batch_data_later_time = t.unsqueeze(batch_data_later_time, 1)  # shape of (batch_size, 1, z, x, y)
                    batch_input = t.cat((batch_input, batch_data_later_time), dim=1)  # shape of (batch_size, 2, z, x, y)
                else:
                    batch_input = batch_data

                batch_ground_truth, batch_input = batch_ground_truth.to(device), batch_input.to(
                    device)

              # get the output from the model
                output, before_sigmoid = model(batch_input)

                if COPY_EVERYTHING_FROM_NILS:
                    output = output[:, MASK_SIDE:-MASK_SIDE, MASK_SIDE:-MASK_SIDE, MASK_SIDE:-MASK_SIDE]
                    before_sigmoid = before_sigmoid[:, MASK_SIDE:-MASK_SIDE, MASK_SIDE:-MASK_SIDE,
                                                    MASK_SIDE:-MASK_SIDE]

                dont_care_mask = sample_batched['dont_care_mask']

                # if the output has a smaller shape because of no padding,
                # need to process the ground_truth, dont_care_mask and root_weight_map to match the shapes of them
                batch_ground_truth = match_shape_big_to_small(batch_ground_truth, output)
                if dont_care:
                    dont_care_mask = match_shape_big_to_small(dont_care_mask, output)

                # adding addtional parameters
                # todo: write a function to add additional parameters!
                additional_params = dict()

                if dont_care:
                    dont_care_mask = dont_care_mask.to(device)
                    additional_params['dont_care_mask'] = dont_care_mask

                additional_params['calculate_gradient_diff_loss'] = calculate_gradient_diff_loss
                additional_params['device'] = device
                additional_params['mode'] = 'validation'

                # calculate loss here
                average_regularized_loss_list, total_regularized_loss_list, total_regularized_root_loss_list, \
                total_regularized_soil_loss_list, average_loss_list, total_loss_list, total_root_loss_list, \
                total_soil_loss_list, root_pixels_total_weight_list, soil_pixels_total_weight_list, \
                iou_loss, true_positive_count_list, \
                false_positive_count_list, false_negative_count_list, true_negative_count_list\
                    = loss_calculator(
                    output, batch_ground_truth, before_sigmoid,
                    root_weight,
                    **additional_params)

                # add to the loss of this epoch
                sum_total_loss += total_regularized_loss_list.detach().sum()
                sum_soil_loss += total_regularized_soil_loss_list.detach().sum()
                sum_root_loss += total_regularized_root_loss_list.detach().sum()
                total_weight_root_pixels += root_pixels_total_weight_list.detach().sum().item()
                total_weight_soil_pixels += soil_pixels_total_weight_list.detach().sum().item()
                num_fn += false_negative_count_list.detach().sum()
                num_fp += false_positive_count_list.detach().sum()
                num_tp += true_positive_count_list.detach().sum()
                num_tn += true_negative_count_list.detach().sum()

                # update the avg loss if using gdl loss
                if calculate_gradient_diff_loss:
                    if avg_val_loss == -1:
                        avg_val_loss = average_regularized_loss_list.mean()
                    else:  # update
                        avg_val_loss = (avg_val_loss + average_regularized_loss_list.mean()) / 2

                # recording the loss and f1 according to different data features
                list_real_data = sample_batched['real_data']
                list_radius = sample_batched['radius']
                list_snr = sample_batched['snr']
                list_data_name = sample_batched['data_name']
                list_noise_type = sample_batched['noise_type']
                list_data_index = sample_batched['data_index']

                # record the validation metrics
                for i in range(len(average_regularized_loss_list)):

                    if list_real_data[i]=='False':  # only applies to virtual data
                        data_feature_kwargs = dict()
                        data_feature_kwargs['radius_multiplier'] = list_radius[i].detach().item()
                        data_feature_kwargs['snr'] = list_snr[i].detach().item()
                        data_feature_kwargs['data_name'] = list_data_name[i]  # strings are not converted to tensors, cannot detach
                        data_feature_kwargs['noise_type'] = list_noise_type[i]
                        data_feature_kwargs['data_index'] = list_data_index[i].detach().item()

                        loss_results_kwargs = dict()
                        loss_results_kwargs['total_loss'] = total_regularized_loss_list[i].detach().item()
                        loss_results_kwargs['soil_loss'] = total_regularized_soil_loss_list[i].detach().item()
                        loss_results_kwargs['root_loss'] = total_regularized_root_loss_list[i].detach().item()
                        loss_results_kwargs['root_pixels_total_weight'] = root_pixels_total_weight_list[i].detach().item()
                        loss_results_kwargs['soil_pixels_total_weight'] = soil_pixels_total_weight_list[i].detach().item()
                        loss_results_kwargs['num_tp'] = true_positive_count_list[i].detach()
                        loss_results_kwargs['num_fp'] = false_positive_count_list[i].detach()
                        loss_results_kwargs['num_fn'] = false_negative_count_list[i].detach()

                        loss_recorder = record_losses(loss_recorder, data_feature_kwargs, loss_results_kwargs)

                del additional_params  # explicitly delete the dictionary?
                del output, before_sigmoid, batch_ground_truth
                del average_regularized_loss_list, total_regularized_loss_list, total_regularized_root_loss_list, \
                   total_regularized_soil_loss_list, average_loss_list, total_loss_list, total_root_loss_list, \
                   total_soil_loss_list, root_pixels_total_weight_list, soil_pixels_total_weight_list,  \
                   iou_loss, true_positive_count_list, \
                   false_positive_count_list, false_negative_count_list, true_negative_count_list

            # after running through all val data
            # calculate the loss of all validation data
            validation_loss = sum_total_loss / (total_weight_root_pixels + total_weight_soil_pixels)
            validation_root_loss = sum_root_loss / total_weight_root_pixels
            validation_soil_loss = sum_soil_loss / total_weight_soil_pixels

            # convert the data type of num_tp, num_fp and num_fn (LongTensors) to float
            num_tp = float(num_tp.item())
            num_fp = float(num_fp.item())
            num_fn = float(num_fn.item())
            num_tn = float(num_tn.item())

            # calculate the F1 score of all validation data
            prec = safe_divide(num_tp, num_tp + num_fp)
            recall = safe_divide(num_tp, num_tp + num_fn)
            f_score = safe_divide(2. * prec * recall, prec + recall)

            writer.add_scalars('Regularized_losses/Validation',
                               {'Total loss': validation_loss.item(),
                                'Root loss': validation_root_loss.item(),
                                'Soil loss': validation_soil_loss.item()},
                               step)  # add to be visualized by tensorboard
            writer.add_scalar('validation_f_score', f_score, step)

            if calculate_gradient_diff_loss:
                writer.add_scalar('Regularized_losses/Average validation loss',
                                  avg_val_loss, step)

            # in case in this round of validation there is no root voxels, thus tp=0, then F1 will always be 0
            # so also report the number of tp, fp, fn, tn
            writer.add_scalars('validation_f_score/number of tp, fp, fn, tn',
                               {'num_tp': num_tp,
                                'num_fp': num_fp,
                                'num_fn': num_fn,
                                'num_tn': num_tn},
                               step)

            # record the loss according to the features of this image (snr, radius, ...)
            add_val_losses_per_category(loss_recorder, writer, step)
            print()

        model.train()  # IMPORTANT! return to the training mode

    else:  # if not training
        model.eval()  # change to eval mode, this might affect BN!!

        with t.no_grad():
            if test:  # should calculate the evaluation metrics
                if training_datalist is not None:
                    datalist_copy = deepcopy(test_datalist)
                    # add one column which represents if this data row needs to be combined with soil
                    comb_soil_column1 = np.repeat(False, datalist_copy.shape[0]).reshape(
                        (datalist_copy.shape[0], 1))
                    datalist_copy = np.concatenate((datalist_copy, comb_soil_column1), axis=1)

                if training_datalist_combining is not None:
                    datalist_combining_copy = deepcopy(test_datalist_combining)
                    # add one column which represents if this data row needs to be combined with soil
                    comb_soil_column2 = np.repeat(True, datalist_combining_copy.shape[0]).\
                                        reshape((datalist_combining_copy.shape[0], 1))
                    datalist_combining_copy = np.concatenate((datalist_combining_copy, comb_soil_column2), axis=1)

                if (training_datalist is not None) and (training_datalist_combining is not None):
                    datalist = np.concatenate((datalist_copy, datalist_combining_copy), axis=0)
                elif training_datalist is not None:
                    datalist = datalist_copy
                else:
                    datalist = datalist_combining_copy

                # todo: calculate the distance tolerant F1 score
            elif validation:  # calculate the evaluation metrics and validation loss
                if training_datalist is not None:
                    datalist_copy = deepcopy(validation_datalist)
                    # add one column which represents if this data row needs to be combined with soil
                    comb_soil_column1 = np.repeat(False, datalist_copy.shape[0]).reshape(
                        (datalist_copy.shape[0], 1))
                    datalist_copy = np.concatenate((datalist_copy, comb_soil_column1), axis=1)

                if training_datalist_combining is not None:
                    datalist_combining_copy = deepcopy(validation_datalist_combining)
                    # add one column which represents if this data row needs to be combined with soil
                    comb_soil_column2 = np.repeat(True, datalist_combining_copy.shape[0]).\
                                        reshape((datalist_combining_copy.shape[0], 1))
                    datalist_combining_copy = np.concatenate((datalist_combining_copy, comb_soil_column2), axis=1)

                if (training_datalist is not None) and (training_datalist_combining is not None):
                    datalist = np.concatenate((datalist_copy, datalist_combining_copy), axis=0)
                elif training_datalist is not None:
                    datalist = datalist_copy
                else:
                    datalist = datalist_combining_copy

                soil_non_training_datalist = soil_data_val_list

                sum_total_loss = 0
                sum_soil_loss = 0
                sum_root_loss = 0
                total_weight_root_pixels = 0
                total_weight_soil_pixels = 0
                F_score_list = []  # the list of f score of all whole data in the val set

                loss_recorder = dict()  # record the loss to be visualized in tensorboard
                loss_recorder['radius_multiplier'] = dict()
                loss_recorder['snr'] = dict()
                loss_recorder['data_name'] = dict()
                loss_recorder['noise_type_level'] = dict()

            elif visualization:
                if training_datalist is not None:
                    datalist_copy = deepcopy(visualization_datalist)
                    # add one column which represents if this data row needs to be combined with soil
                    comb_soil_column1 = np.repeat(False, datalist_copy.shape[0]).reshape(
                        (datalist_copy.shape[0], 1))
                    datalist_copy = np.concatenate((datalist_copy, comb_soil_column1), axis=1)
                    datalist = datalist_copy

                elif training_datalist_combining is not None:
                    # because usually the combining vis dataset (commonly only contains real data)
                    # is a subset of the vis dataset, so when the non-combining dataset already exists,
                    # there's no need to add the combining dataset to it
                    datalist_combining_copy = deepcopy(visualization_datalist_combining)
                    # add one column which represents if this data row needs to be combined with soil
                    comb_soil_column2 = np.repeat(True, datalist_combining_copy.shape[0]).\
                                        reshape((datalist_combining_copy.shape[0], 1))
                    datalist_combining_copy = np.concatenate((datalist_combining_copy, comb_soil_column2), axis=1)
                    datalist = datalist_combining_copy

                    # if combine_soil_with_intensity:
                    soil_non_training_datalist = soil_data_vis_list

            data_serial_num = 1
            if debug:
                print("#" * 100)
                print("length of data list:", len(datalist))
                print("Debugging mode?", debug)

            for data in datalist:  # generate a data list for one single image
                print("\n(Non-training) Processing the {}-th data".format(data_serial_num))
                data_serial_num += 1

                if validation:  # calculate the F score for each whole data
                    num_tp = 0
                    num_fp = 0
                    num_fn = 0

                # get the features of this data point
                [radius, rotation, x_flip, y_flip, x_y_swap, noise_type,
                 data_index, data_name, slice_count, width, height, real_data, combine_with_soil] = data

                # convert the features to its own data type
                radius = float(radius)
                rotation = int(rotation)
                x_flip = int(x_flip)
                y_flip = int(y_flip)
                x_y_swap = int(x_y_swap)
                slice_count = int(slice_count)
                width = int(width)
                height = int(height)
                if real_data == 'True':
                    real_data = True
                elif real_data == 'False':
                    real_data = False
                else:
                    raise ValueError('Unknown value for feature real_data: \'{0}\'.'.format(real_data))

                # generate the splitted data list (split each 3D image according to the user)
                splitted_datalist, out_cropsize_xyz, \
                overlappings_xyz, special_overlappings_xyz = cut_non_training_data(data[:-1], shape_decreases,
                                                                                   non_training_crop_size,
                                                                                   super_res_factor=super_resolution_factor,
                                                                                   overlaps_after_prediction=overlaps_after_prediction)
                # using data[:-1] because the last feature is boolean combine_with_soil, which will not be used here

                padding_x, padding_y, padding_z = splitted_datalist[0, -3:]
                padding_x, padding_y, padding_z = int(float(padding_x)), int(float(padding_y)), int(float(padding_z))

                if validation:
                    use_dont_care = dont_care

                else:  # no need to use dont care mask if it's test or visualization mode
                    use_dont_care = False

                ground_truth_path = None
                occupancy_path = None
                img_path = None
                snr = None
                if combine_with_soil == 'True':
                    soil_data_path = None

                soil_scaling_factor = None

                if combine_with_soil == 'False' or real_data:
                    data_manager = DatasetManager(splitted_datalist, data_dir,
                                                  super_resolution_factor=super_resolution_factor,
                                                  add_plane_artifacts=False, normalize=normalize_input_tensor,
                                                  is_training_set=False,
                                                  is_regression=(loss_type == 'mse'),
                                                  dont_care=use_dont_care,
                                                  random_scaling=False,
                                                  length=len(splitted_datalist),
                                                  diff_times=args.use_later_time,
                                                  use_depth=args.use_depth,
                                                  use_dist_to_center=args.use_dist_to_center)
                else:
                    # if is not real data, although soil_scaling_factor is taken, it is not used in data loading
                    data_manager = DatasetManager_combining(splitted_datalist, data_dir,
                                                            super_resolution_factor=super_resolution_factor,
                                                            add_plane_artifacts=False,
                                                            normalize=normalize_input_tensor,
                                                            is_training_set=False,
                                                            soil_data_list=soil_non_training_datalist,
                                                            soil_scaling_factor=soil_scaling_factor,
                                                            random_scaling=False,
                                                            length=len(splitted_datalist),
                                                            diff_times=args.use_later_time,
                                                            for_val=True,
                                                            use_depth=args.use_depth,
                                                            use_dist_to_center=args.use_dist_to_center)

                if test:
                    worker_ind = 16
                elif validation:
                    worker_ind = 32
                elif visualization:
                    worker_ind = 48

                data_loader = DataLoader(data_manager, batch_size=non_training_batch_size, shuffle=False, num_workers=DATALOADER_NUM_WORKERS,
                                         worker_init_fn=lambda x: np.random.seed(x + np_seed + worker_ind))

                loop_start_time = time()

                for i_batch, sample_batched in enumerate(data_loader):

                    print_remaining_time(loop_start_time, i_batch, data_loader)

                    batch_data = sample_batched['input']  # should not remove_first_axis

                    batch_start_position_xyz = sample_batched['out_start_xyz']  # the output start positions in the padded ground truth: xyz_in_start * self.super_res_factor

                    if use_loc_input_channels:  # first add distance_to_center, if exists, then depth_array
                        batch_input = t.unsqueeze(batch_data, 1)  # shape of (batch_size, 1, z, x, y)
                        if args.use_dist_to_center:
                            batch_distance_to_center = sample_batched[
                                'distance_to_center']  # shape of (batch_size, z, x, y)
                            batch_distance_to_center = t.unsqueeze(batch_distance_to_center,
                                                                   1)  # shape of (batch_size, 1, z, x, y)
                            # concatenate as a new channel
                            batch_input = t.cat((batch_input, batch_distance_to_center),
                                                dim=1)  # shape of (batch_size, 2, z, x, y)
                        if args.use_depth:
                            batch_depth_array = sample_batched['depth_array']
                            batch_depth_array = t.unsqueeze(batch_depth_array, 1)
                            # concatenate as a new channel
                            batch_input = t.cat((batch_input, batch_depth_array), dim=1)
                    elif args.use_later_time:
                        batch_input = t.unsqueeze(batch_data, 1)  # shape of (batch_size, 1, z, x, y)
                        batch_data_later_time = sample_batched['input_later_time']
                        batch_data_later_time = t.unsqueeze(batch_data_later_time, 1)  # shape of (batch_size, 1, z, x, y)
                        batch_input = t.cat((batch_input, batch_data_later_time),
                                            dim=1)  # shape of (batch_size, 2, z, x, y)
                    else:
                        batch_input = batch_data

                    batch_input = batch_input.to(device)

                    if ground_truth_path is None:
                        gt_dim_z, gt_dim_x, gt_dim_y = 0, 0, 0
                        if data_name == 'pure_soil':
                            gt_dim_z = slice_count * super_resolution_factor
                            gt_dim_x = width * super_resolution_factor
                            gt_dim_y = height * super_resolution_factor

                            ground_truth = np.zeros((gt_dim_z, gt_dim_x, gt_dim_y)).astype(np.float32)
                        else:
                            ground_truth_path = sample_batched['ground_truth_path'][0]
                            ground_truth = load_array_from_file(ground_truth_path, ground_truth_path)
                            gt_dim_z, _, gt_dim_x, gt_dim_y = ground_truth.shape

                            # load the ground truth
                            ground_truth = np.squeeze(ground_truth, axis=1)  # to shape (z, x, y)
                            ground_truth = ground_truth.astype(np.float32) / 255.  # to float

                        # pad the ground truth (because the noisy image is padded)
                        ground_truth = np.pad(ground_truth,
                                              ((padding_z // 2 * super_resolution_factor, (padding_z - padding_z // 2) * super_resolution_factor),
                                               (padding_x // 2 * super_resolution_factor, (padding_x - padding_x // 2) * super_resolution_factor),
                                               (padding_y // 2 * super_resolution_factor, (padding_y - padding_y // 2) * super_resolution_factor)),
                                              'constant', constant_values=(0,))
                        ground_truth = t.from_numpy(ground_truth)  # to tensor

                        if debug:
                            whole_debugging_tensor = t.zeros((gt_dim_z, gt_dim_x, gt_dim_y))  ### debugging

                        whole_prediction = t.zeros((gt_dim_z, gt_dim_x, gt_dim_y)).to(device)

                    if occupancy_path is None:
                        occupancy_path = sample_batched['occupancy_path'][0]
                        if use_dont_care:
                            if data_name == 'pure_soil':
                                dont_care_mask = deepcopy(ground_truth)  # both are all-zero
                            else:
                                occupancy = load_array_from_file(occupancy_path,
                                                                 occupancy_path)  # a function in DataLoader.py
                                occupancy = np.moveaxis(occupancy, 2, 0)  # to shape (z, x, y)
                                occupancy = occupancy.astype(np.float32) / 255.  # to float
                                occupancy = t.from_numpy(occupancy)  # to tensor

                                # pad the occupancy the same way as gt
                                occupancy = np.pad(occupancy,
                                                      ((padding_z // 2 * super_resolution_factor,
                                                        (padding_z - padding_z // 2) * super_resolution_factor),
                                                       (padding_x // 2 * super_resolution_factor,
                                                        (padding_x - padding_x // 2) * super_resolution_factor),
                                                       (padding_y // 2 * super_resolution_factor,
                                                        (padding_y - padding_y // 2) * super_resolution_factor)),
                                                      'constant', constant_values=(0,))

                                dont_care_mask = generate_dont_care_mask(ground_truth, occupancy)

                    if snr is None:  # get the signal-to-noise ratio for this image
                        snr = sample_batched['snr'][0].item()

                    if img_path is None:
                        img_path = sample_batched['img_path'][0]

                    if combine_with_soil == 'True':
                        if soil_data_path is None:
                            soil_data_path = sample_batched['soil_data_path'][0]

                    # adding addtional parameters
                    additional_params = dict()

                    batch_output, batch_before_sigmoid = model(batch_input, **additional_params)  # (batchsize, z, x, y)

                    if (data_name in DATA_Z_FLIP) and use_loc_input_channels:  # flip the z axis back to original
                        batch_output = t.flip(batch_output, dims=[1])
                        batch_before_sigmoid = t.flip(batch_before_sigmoid, dims=[1])

                    if COPY_EVERYTHING_FROM_NILS:  # cut the outside of the crop
                        batch_output = batch_output[:, MASK_SIDE:-MASK_SIDE, MASK_SIDE:-MASK_SIDE, MASK_SIDE:-MASK_SIDE]
                        batch_before_sigmoid = batch_before_sigmoid[:, MASK_SIDE:-MASK_SIDE, MASK_SIDE:-MASK_SIDE, MASK_SIDE:-MASK_SIDE]

                    if use_dont_care:
                        batch_dont_care_mask_processed = []

                    # processing the output crops and add to the corresponding parts in the whole output
                    for i in range(batch_start_position_xyz.size()[0]):  # get one output from the mini-batch

                        output = batch_output[i]
                        before_sigmoid = batch_before_sigmoid[i]

                        out_start_x, out_start_y, out_start_z = batch_start_position_xyz[i]

                        # matching the size of gt to output
                        out_dim_z, out_dim_x, out_dim_y = output.size()
                        out_start_x_gt = out_start_x + (non_training_crop_size[0] * super_resolution_factor - out_dim_x) // 2  # imitating match_shape_big_to_small()
                        out_start_y_gt = out_start_y + (non_training_crop_size[1] * super_resolution_factor - out_dim_y) // 2
                        out_start_z_gt = out_start_z + (non_training_crop_size[2] * super_resolution_factor - out_dim_z) // 2

                        output, crop_x_start, crop_x_end, crop_y_start, crop_y_end, \
                        crop_z_start, crop_z_end = cut_overlappings(output,
                                                                    (out_start_x, out_start_y, out_start_z),
                                                                    whole_prediction.size(),
                                                                    overlappings_xyz, special_overlappings_xyz)

                        # update the start positions of the output crop
                        out_start_x += crop_x_start  # out_start_x corresponding to non-padded ground truth
                        out_start_y += crop_y_start
                        out_start_z += crop_z_start
                        out_start_x_gt += crop_x_start
                        out_start_y_gt += crop_y_start
                        out_start_z_gt += crop_z_start

                        # cut before_sigmoid in the same way as output crop
                        before_sigmoid = before_sigmoid[crop_z_start:crop_z_end, crop_x_start:crop_x_end,
                                                        crop_y_start:crop_y_end]

                        # assemble the output crops into a whole image
                        whole_prediction = add_one_crop_to_whole_prediction(output, whole_prediction,
                                                         (out_start_x, out_start_y, out_start_z))

                        if debug:
                            ### for debugging if the adding is correct
                            place_holder_crop = t.ones(output.size())
                            # assign the borders of the crop with higher value
                            place_holder_crop[0, :, :] = 10
                            place_holder_crop[-1, :, :] = 10
                            place_holder_crop[:, 0, :] = 10
                            place_holder_crop[:, -1, :] = 10
                            place_holder_crop[:, :, 0] = 10
                            place_holder_crop[:, :, -1] = 10
                            add_one_crop_to_whole_prediction(place_holder_crop, whole_debugging_tensor,
                                                             (out_start_x, out_start_y, out_start_z))

                        if validation:  # directly calculate the loss of each crop (the whole image takes too much memory)
                            # get the corresponding ground_truth crop
                            out_shape_z, out_shape_x, out_shape_y = output.size()
                            # the ground truth should be the same shape as the whole prediction!!
                            # Otherwise the indexing is no longer valid
                            crop_gt = ground_truth[out_start_z_gt:out_start_z_gt + out_shape_z,
                                                   out_start_x_gt:out_start_x_gt + out_shape_x,
                                                   out_start_y_gt:out_start_y_gt + out_shape_y]
                            crop_gt = crop_gt.to(device)

                            # adding addtional parameters
                            # todo: write a function to add additional parameters! There are still some argument not added here!
                            additional_params = dict()
                            additional_params['device'] = device

                            additional_params['mode'] = 'validation'

                            if use_dont_care:
                                crop_dont_care_mask = dont_care_mask[out_start_z_gt:out_start_z_gt + out_shape_z,
                                                                     out_start_x_gt:out_start_x_gt + out_shape_x,
                                                                     out_start_y_gt:out_start_y_gt + out_shape_y]
                                additional_params['dont_care_mask'] = crop_dont_care_mask.to(device)

                            # calculate the loss for each output crop
                            average_regularized_loss, total_regularized_loss, total_regularized_root_loss, total_regularized_soil_loss, \
                            average_loss, total_loss, total_root_loss, total_soil_loss, root_pixels_total_weight, \
                            soil_pixels_total_weight, iou_loss, \
                            true_positive_count, false_positive_count, false_negative_count, true_negative_count\
                                = loss_calculator(
                                output, crop_gt, before_sigmoid, root_weight,
                                **additional_params)

                            # no need to .sum() because the input to loss_calculator is not in batch but only individual
                            total_weight_root_pixels += root_pixels_total_weight.detach().item()
                            total_weight_soil_pixels += soil_pixels_total_weight.detach().item()
                            sum_total_loss += total_regularized_loss.detach().item()
                            sum_soil_loss += total_regularized_soil_loss.detach().item()
                            sum_root_loss += total_regularized_root_loss.detach().item()

                            num_tp += true_positive_count.detach()
                            num_fp += false_positive_count.detach()
                            num_fn += false_negative_count.detach()

                            # in order to avoid memory increase?
                            del additional_params
                            del output, before_sigmoid, crop_gt

                            del average_regularized_loss, total_regularized_loss, total_regularized_root_loss, total_regularized_soil_loss, \
                                average_loss, total_loss, total_root_loss, total_soil_loss, root_pixels_total_weight, \
                                soil_pixels_total_weight, iou_loss, \
                                true_positive_count, false_positive_count, false_negative_count, true_negative_count

                # after the whole image prediction is assembled
                if validation:  # todo: calculate the loss of the whole image
                    # here record the losses and weights according to the category of the current data (augmented)
                    if not real_data:
                        data_feature_kwargs = dict()
                        data_feature_kwargs['radius_multiplier'] = radius
                        data_feature_kwargs['snr'] = snr
                        data_feature_kwargs['data_name'] = data_name
                        data_feature_kwargs['noise_type'] = noise_type
                        data_feature_kwargs['data_index'] = data_index

                        loss_results_kwargs = dict()
                        loss_results_kwargs['total_loss'] = sum_total_loss
                        loss_results_kwargs['soil_loss'] = sum_soil_loss
                        loss_results_kwargs['root_loss'] = sum_root_loss
                        loss_results_kwargs['root_pixels_total_weight'] = total_weight_root_pixels
                        loss_results_kwargs['soil_pixels_total_weight'] = total_weight_soil_pixels
                        loss_results_kwargs['num_tp'] = num_tp
                        loss_results_kwargs['num_fp'] = num_fp
                        loss_results_kwargs['num_fn'] = num_fn

                        loss_recorder = record_losses(loss_recorder, data_feature_kwargs, loss_results_kwargs)

                    # calculate the F score of the current whole image
                    # convert the data type of num_tp, num_fp and num_fn (LongTensors) to float
                    num_tp = float(num_tp.item())
                    num_fp = float(num_fp.item())
                    num_fn = float(num_fn.item())

                    # calculate the F1 score of all validation data
                    prec = safe_divide(num_tp, num_tp + num_fp)
                    recall = safe_divide(num_tp, num_tp + num_fn)
                    f_score = safe_divide(2. * prec * recall, prec + recall)
                    F_score_list.append(f_score)

                elif visualization:  # store the output as file

                    if real_data:  # notice that real_data was a string when just loaded, need to convert to boolean

                        dir_name = '{0}/Step_{1}/{2}/mri'.format(outputs_dir, step, data_name)

                        if data_name != "pure_soil":
                            real_data_manual_recon_path = \
                                '{0}/{2}/r_factor_1.00/rot_0/x_flip_0/y_flip_0/' \
                                'x_y_swap_0/{3}x{4}x{5}/ground_truth{6}.npy'.format(
                                    data_dir, step, data_name, width, height, slice_count,
                                    "" if super_resolution_factor == 1
                                    else "_res_{}x".format(super_resolution_factor))

                    else:
                        dir_name = '{0}/Step_{1}/{2}/rad_{3:.2f}/rot_{4}/x_flip_{5}/y_flip_{6}/x_y_swap{7}/'.format(
                            outputs_dir, step, data_name, float(radius), rotation, x_flip, y_flip, x_y_swap)

                    os.makedirs(dir_name, exist_ok=True)

                    # check if img_path is real noisy image path or intensity,
                    # if intensity, need to generate the noisy image here
                    # todo: instead of generating the noisy image again, try assemble the input crops?
                    if img_path.endswith('intensity.npz'):
                        # todo: get soil_scaling_factor and soil_data_path
                        file_name = "{0}/out_soilScalingFac_{}".format(dir_name, soil_scaling_factor)

                        input_img_filename = "{0}/in_soilScalingFac_{}".format(dir_name, soil_scaling_factor)
                        intensity = np.load(img_path)['arr_0']
                        intensity = np.moveaxis(intensity, 2, 0)
                        intensity = np.expand_dims(intensity, axis=1)  # convert to the required shape for combine_whole_intensity_with_soil
                        occupancy = np.load(occupancy_path)['arr_0']
                        occupancy = np.moveaxis(occupancy, 2, 0)
                        occupancy = np.expand_dims(occupancy, axis=1)

                        soil_data = np.load(soil_data_path)
                        out_of_pot_mask = generate_out_of_pot_mask(rotation, x_flip, y_flip, x_y_swap,
                                                                   data_type_name=TO_DATA_TYPE_NAME_DICT[data_name])
                        combined_noisy_image = combine_whole_intensity_with_soil(intensity, occupancy, soil_data,
                                                                                 soil_scaling_factor,
                                                                                 out_of_pot_mask=out_of_pot_mask)
                        np.savez(input_img_filename, combined_noisy_image)
                    else:
                        file_name = "{0}/out_{1}_{2}.npz".format(dir_name, noise_type, data_index)
                        img_sym_path = "{0}/in_{1}_{2}.npy".format(dir_name, noise_type, data_index)
                        symlink_force(img_path, img_sym_path)

                    with open(file_name, 'wb+') as file_to_write:
                        # convert data type to Byte
                        whole_prediction = (whole_prediction.cpu().numpy() * 255).astype(np.uint8)
                        np.savez_compressed(file_to_write, whole_prediction)

                    if debug:
                        # for debugging, save whole_debugging_tensor
                        debug_file_name = "{0}/debug_out.npy".format(dir_name)
                        with open(debug_file_name, 'wb+') as file_to_write:
                            whole_debugging_tensor = whole_debugging_tensor.numpy()
                            # convert to uint8
                            whole_debugging_tensor = whole_debugging_tensor.astype(np.uint8)
                            np.save(file_to_write, whole_debugging_tensor)

                elif test:  # calculate the metrics
                    pass

            # after running the model on all the images
            if validation:  # if this is the last validation input data: average the val loss / F1 over all validation data
                # calculate the loss of all validation data
                validation_loss = sum_total_loss / (total_weight_root_pixels + total_weight_soil_pixels)
                validation_root_loss = sum_root_loss / total_weight_root_pixels
                validation_soil_loss = sum_soil_loss / total_weight_soil_pixels

                mean_f_score = sum(F_score_list)/len(F_score_list)  # average F score of all whole data

                writer.add_scalars('Regularized_losses/Validation',
                                   {'Total loss': validation_loss,
                                    'Root loss': validation_root_loss,
                                    'Soil loss': validation_soil_loss},
                                   step)  # add to be visualized by tensorboard
                writer.add_scalar('validation_f_score', mean_f_score, step)

                # record the loss according to the features of this image (snr, radius, ...)
                add_val_losses_per_category(loss_recorder, writer, step)

        model.train()  # IMPORTANT! return to the training mode


def safe_divide(x, y): return x / y if y != 0 else 0


def match_shape_big_to_small(t_big, t_small):
    '''
    When no padding in the conv net, match the shapes of tensors by removing the borders on each dimension
    Only correct when the image shape is (batch_size, 3D_image_shape)
    :param t_big:
    :param t_small:
    :return:
    '''
    assert type(t_big)==type(t_small)  # should be the same type, either Tensors or arrays

    try:
        if isinstance(t_big, t.Tensor):
            big_shape = t_big.size()
            small_shape = t_small.size()
        elif isinstance(t_big, np.ndarray):
            big_shape = t_big.shape
            small_shape = t_small.shape
        else:
            raise InvalidTypeError
    except InvalidTypeError:
        print("Invalid type, the type should either be torch.Tensor or numpy.array")

    assert len(big_shape) == 4 and len(small_shape) == 4

    diff_d = big_shape[1] - small_shape[1]
    diff_w = big_shape[2] - small_shape[2]
    diff_h = big_shape[3] - small_shape[3]
    t_big_cropped = t_big[:, diff_d // 2:diff_d // 2 + small_shape[1],
            diff_w // 2:diff_w // 2 + small_shape[2], diff_h // 2:diff_h // 2 + small_shape[3]]
    return t_big_cropped


def record_losses(loss_recorder, data_feature_kwargs, loss_results_kwargs):
    radius_multiplier = data_feature_kwargs['radius_multiplier']
    snr = data_feature_kwargs['snr']

    if radius_multiplier not in loss_recorder['radius_multiplier']:
        # the first time this radius_multiplier is encountered
        loss_recorder['radius_multiplier'][radius_multiplier] = dict()
        for key in loss_results_kwargs:
            loss_recorder['radius_multiplier'][radius_multiplier][key] = loss_results_kwargs[key]
    else:
        for key in loss_results_kwargs:
            loss_recorder['radius_multiplier'][radius_multiplier][key] += loss_results_kwargs[key]

    if snr != 0:  # the levels can be one of [1,2,3,4,5], level5 is range(100, inf), which will be ignored
        snr_level = len(SNR_THRESHOLDS) - ((SNR_THRESHOLDS / snr) > 1).sum()
    else:
        snr_level = 1
    if snr_level not in loss_recorder['snr']:
        # the first time this snr level is encountered
        loss_recorder['snr'][snr_level] = dict()
        for key in loss_results_kwargs:
            loss_recorder['snr'][snr_level][key] = loss_results_kwargs[key]
    else:
        for key in loss_results_kwargs:
            loss_recorder['snr'][snr_level][key] += loss_results_kwargs[key]

    return loss_recorder


def add_val_losses_per_category(loss_recorder, _writer, step):
    for feature in loss_recorder:  # features including radius, snr, and so on
        loss_dict = dict()
        root_loss_dict = dict()
        soil_loss_dict = dict()
        f1_dict = dict()

        for value in loss_recorder[feature]:
            _total_weight_root_pixels = loss_recorder[feature][value]['root_pixels_total_weight']
            _total_weight_soil_pixels = loss_recorder[feature][value]['soil_pixels_total_weight']
            _sum_total_loss = loss_recorder[feature][value]['total_loss']
            _sum_soil_loss = loss_recorder[feature][value]['soil_loss']
            _sum_root_loss = loss_recorder[feature][value]['root_loss']

            _num_tp = loss_recorder[feature][value]['num_tp']
            _num_fp = loss_recorder[feature][value]['num_fp']
            _num_fn = loss_recorder[feature][value]['num_fn']

            if feature == "snr":
                if value < len(SNR_THRESHOLDS):
                    value_name = '{:.2f}:{:.2f}'.format(SNR_THRESHOLDS[value - 1], SNR_THRESHOLDS[value])
                else:
                    value_name = '{:.2f}:infinity'.format(SNR_THRESHOLDS[value - 1])
            elif type(value) == float:
                value_name = '{:.2f}'.format(value)
            else:
                value_name = str(value)

            # calculate and record the losses
            if _total_weight_root_pixels!=0:
                root_loss_dict[value_name] = _sum_root_loss / _total_weight_root_pixels
            else:  # if there are no root voxels in the validation crops
                root_loss_dict[value_name] = 0

            if _total_weight_soil_pixels!=0:
                soil_loss_dict[value_name] = _sum_soil_loss / _total_weight_soil_pixels
            else:
                soil_loss_dict[value_name] = 0

            if (_total_weight_root_pixels + _total_weight_soil_pixels) != 0:
                loss_dict[value_name] = _sum_total_loss / (_total_weight_root_pixels + _total_weight_soil_pixels)
            else:
                loss_dict[value_name] = 0

            # convert the data type of num_tp, num_fp and num_fn (LongTensors) to float
            _num_tp = float(_num_tp.item())
            _num_fp = float(_num_fp.item())
            _num_fn = float(_num_fn.item())

            # calculate the F1 score of all validation data
            prec = safe_divide(_num_tp, _num_tp + _num_fp)
            recall = safe_divide(_num_tp, _num_tp + _num_fn)
            f_score = safe_divide(2. * prec * recall, prec + recall)

            # record the F1 score
            f1_dict[value_name] = f_score

        if debug:
            print("loss_dict" + "@" * 50)
            print(loss_dict)
            print("f1_dict" + "#" * 50)
            print(f1_dict)

        # write the losses and f1 in the tensorboard writer
        _writer.add_scalars('Validation_loss/VS. {}'.format(feature),
                            loss_dict, step)
        _writer.add_scalars('Validation_root_loss/VS. {}'.format(feature),
                            root_loss_dict, step)
        _writer.add_scalars('Validation_soil_loss/VS. {}'.format(feature),
                            soil_loss_dict, step)
        _writer.add_scalars('validation_f_score/VS. {}'.format(feature),
                            f1_dict, step)


def write_crop_prediction_results(output, sample_batched, batch_data, _train_outputs_dir, step,
                                  soil_loss_list, root_loss_list,
                                  importance_weights=[-1], _reweight_loss=False, mode=None, use_later_time=False):

    # here output and batch_data are numpy array! only the gt in sample_batched is tensor in gpu
    batch_ground_truth = sample_batched['ground_truth']

    output = output.cpu().detach().numpy()
    batch_data = batch_data.numpy()  # .cpu()
    if use_later_time:
        batch_data_later_time = sample_batched['input_later_time'].cpu().numpy()
    soil_loss_list = soil_loss_list.cpu().detach().numpy()
    root_loss_list = root_loss_list.cpu().detach().numpy()
    if _reweight_loss:
        importance_weights = importance_weights.cpu().numpy()

    for i in range(output.shape[0]):
        # save for each data in one mini-batch
        _data_name = sample_batched['data_name'][i]
        _radius = sample_batched['radius'][i].item()
        _rotation = sample_batched['rotation'][i].item()
        _x_flip = sample_batched['x_flip'][i].item()
        _y_flip = sample_batched['y_flip'][i].item()
        _x_y_swap = sample_batched['x_y_swap'][i].item()
        noise_type = sample_batched['noise_type'][i]  #.item()  # a string
        data_index = sample_batched['data_index'][i].item()

        gt = batch_ground_truth[i]

        _maxpool3d = t.nn.MaxPool3d(super_resolution_factor).to(device)
        gt_shrinked = t.unsqueeze(gt, 0)
        gt_shrinked = t.unsqueeze(gt_shrinked, 0)  # convert to the right shape for maxpooling
        gt_shrinked = _maxpool3d(gt_shrinked)
        gt_shrinked = t.squeeze(gt_shrinked, 0)
        gt_shrinked = t.squeeze(gt_shrinked, 0).cpu().numpy()  # convert to the original shape
        del _maxpool3d

        batch_data_i = batch_data[i]
        output_i = output[i]
        output_i = np.expand_dims(output_i, axis=0)
        gt = t.unsqueeze(gt, 0)
        gt = gt.cpu().numpy()

        gt = match_shape_big_to_small(gt, output_i)  # match gt shape to output shape
        gt = np.squeeze(gt, axis=0)
        output_i = np.squeeze(output_i, axis=0)

        os.makedirs(_train_outputs_dir, exist_ok=True)
        dir_to_save = '{0}/Step_{1}/{2}/rad_{3:.2f}/rot_{4}/x_flip_{5}/y_flip_{6}/x_y_swap{7}/'.format(
            _train_outputs_dir, step, _data_name, _radius, _rotation, _x_flip, _y_flip, _x_y_swap)

        os.makedirs(dir_to_save, exist_ok=True)

        if _reweight_loss:
            root_loss = root_loss_list[i]*importance_weights[i]
            soil_loss = soil_loss_list[i]*importance_weights[i]
        else:
            root_loss = root_loss_list[i]
            soil_loss = soil_loss_list[i]

        current_time = float(time())
        output_file_name = "{0}/out_{1}_{2}_rootLoss{3:.3f}_soilLoss{4:.3f}_time{5:.3f}.npz".format(dir_to_save, noise_type,
                                                                                               data_index, root_loss,
                                                                                               soil_loss, current_time)

        # rough estimation of noisiness of this crop by calculating
        # the percentage of non-root voxels with value larger than 0
        noisiness = (batch_data_i[gt_shrinked == 0] > 0).sum() / batch_data_i.size
        del gt_shrinked

        input_file_name = "{0}/in_{1}_{2}_noisiness{3:.3f}_time{4:.3f}.npz".format(dir_to_save, noise_type, data_index,
                                                                              noisiness, current_time)

        if COPY_EVERYTHING_FROM_NILS and mode == 'training':
            gt = gt[MASK_SIDE:-MASK_SIDE, MASK_SIDE:-MASK_SIDE, MASK_SIDE:-MASK_SIDE]
        root_perc = (gt > 0).sum() / gt.size  # percentage of root voxels
        gt_file_name = "{0}/gt_{1}_{2}_rootPerc{3:.3f}_time{4:.3f}.npz".format(dir_to_save, noise_type,
                                                                          data_index, root_perc, current_time)

        with open(output_file_name, 'wb+') as file_to_write:
            # convert data type to Byte
            output_i = (output_i * 255).astype(np.uint8)
            if COPY_EVERYTHING_FROM_NILS and mode=='training':
                output_i = output_i[MASK_SIDE:-MASK_SIDE, MASK_SIDE:-MASK_SIDE, MASK_SIDE:-MASK_SIDE]
            np.savez_compressed(file_to_write, output_i)

        if use_later_time:  # also save the input channel of another time point
            batch_data_later_time_i = batch_data_later_time[i]
            # convert data type to Byte
            # normalizing to (0,255) may make the image look too different from original when its very pale
            # (e.g. random scaling factor is low)
            batch_data_later_time_i = np.clip(batch_data_later_time_i * 255, a_max=255, a_min=0).astype(np.uint8)
            # Not directly multiplying 255, because that may lead to numeric overflow!
            filename = "{0}/in_noise_{1}_laterTimePoint_time{2:.3f}.npz".format(dir_to_save, noise_type, current_time)
            with open(filename, 'wb+') as file_to_write:
                np.savez_compressed(file_to_write, batch_data_later_time_i)

        with open(input_file_name, 'wb+') as file_to_write:
            # convert data type to Byte
            # normalizing to (0,255) may make the image look too different from original when its very pale
            # (e.g. random scaling factor is low)
            batch_data_i = np.clip(batch_data_i * 255, a_max=255, a_min=0).astype(np.uint8)
            # Not directly multiplying 255, because that may lead to numeric overflow!
            np.savez_compressed(file_to_write, batch_data_i)

        # notice: also need to write the ground truth!!
        with open(gt_file_name, 'wb+') as file_to_write:
            # convert data type to Byte
            gt = (gt * 255).astype(np.uint8)
            np.savez_compressed(file_to_write, gt)


def get_gradient_norm(model, norm_type=2):
    num_grad0 = 0
    num_params = 0
    total_norm = 0
    # start_time = time()

    for p in model.parameters():
        if p.requires_grad:
            num_params += 1
            p_norm = p.grad.data.norm(norm_type)
            total_norm += p_norm.item() ** norm_type
            if p_norm.item() <= LOW_THRESHOLD_OF_GRADIENT:
                num_grad0 += 1

    total_norm = total_norm ** (1. / norm_type)
    perc_grad_0 = num_grad0 / num_params

    return total_norm, perc_grad_0  # average gradient norm is too small


def free_cached_memo_and_record_gpu_usage(step, mode):
    t.cuda.empty_cache()  # to prevent memory leaking
    allocated_memory = t.cuda.memory_allocated(device=None)
    cached_memory = t.cuda.memory_cached(device=None)
    writer.add_scalars('GPU memory usage/{}'.format(mode), {'allocated_memory': allocated_memory,
                                                            'cached_memory': cached_memory}, step)


def train_network(train_step):
    run_network(train_step, 'Training')


def validate_network(train_step):
    with t.no_grad():
        run_network(train_step, 'Validation')
        free_cached_memo_and_record_gpu_usage(train_step, 'Validation')


def visualize_outputs(train_step):
    with t.no_grad():
        run_network(train_step, 'Visualization')
        free_cached_memo_and_record_gpu_usage(train_step, 'Visualization')


def save_network(step):
    model_path = '{0}/model_{1}.tm'.format(models_dir, step)
    t.save(
        {
            "net": model.state_dict(),
            "optimizer": adam_optimizer.state_dict()
        }, model_path)
    print("\n")
    print("Model saved at {0}".format(model_path))
    t.cuda.empty_cache()  # to prevent memory leaking


def check_memory_leak():
    print('%'*10, 'Allocated memory', t.cuda.memory_allocated(device=None))
    print('%'*10, 'Cached memory', t.cuda.memory_cached(device=None))


def start_learning_loop():
    if importance_sampling_gradient_norm is True:
        # because we want to start from scratch every time we train a new model
        dir_gradient_norm_weight_matrices = join(weight_base_dir, 'gradient_norm_based')
        if exists(dir_gradient_norm_weight_matrices):
            print("~" * 100)
            print("removing directory:", dir_gradient_norm_weight_matrices)
            shutil.rmtree(dir_gradient_norm_weight_matrices)

    for epoch in range(training_start_step, total_training_steps):
        # visualize_outputs(epoch)
        train_network(epoch)

    writer.export_scalars_to_json(join(env_dir, "all_scalars_tensorboard.json"))
    writer.close()


# save the current constant.py in env folder
if not exists(join(env_dir, 'constants.py')):
    with open(join(env_dir, 'constants.py'), 'w') as f1:
        with open('Utils/constants.py', 'r') as f2:
            for l in f2.readlines():
                f1.write(l)
else:
    print('constants.py already exists in the model directory! Keeping the old constants file, '
          'May need to check if the current running constants comply with it...')


start_learning_loop()
