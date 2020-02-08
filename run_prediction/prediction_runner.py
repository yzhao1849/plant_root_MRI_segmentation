import importlib
import torch as t
import sys
import os
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
import models
from os.path import join, exists
import numpy as np
from torch.utils.data import DataLoader
from time import time
import prediction_utils
from prediction_utils import print_remaining_time, cut_overlappings, add_one_crop_to_whole_prediction, \
    generate_datalist_real, cut_non_training_data, DatasetManager
import json
import gc

# parse the command line arguments
args = prediction_utils.parse_args()
model_dir = args.model_dir  # the dir where the model is stored

# load the constants from constants.txt
model_constants_path = join(model_dir, 'constants.txt')
with open(model_constants_path, 'r') as f:
    f_str = f.read()
    model_constants = json.loads(f_str)

# print('constants.OUTCROP_OVERLAP:', model_constants[''])
print('constants.IF_USE_LATER_TIMEPOINT:',  model_constants['if_use_later_timepoint'])
print()

cuda_ind = args.cuda_device
cuda = not args.no_cuda
output_dir = args.output_dir  # the dir where the visualization output put will be stored
os.makedirs(output_dir, exist_ok=True)
input_data_dir = args.input_data_dir
batch_size = args.batch_size
print('batch_size:', batch_size)

# the following arguments are ones that should be borrowed from the model
net_name = model_constants['net']
super_resolution_factor = model_constants['super_resolution_factor']
np_seed = model_constants['numpy_seed']
torch_seed = model_constants['pytorch_seed']
normalize_input_tensor = model_constants['normalize_all']  # default is False
non_training_crop_size = model_constants['crop_side_length']  # the cropsize for testing is the same as training
non_training_crop_size = (non_training_crop_size, non_training_crop_size, non_training_crop_size)

device = t.device("cuda:{}".format(cuda_ind[0]) if cuda else "cpu")
env_name = model_dir.split('/')[-1]
print("Environment: {0}".format(env_name))
print("Using device: '{}'".format(device))
print('normalize_input_tensor: {}'.format(normalize_input_tensor))

# set the random seed
np.random.seed(np_seed)
t.manual_seed(torch_seed)

# load the model:
# Remove py extension from Net file
net_name = net_name.replace('.py', '')
# Import the class 'Net' from specified package name
Net = importlib.import_module('.{}'.format(net_name), package='models').Net
model = Net()

# Get shape_decreases from the model
shape_decreases = model.calculate_shape_decreases_3D_Net(non_training_crop_size)

# load the saved model
for fn in os.listdir(model_dir):
    if fn.endswith('.tm'):
        saved_model_path = join(model_dir, fn)
        break
# load state dict immediately loads to GPU so we first load to CPU; then move the GPU
state_dict = t.load(saved_model_path, map_location='cpu')
model.load_state_dict(state_dict['net'])

if cuda and len(cuda_ind) > 1:
    model = t.nn.DataParallel(model, device_ids=cuda_ind)


model = model.to(device)
model.eval()  # for correct batch norm

# get the datalist from the real data folder
if input_data_dir == '':
    loaded_data = np.load('/home/user/zhaoy/root_mri/temp/debug_out_dir/test_prediction_runner/input_data_dir/Lupine_22august_mri.npy')
    one_data = True
else:
    loaded_data = None
    one_data = os.path.isfile(input_data_dir)

vis_datalist, input_data_dir = generate_datalist_real(input_data_dir, one_data=one_data, loaded_data=loaded_data)  # only 1 data each time #todo: add parameter to provide the loaded numpy array directly
print('vis_datalist:\n', vis_datalist)  # debugging

for data in vis_datalist:
    splitted_datalist, out_cropsize_xyz, \
    overlappings_xyz, special_overlappings_xyz = cut_non_training_data(data, shape_decreases,
                                                                       non_training_crop_size,
                                                                       super_res_factor=super_resolution_factor,
                                                                       overlaps_after_prediction=(0,0,0))

    print('\n%%%%%%%%% data name:', data[0])
    print('input image padding:', splitted_datalist[0, -3:])

    data_manager = DatasetManager(splitted_datalist,
                                  super_resolution_factor=super_resolution_factor,
                                  normalize=normalize_input_tensor,
                                  diff_times=model_constants['if_use_later_timepoint'],
                                  test_data_dir=input_data_dir,
                                  loaded_data=loaded_data)
    worker_ind = 0
    data_loader = DataLoader(data_manager, batch_size=batch_size, shuffle=False, num_workers=16,
                             worker_init_fn=lambda x: np.random.seed(x + np_seed + worker_ind))

    [data_name, dim_z, dim_x, dim_y] = data
    dim_z, dim_x, dim_y = int(dim_z), int(dim_x), int(dim_y)
    whole_prediction = t.zeros((dim_z*super_resolution_factor,
                                dim_x*super_resolution_factor,
                                dim_y*super_resolution_factor)).to(device)

    loop_start_time = time()
    for i_batch, sample_batched in enumerate(data_loader):
        print_remaining_time(loop_start_time, i_batch, data_loader)

        batch_data = sample_batched['input']  # should not remove_first_axis

        batch_start_position_xyz = sample_batched[
            'out_start_xyz']  # the output start positions in the padded ground truth

        if model_constants['if_use_later_timepoint']:
            batch_input = t.unsqueeze(batch_data, 1)  # shape of (batch_size, 1, z, x, y)
            batch_data_later_time = sample_batched['input_later_time']
            batch_data_later_time = t.unsqueeze(batch_data_later_time, 1)  # shape of (batch_size, 1, z, x, y)
            batch_input = t.cat((batch_input, batch_data_later_time),
                                dim=1)  # shape of (batch_size, 2, z, x, y)
        else:
            batch_input = batch_data

        batch_input = batch_input.to(device)
        batch_output, _ = model(batch_input)

        for i in range(batch_start_position_xyz.size()[0]):  # get one output from the mini-batch

            output = batch_output.detach()[i]  # shape of (z,x,y)
            # need to use detach here! otherwise the whole computation graph will be stored

            out_start_x, out_start_y, out_start_z = batch_start_position_xyz[i]

            # cut to make sure there is no overlapping between crops
            output, crop_x_start, crop_x_end, crop_y_start, crop_y_end, \
            crop_z_start, crop_z_end = cut_overlappings(output,
                                                        (out_start_x, out_start_y, out_start_z),
                                                        whole_prediction.size(),
                                                        overlappings_xyz, special_overlappings_xyz)

            # update the start positions of the output crop
            out_start_x += crop_x_start
            out_start_y += crop_y_start
            out_start_z += crop_z_start

            add_one_crop_to_whole_prediction(output, whole_prediction,
                                             (out_start_x, out_start_y, out_start_z))
            del output

        del batch_input, batch_data, batch_output
        t.cuda.empty_cache()

    # output the whole prediction to the designated folder
    whole_prediction = (whole_prediction.detach().squeeze().cpu().numpy() * 255).astype(np.uint8)  # shape (z,x,y)

    # if the padding is odd, the padding on each side is not symmetric for each dimension,
    # then need to process the prediction output to match the input precisely
    if float(splitted_datalist[0, -1]) % 2 != 0:
        whole_prediction_shifted = np.zeros(whole_prediction.shape).astype(np.uint8)
        whole_prediction_shifted[1:, 1:, 1:] += whole_prediction[:-1, :-1, :-1]
        whole_prediction = whole_prediction_shifted

    # if loaded_data is None:
    if data_name.endswith('.npy'):  # remove the suffix
        data_name = data_name[:-4]
    file_name = "{}/visualized_out_{}.npz".format(output_dir, data_name)

    print("saving the results to files......")
    with open(file_name, 'wb+') as file_to_write:
        np.savez_compressed(file_to_write, whole_prediction)

    del whole_prediction

