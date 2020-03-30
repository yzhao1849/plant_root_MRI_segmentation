import sys
import os
import pickle
from os.path import expanduser, join
import json
import time
from scipy.interpolate import CubicSpline
import numpy as np
from data_root_dir import data_root
from block_reduce_helper import block_reduce, divide_and_reduce


home = expanduser("~")
# mris_dir = data_root+'/Real MRI/'
# manual_rec_dir = mris_dir + '/manual-reconstructiontemp/'
# manual_rec_dir = '/home/stud/zhaoy/root_mri/experiments/voxellization_experiments'  # bigcuda4
# manual_rec_dir = '/home/user/zhaoy/root_mri/experiments/voxellization_experiments'  #cuda4
# manual_rec_dir = '/home/user/zhaoy/root_mri/experiments/'  #cuda4
#manual_rec_dir = "temp/"

high_res_factor = 2 ** 0
x_stretch = np.asarray([1,1,1])
ds_r = 1/3

def create_a_volume(params):
    volumes_creation_start = time.time()
    [dataset_name, dim, depth_axis, radius_multiplier, [x_translate, y_translate, z_translate], [x_min, y_min, z_min],
     [x_max, y_max, z_max], fact, z_fact, z_rotation, num, double_stream, data_type,
     reverse_x, reverse_y, reverse_z] = params
    # fact is used to rescale the annotation coordinates to the mri coordinates
    fact_x, fact_y, fact_z = fact
    print('**********fact:{}**********z_fact:{}'.format(fact, z_fact))

    # data_type = 'root_sys'
    print('data_type:', data_type)

    # The console output is written to a txt file also
    def print_func(str="\n", end="\n"):
        print(str, file=double_stream, end=end)

    inputdir = manual_rec_dir + "/" + dataset_name + "/"
    print('inputdir:', inputdir)

    os.chdir(inputdir)

    volume_create_start = time.time()

    with open("structure.pydump", 'rb') as f:

        loaded_pickle = pickle.load(f)
        extra_params = None
        if len(loaded_pickle)==12:
            [root_age, num_of_seeds, (root_id_x_y), (rdm_sdm_larea), (
                strong_conc_), num_axes, num_branches, num_seg_records, segment_records, num_grw_brn_tip, tip_records,
             extra_params] = loaded_pickle
        elif len(loaded_pickle)==11:
            [root_age, num_of_seeds, (root_id_x_y), (rdm_sdm_larea), (
                strong_conc_), num_axes, num_branches, num_seg_records, segment_records, num_grw_brn_tip, tip_records] \
                = loaded_pickle

    if all(b == -1 for b in [x_min, y_min, z_min, x_max, y_max, z_max]) and extra_params is not None:
        [x_min, y_min, z_min, x_max, y_max, z_max] = [float(extra_params.get("x_min", 0)),
                                                      float(extra_params.get("y_min", 0)),
                                                      float(extra_params.get("z_min", 0)),
                                                      float(extra_params.get("x_max", 0)),
                                                      float(extra_params.get("y_max", 0)),
                                                      float(extra_params.get("z_max", 0)),
                                                      ]

    segment_records[:, 1] += x_translate
    segment_records[:, 2] += y_translate
    segment_records[:, 3] += z_translate

    segment_records[:, 1] *= fact_x
    segment_records[:, 2] *= fact_y
    segment_records[:, 3] *= fact_z
    if depth_axis == 'x':
        segment_records[:, 1] *= z_fact
    elif depth_axis == 'y':
        segment_records[:, 2] *= z_fact
    else:
        segment_records[:, 3] *= z_fact

    # moved to line 65: it is more intuitive to first do the translation and then scaling
    # segment_records[:, 1] += x_translate
    # segment_records[:, 2] += y_translate
    # segment_records[:, 3] += z_translate

    avg_fact = (fact_x + fact_y + fact_z) / 3
    if data_type == 'root_sys':
        # origin = [x_translate, y_translate, z_translate]
        segment_records[:, 7] *= avg_fact
        segment_records[:, 8] *= avg_fact * avg_fact
    else:
        segment_records[:, 5] *= avg_fact

    x_translate = y_translate = z_translate = None  # Ensure that these params are not reused.

    # Arrange the axises so that 'z' is the depth

    x_coords = segment_records[:, 1].copy()
    y_coords = segment_records[:, 2].copy()
    z_coords = segment_records[:, 3].copy()
    if depth_axis == 'x':
        segment_records[:, 1] = z_coords
        segment_records[:, 3] = x_coords  # put the depth axis at the 4th column of segment_records
        x_min, x_max, z_min, z_max = (z_min, z_max, x_min, x_max)
        dim[0], dim[2] = (dim[2], dim[0])
        reverse_x, reverse_z = reverse_z, reverse_x
    if depth_axis == 'y':
        segment_records[:, 2] = z_coords
        segment_records[:, 3] = y_coords
        y_min, y_max, z_min, z_max = (z_min, z_max, y_min, y_max)
        dim[1], dim[2] = (dim[2], dim[1])
        reverse_y, reverse_z = reverse_z, reverse_y
    x_coords = y_coords = z_coords = None

    x_middle = (x_max + x_min) / 2.
    y_middle = (y_max + y_min) / 2.

    segment_records[:, 1] -= x_middle
    segment_records[:, 2] -= y_middle

    # rotation around center of z
    x_rotated = segment_records[:, 1] * np.cos(z_rotation) - segment_records[:, 2] * np.sin(z_rotation)
    y_rotated = segment_records[:, 2] * np.cos(z_rotation) + segment_records[:, 1] * np.sin(z_rotation)
    segment_records[:, 1] = x_rotated
    segment_records[:, 2] = y_rotated

    segment_records[:, 1] += x_middle
    segment_records[:, 2] += y_middle


    def get_segment_by_id(seg_id):

        if seg_id == 0. and data_type == 'root_sys':
            seg = get_segment_by_id(1.).copy()
            # seg[1:4] = origin
            return seg
        return segment_records[segment_records[:, 0] == seg_id][0]

    def get_radius_from_area(seg_id):
        segment = get_segment_by_id(seg_id)
        try:
            rad = segment[8] / np.pi / 2 / segment[7]
        except:
            print_func("lol")
        # print('-------- rescaling radius with radius_multiplier {}', radius_multiplier)
        return rad * radius_multiplier

    # spacing = [(x_max - x_min) / float(dim[0]), (y_max - y_min) / float(dim[1]), (z_max - z_min) / float(dim[2])]
    print('xmax, xmin, ymax, ymin, zmax, zmin:', x_max, x_min, y_max, y_min, z_max, z_min)
    print('dim 123:', dim)

    # x_r = np.linspace(x_min + spacing[0] / 2., x_max - spacing[0] / 2., dim[0])
    # y_r = np.linspace(y_min + spacing[1] / 2., y_max - spacing[1] / 2., dim[1])  # originally spacing[0]: is that a bug?
    # z_r = np.linspace(z_min + spacing[2] / 2., z_max - spacing[2] / 2., dim[2])

    occupancy_grid = np.zeros(np.array(dim).astype(int))  # , dtype=np.uint16)
    # debugging
    print('shape of occupancy:', occupancy_grid.shape)
    intensity_grid = np.zeros(np.array(dim).astype(int))  #, dtype=np.uint16)
    radius_grid = np.zeros(np.array(dim).astype(int))  #, dtype=np.float16)
    distance_grid = np.zeros(np.array(dim).astype(int))  #, dtype=np.float16)

    branch_tip_ids = tip_records[:, 4]
    ctr = 1
    print()
    branch_tip_ctr = 0
    progress_format = "\r{}: {:xd}/{:xd}".replace("x", str((int(np.ceil(np.log10(len(branch_tip_ids)))))))
    set_visited_seg = set()  # store the already voxelized segments, to avoid repeated voxelization

    # assert depth_fact == 3
    # assert res_mult_fac == 2
    for branch_tip_id in branch_tip_ids:
        branch_tip_ctr = branch_tip_ctr + 1
        # if branch_tip_ctr>1:
        #     continue
        print_func(progress_format.format(int(num), int(branch_tip_ctr), int(len(branch_tip_ids))), end="")
        # double_stream.write()
        # double_stream.flush()
        branch_points = list()
        branch_radii = list()

        current_id = branch_tip_id
        first_seg = get_segment_by_id(current_id)

        if data_type == 'root_sys':
            # We traverse from tip to beginning of root
            # Some segments may be visited more than once
            while current_id != 0:

                seg = get_segment_by_id(current_id)
                prev_seg_id = seg[4]
                prev_seg = get_segment_by_id(prev_seg_id)

                middle_loc = (seg[1:4] + prev_seg[1:4]) / 2.
                beginning = prev_seg[1:4]

                branch_points.append(middle_loc)
                branch_points.append(beginning)

                middle_radius = get_radius_from_area(current_id)
                beginning_radius = (get_radius_from_area(prev_seg_id) + middle_radius) / 2.

                branch_radii.append(middle_radius)
                branch_radii.append(beginning_radius)

                if current_id in set_visited_seg:  # once reach a segment that has already been visited, stop
                    break
                else:
                    set_visited_seg.add(current_id)
                current_id = prev_seg_id

        else:
            while current_id != -1.:  # -1 means prev is tree, we reached root

                seg = get_segment_by_id(current_id)
                prev_seg_id = seg[4]

                radius = seg[5] * radius_multiplier

                loc = seg[1:4]

                branch_points.append(loc)
                branch_radii.append(radius)

                if current_id in set_visited_seg:
                    break
                else:
                    set_visited_seg.add(current_id)
                current_id = prev_seg_id



        branch_radii = [0] + branch_radii
        branch_radii = branch_radii[:-1]

        # revert the branch order: new order is from shoot to tip
        branch_radii = np.flip(np.array(branch_radii),axis=0)
        branch_points = np.flip(np.array(branch_points),axis=0)
        print('original x max 0:', np.max(branch_points[:, 0]))
        print('original x min 0:', np.min(branch_points[:, 0]))
        print('original y max 0:', np.max(branch_points[:, 1]))
        print('original y min 0:', np.min(branch_points[:, 1]))
        print('original z max 0:', np.max(branch_points[:, 2]))
        print('original z min 0:', np.min(branch_points[:, 2]))

        # print('branch_radii:')
        # for l in branch_radii:
        #     print(l)

        # # transform from the annotation distance to the image distance (according to the resolutions)
        # branch_points[:, 0] /= spacing[0]
        # branch_points[:, 1] /= spacing[1]
        # branch_points[:, 2] /= spacing[2]

        if len(branch_radii)==1 and branch_radii[0]==0:  # normally does not occur
            continue

        # branch_radii /= spacing[2]  # todo: make it anisotropic
        # print('spacing:', spacing)
        # print('spacing[2]/spacing[0]:', spacing[2]/spacing[0])
        global x_stretch
        # x_stretch = np.array([spacing[2]/spacing[0], spacing[2]/spacing[1], 1])  # the radius of
        # print('x_stretch', x_stretch)
        # print('spacing:', spacing)

        local_max_rad = np.max(branch_radii)
        # if branch_tip_ctr == 64:
        # print('local_max_rad:', local_max_rad)
        gaussian_trunc_bound = 3  # should be the same for truncated_gauss_blob()
        # add 15 to make sure the gaussian blob range will be within this local crop,
        # because the interpolation result of cubic spline may result in larger/smaller values than the original max/min
        local_x_max = int(np.max(branch_points[:, 0]) + int(gaussian_trunc_bound * (int(local_max_rad) + 1) * x_stretch[0])) + 15
        local_x_min = int(np.min(branch_points[:, 0]) - int(gaussian_trunc_bound * (int(local_max_rad) + 1) * x_stretch[0])) - 15
        local_y_max = int(np.max(branch_points[:, 1]) + int(gaussian_trunc_bound * (int(local_max_rad) + 1) * x_stretch[1])) + 15
        local_y_min = int(np.min(branch_points[:, 1]) - int(gaussian_trunc_bound * (int(local_max_rad) + 1) * x_stretch[1])) - 15
        local_z_max = int(np.max(branch_points[:, 2]) + int(gaussian_trunc_bound * (int(local_max_rad) + 1) * x_stretch[2])) + 15
        local_z_min = int(np.min(branch_points[:, 2]) - int(gaussian_trunc_bound * (int(local_max_rad) + 1) * x_stretch[2])) - 15

        print('original local_x_min:{}, local_x_max:{}, local_y_min:{}, local_y_max:{}, local_z_min:{}, local_z_max:{}'.format(local_x_min, local_x_max, local_y_min, local_y_max, local_z_min, local_z_max))

        print('original x max:', np.max(branch_points[:, 0]))
        print('original x min:', np.min(branch_points[:, 0]))
        print('original y max:', np.max(branch_points[:, 1]))
        print('original y min:', np.min(branch_points[:, 1]))
        print('original z max:', np.max(branch_points[:, 2]))
        print('original z min:', np.min(branch_points[:, 2]))
        # print('x extension:', int(gaussian_trunc_bound * (int(local_max_rad) + 1) * x_stretch[0]) + 5)
        # print('y extension:', int(gaussian_trunc_bound * (int(local_max_rad) + 1) * x_stretch[1]) + 5)
        # print('z extension:', int(gaussian_trunc_bound * (int(local_max_rad) + 1) * x_stretch[2]) + 5)


        # voxelize each branch before adding to occupancy
        current_branch_occupancy = np.zeros((local_x_max - local_x_min,
                                             local_y_max - local_y_min,
                                             local_z_max - local_z_min,))
        # current_branch_occupancy = np.zeros(np.array(dim).astype(int))

        # transform the locations to the relative local coordinates
        branch_points[:, 0] -= local_x_min
        branch_points[:, 1] -= local_y_min
        branch_points[:, 2] -= local_z_min

        # the indices of the usable area of the local image
        x_end = current_branch_occupancy.shape[0]
        x_start = 0
        y_end = current_branch_occupancy.shape[1]
        y_start = 0
        z_end = current_branch_occupancy.shape[2]
        z_start = 0

        original_local_x_max = local_x_max
        original_local_x_min = local_x_min
        original_local_y_max = local_y_max
        original_local_y_min = local_y_min
        original_local_z_max = local_z_max
        original_local_z_min = local_z_min

        # now make sure the indices do not exceed the boundary of the whole image
        local_x_max = min(dim[0], local_x_max)
        local_x_min = max(0, local_x_min)
        local_y_max = min(dim[1], local_y_max)
        local_y_min = max(0, local_y_min)
        local_z_max = min(dim[2], local_z_max)
        local_z_min = max(0, local_z_min)
        # print('local_x_min:{} - local_x_max:{}, local_y_min:{} - local_y_max:{}, '
        #       'local_z_min:{} - local_z_max:{}'.format(local_x_min, local_x_max, local_y_min,
        #                                                local_y_max, local_z_min, local_z_max))
        # print('original_local_x_min:{} - original_local_x_max:{}, original_local_y_min:{} - original_local_y_max:{}, '
        #       'original_local_z_min:{} - original_local_z_max:{}'.format(original_local_x_min, original_local_x_max,
        #                                                                  original_local_y_min, original_local_y_max,
        #                                                                  original_local_z_min, original_local_z_max))

        # adjust the indices of the usable area of the local image accordingly
        x_end -= original_local_x_max - local_x_max
        x_start -= original_local_x_min - local_x_min
        y_end -= original_local_y_max - local_y_max
        y_start -= original_local_y_min - local_y_min
        z_end -= original_local_z_max - local_z_max
        z_start -= original_local_z_min - local_z_min

        # num_pieces = 4
        radii_spline = CubicSpline([el for el in list(range(len(branch_points)))], branch_radii)  # why not used?? Using linear interpolation instead
        # radii_spline_d = radii_spline.derivative(1)
        loc_spline = CubicSpline([el for el in list(range(len(branch_points)))], branch_points)
        # loc_spline_d = loc_spline.derivative(1)

        # print('branch_radii:')
        # for l in branch_radii:
        #     print(l)
        # print('branch_points:')
        # for l in branch_points:
        #     print(l)

        # print('loc_spline 0.5, 10.5, 20.5')
        # print(loc_spline(0.5))
        # print(loc_spline(10.5))
        # print(loc_spline(20.5))

        # print('branch:')
        branch_info = np.concatenate((branch_points, np.expand_dims(branch_radii, axis=1)), axis=1)


        # print(branch_info)
        current_branch = generate_branch(branch_info, loc_spline, radii_spline)

        s = 0
        s_small_step = None
        second_last_radius = None
        # idx = 0
        while True:
            xyz, r, s_idx = current_branch(s)
            # print('number of annotated points:', len(branch_info))
            # print('s:{}, r:{}'.format(s, r))
            # print('s_idx:{}, r:{}'.format(s_idx, r))
            # print('xyz: {}, r: {}, s_idx: {}'.format(xyz, r, s_idx))
            if r <= 0 or s_idx>=len(branch_info):
                break

            # # only voxelize the points that are annotated
            # if idx>=branch_info.shape[0]:
            #     break
            # xyz = branch_info[idx, 0:3]
            # r = float(branch_info[idx, 3])
            # idx += 1

            # xyz = xyz * x_stretch
            x_range, y_range, z_range, blob = truncated_gauss_blob(xyz, r)
            # print('x_stretch', x_stretch)
            # print('xyz: {}, r: {}, s_idx: {}'.format(xyz, r, s_idx))
            # print('x_range[1]-x_range[0]: {}-{}, y_range[1]-y_range[0]: {}-{}, z_range[1]-z_range[0]: {}-{}'.format(x_range[1],x_range[0], y_range[1],y_range[0], z_range[1],z_range[0]))
            # print()
            # print('shape of blob:', blob.shape)
            # print('shape of current_branch_occupancy', current_branch_occupancy.shape)
            # print('shape of occupancy_grid[x_range[0]:x_range[1], y_range[0]:y_range[1], z_range[0]:z_range[1]]:', occupancy_grid[x_range[0]:x_range[1], y_range[0]:y_range[1], z_range[0]:z_range[1]].shape)

            # cut off the part of blob that exceeds the boundary of the image
            dim_x_min = 0
            dim_y_min = 0
            dim_z_min = 0
            dim_x_max, dim_y_max, dim_z_max = current_branch_occupancy.shape
            x_start_i = max(x_range[0], dim_x_min)
            y_start_i = max(y_range[0], dim_y_min)
            z_start_i = max(z_range[0], dim_z_min)
            x_end_i = min(x_range[1], dim_x_max)
            y_end_i = min(y_range[1], dim_y_max)
            z_end_i = min(z_range[1], dim_z_max)
            blob = blob[x_start_i-x_range[0]:x_end_i-x_range[0], y_start_i-y_range[0]:y_end_i-y_range[0],
                        z_start_i-z_range[0]:z_end_i-z_range[0]]
            #
            # # print('x_start-x_end: {}-{}, y_start-y_end: {}-{}, z_start-z_end: {}-{}'.format(
            # #     x_start, x_end, y_start, y_end, z_start, z_end))
            # # print('shape of blob modified:', blob.shape)
            #
            # current_branch_occupancy[x_start:x_end, y_start:y_end, z_start:z_end] += blob
            # # occupancy_grid[x_start:x_end, y_start:y_end, z_start:z_end] += blob

            # report = True
            # if x_range[1]>current_branch_occupancy.shape[0]:
            #     print('x exceeding range')
            #     report = True
            # if y_range[1]>current_branch_occupancy.shape[1]:
            #     print('y exceeding range')
            #     report = True
            # if z_range[1]>current_branch_occupancy.shape[2]:
            #     print('z exceeding range')
            #     report = True
            #
            # if report:
            #     print('x_range[1]-x_range[0]: {}-{}, y_range[1]-y_range[0]: {}-{}, z_range[1]-z_range[0]: {}-{}'.format(
            #         x_range[1], x_range[0], y_range[1], y_range[0], z_range[1], z_range[0]))
            #     print('shape of blob:', blob.shape)
            #     print('shape of current_branch_occupancy', current_branch_occupancy.shape)
            #
            # print()

            current_branch_occupancy[x_range[0]:x_range[1], y_range[0]:y_range[1], z_range[0]:z_range[1]] += blob

            if s_idx>=len(branch_info)-2:
                # when the radius is already very small, the step of s might be too small then the loop never ends
                # so when approaching the tip of branch, use some constant step size
                # if s_small_step is None:
                #     # last_segment_length = np.linalg.norm(branch_info[-1, :3] - branch_info[-2, :3])
                #     s_small_step = ds_r * r / 2  #last_segment_length/3  # use as a constant step size for the later iterations
                # s += s_small_step

                # if second_last_radius is None:
                #     second_last_radius = r
                s += ds_r * r
                if x_range[1]-x_range[0]<1 or y_range[1]-y_range[0]<1 or z_range[1]-z_range[0]<1:
                    # when the blob volume is 0, stop voxelization
                    break
                # s += ds_r * r * (second_last_radius/r)**0.3  # to compensate for rapid radius reduction
                # print('small step size:', ds_r * r * (second_last_radius/r)**0.3)

                # s += min(spacing)  # move at least one voxel
                # # print('last seg')

            else:
                s += ds_r * r
            # print('s:{}, r:{}'.format(s, r))

        start = time.time()
        # after voxelizing the whole branch, rectify it
        # print("rectifying current_branch_occupancy")
        current_branch_occupancy = (current_branch_occupancy - 0.5) * 5 + 0.5
        # print('num of voxels in current_branch_occupancy > 10:', (current_branch_occupancy > 10).sum())
        # print('num of voxels in current_branch_occupancy < 0:', (current_branch_occupancy < 0).sum())
        # print('max/min of current_branch_occupancy:', current_branch_occupancy.max(), current_branch_occupancy.min())
        current_branch_occupancy[current_branch_occupancy > 10] = 10
        current_branch_occupancy[current_branch_occupancy < 0] = 0
        current_branch_occupancy /= 10

        print('shape of occupancy_grid:', occupancy_grid.shape)
        print('local_x_min:{}, local_x_max:{}, local_y_min:{}, local_y_max:{}, local_z_min:{}, local_z_max:{}'.format(local_x_min, local_x_max, local_y_min, local_y_max, local_z_min, local_z_max))
        print('shape of current_branch_occupancy:', current_branch_occupancy.shape)
        print('x_start:{}, x_end:{}, y_start:{}, y_end:{}, z_start:{}, z_end:{}'.format(x_start, x_end, y_start, y_end, z_start, z_end))
        # add this new branch to the occupancy grid
        occupancy_grid[local_x_min:local_x_max,
                    local_y_min:local_y_max,
                    local_z_min:local_z_max] = np.maximum(current_branch_occupancy[x_start:x_end,
                                                                                  y_start:y_end,
                                                                                  z_start:z_end],
                                                          occupancy_grid[local_x_min:local_x_max,
                                                                      local_y_min:local_y_max,
                                                                      local_z_min:local_z_max])
        # and set to some random value (use a random intensity for a certain branch)
        random_multiplier = np.random.uniform(0.4, 1.)
        current_branch_occupancy *= random_multiplier
        # then add this new branch to the intensity grid
        intensity_grid[local_x_min:local_x_max,
                       local_y_min:local_y_max,
                       local_z_min:local_z_max] = np.maximum(current_branch_occupancy[x_start:x_end,
                                                                                      y_start:y_end,
                                                                                      z_start:z_end],
                                                             intensity_grid[local_x_min:local_x_max,
                                                                            local_y_min:local_y_max,
                                                                            local_z_min:local_z_max])
        # print('time used for rectifying the current branch:', time.time() - start)


    occupancy_grid *= 255
    intensity_grid *= 255

    # if the annotation axis direction is reversed compared to the MRI
    if reverse_x:
        print('reversing x (first) axis...')
        occupancy_grid = np.flip(occupancy_grid, 0)
        intensity_grid = np.flip(intensity_grid, 0)
    if reverse_y:
        print('reversing y (second) axis...')
        occupancy_grid = np.flip(occupancy_grid, 1)
        intensity_grid = np.flip(intensity_grid, 1)
    if reverse_z:
        print('reversing z (third) axis...')
        occupancy_grid = np.flip(occupancy_grid, 2)
        intensity_grid = np.flip(intensity_grid, 2)

    # occupancy_grid = np.clip(occupancy_grid, 0, 255).astype(np.uint8)  # for faster reducing using c: clipping not necessary?
    # intensity_grid = np.clip(intensity_grid, 0, 255).astype(np.uint8)

    assert alias_fact==3
    # assert res_mult_fac==2
    half_res_occ = block_reduce(occupancy_grid, 2, "mean")
    half_res_intensity = block_reduce(intensity_grid, 2, "mean")

    # occupancy_grid = occupancy_grid.astype(np.float32)  # for the following computation not to overflow
    # intensity_grid = intensity_grid.astype(np.float32)
    # half_res_occ = half_res_occ.astype(np.float32)
    # half_res_intensity = half_res_intensity.astype(np.float32)

    occ_2x_res = (occupancy_grid[:,:,0::3] + occupancy_grid[:,:,1::3] + occupancy_grid[:,:,2::3]) / 3
    intensity_2x_res = (intensity_grid[:,:,0::3] + intensity_grid[:,:,1::3] + intensity_grid[:,:,2::3]) / 3
    occ_1x_antialiased = (half_res_occ[:,:,0::3] + half_res_occ[:,:,1::3] + half_res_occ[:,:,2::3]) / 3
    intensity_1x_antialiased = (half_res_intensity[:,:,0::3] + half_res_intensity[:,:,1::3] + half_res_intensity[:,:,2::3]) / 3
    occ_1x_aliased = half_res_occ[:, :, 1::3]
    intensity_1x_aliased = half_res_intensity[:, :, 1::3]

    out_dir = '{0}/r_factor_{1:.2f}/rot_{2}/x_flip_0/y_flip_0/x_y_swap_0'.format(
        inputdir,
        radius_multiplier,
        int(round(z_rotation * 180. / np.pi))
    )
    os.makedirs(out_dir, exist_ok=True)


    np.savez('{0}/original_{1}x{2}x{3}_occupancy.npz'.format(out_dir,
                                                             occupancy_grid.shape[0],
                                                             occupancy_grid.shape[1],
                                                             occupancy_grid.shape[2]), occupancy_grid.astype(np.uint8))
    np.savez('{0}/original_{1}x{2}x{3}_intensity.npz'.format(out_dir,
                                                             intensity_grid.shape[0],
                                                             intensity_grid.shape[1],
                                                             intensity_grid.shape[2]), intensity_grid.astype(np.uint8))

    # np.savez('{0}/half_res_occ_{1}x{2}x{3}_occupancy.npz'.format(out_dir,
    #                                                              half_res_occ.shape[0],
    #                                                              half_res_occ.shape[1],
    #                                                              half_res_occ.shape[2]), half_res_occ.astype(np.uint8))
    # np.savez('{0}/half_res_intensity_{1}x{2}x{3}_intensity.npz'.format(out_dir,
    #                                                                    half_res_intensity.shape[0],
    #                                                                    half_res_intensity.shape[1],
    #                                                                    half_res_intensity.shape[2]),
    #          half_res_intensity.astype(np.uint8))

    occ_2x_res_dim = occ_2x_res.shape
    occ_2x_res_file_name = '{0}/occ_2x_res_{1}x{2}x{3}.npz'.format(out_dir,
                                                                             occ_2x_res_dim[0],
                                                                             occ_2x_res_dim[1],
                                                                             occ_2x_res_dim[2])
    intensity_2x_res_file_name = '{0}/intensity_2x_res_{1}x{2}x{3}.npz'.format(out_dir,
                                                                                         occ_2x_res_dim[0],
                                                                                         occ_2x_res_dim[1],
                                                                                         occ_2x_res_dim[2])
    np.savez(join(out_dir, occ_2x_res_file_name), occ_2x_res.astype(np.uint8))
    np.savez(join(out_dir, intensity_2x_res_file_name), intensity_2x_res.astype(np.uint8))

    occ_1x_antialiased_dim = occ_1x_antialiased.shape
    occ_1x_antialiased_file_name = '{0}/occ_1x_antialiased_{1}x{2}x{3}.npz'.format(out_dir,
                                                                                             occ_1x_antialiased_dim[0],
                                                                                             occ_1x_antialiased_dim[1],
                                                                                             occ_1x_antialiased_dim[2])
    intensity_1x_antialiased_file_name = '{0}/intensity_1x_antialiased_{1}x{2}x{3}.npz'.format(out_dir,
                                                                                                 occ_1x_antialiased_dim[0],
                                                                                                 occ_1x_antialiased_dim[1],
                                                                                                 occ_1x_antialiased_dim[2])
    np.savez(join(out_dir, occ_1x_antialiased_file_name), occ_1x_antialiased.astype(np.uint8))
    np.savez(join(out_dir, intensity_1x_antialiased_file_name), intensity_1x_antialiased.astype(np.uint8))

    occ_1x_aliased_dim = occ_1x_aliased.shape
    occ_1x_aliased_file_name = '{0}/occ_1x_aliased_{1}x{2}x{3}.npz'.format(out_dir,
                                                                                     occ_1x_aliased_dim[0],
                                                                                     occ_1x_aliased_dim[1],
                                                                                     occ_1x_aliased_dim[2])
    intensity_1x_aliased_file_name = '{0}/intensity_1x_aliased_{1}x{2}x{3}.npz'.format(out_dir,
                                                                                     occ_1x_aliased_dim[0],
                                                                                     occ_1x_aliased_dim[1],
                                                                                     occ_1x_aliased_dim[2])
    np.savez(join(out_dir, occ_1x_aliased_file_name), occ_1x_aliased.astype(np.uint8))
    np.savez(join(out_dir, intensity_1x_aliased_file_name), intensity_1x_aliased.astype(np.uint8))


    print_func("Volume created in {}".format(time.time() - volume_create_start))

    # spacing_x, spacing_y, spacing_z = (spacing[0], spacing[1], spacing[2])
    # for i in range(0, 1):  # original: range(2, 4)  # second: range(3, 4)
        # print_func("factor {0}".format(i))
        # factor = int(2 ** i)
        # if factor == 1:
        #     downsampled_occupancy_grid = occupancy_grid
        #     downsampled_intensity_grid = intensity_grid
        #     downsampled_radius_grid = radius_grid
        #     downsampled_distance_grid = distance_grid
        # else:
        #     downsampled_occupancy_grid = divide_and_reduce(occupancy_grid, factor, "mean")  # type: np.ndarray
        #     downsampled_occupancy_grid = downsampled_occupancy_grid.astype(np.ubyte)
        #     downsampled_intensity_grid = divide_and_reduce(intensity_grid, factor, "mean")  # type: np.ndarray
        #     downsampled_intensity_grid = downsampled_intensity_grid.astype(np.ubyte)
        #
        #     downsampled_radius_grid = divide_and_reduce(radius_grid, factor, "max")  # type: np.ndarray
        #     downsampled_radius_grid = downsampled_radius_grid.astype(np.float16)
        #
        #     downsampled_distance_grid = divide_and_reduce(distance_grid, factor, "max")  # type: np.ndarray
        #     downsampled_distance_grid = downsampled_distance_grid.astype(np.float16)
        #
        # max_distance = downsampled_distance_grid.max()
        # downsampled_distance_grid *= 255 / max_distance
        # max_radius = downsampled_radius_grid.max()
        # downsampled_radius_grid *= 255 / max_radius
        # # print_func("Unique values: {0}".format(np.unique(reduced_image)))

        # # todo: currently change this for quicker experiment
        # for x_flip in range(1):  # changed from range(2)
        #     for y_flip in range(1):  # changed from range(2)
        #         for x_y_swap in range(1):   # changed from range(2)
        #             process(downsampled_occupancy_grid, downsampled_intensity_grid, downsampled_radius_grid,
        #                     downsampled_distance_grid, x_flip, y_flip, x_y_swap, spacing_x, spacing_y, spacing_z,
        #                     inputdir, radius_multiplier, z_rotation, dataset_name, max_distance, max_radius, print_func)

    volumes_creation_end = time.time()
    print_func("Volume {}:{} creation took {} seconds.".format(dataset_name, num,
                                                                   volumes_creation_end - volumes_creation_start))


def process(downsampled_occupancy, downsampled_intensity, downsampled_radius, downsampled_distance, x_flip, y_flip,
            x_y_swap, spacing_x, spacing_y, spacing_z, inputdir, radius_multiplier, z_rotation, dataset_name,
            max_distance, max_radius, print_func):
    flipops_start = time.time()
    if x_flip == 1:
        o_x_flipped = np.flip(downsampled_occupancy, axis=0)
        i_x_flipped = np.flip(downsampled_intensity, axis=0)
        r_x_flipped = np.flip(downsampled_radius, axis=0)
        d_x_flipped = np.flip(downsampled_distance, axis=0)
    else:
        o_x_flipped = downsampled_occupancy
        i_x_flipped = downsampled_intensity
        r_x_flipped = downsampled_radius
        d_x_flipped = downsampled_distance

    if y_flip == 1:
        o_y_flipped = np.flip(o_x_flipped, axis=1)
        i_y_flipped = np.flip(i_x_flipped, axis=1)
        d_y_flipped = np.flip(d_x_flipped, axis=1)
        r_y_flipped = np.flip(r_x_flipped, axis=1)
    else:
        o_y_flipped = o_x_flipped
        i_y_flipped = i_x_flipped
        d_y_flipped = d_x_flipped
        r_y_flipped = r_x_flipped

    if x_y_swap == 1:
        o_x_y_swapped = np.swapaxes(o_y_flipped, 0, 1)
        i_x_y_swapped = np.swapaxes(i_y_flipped, 0, 1)
        r_x_y_swapped = np.swapaxes(r_y_flipped, 0, 1)
        d_x_y_swapped = np.swapaxes(d_y_flipped, 0, 1)
        spacing_x, spacing_y = spacing_y, spacing_x
    else:
        i_x_y_swapped = i_y_flipped
        o_x_y_swapped = o_y_flipped
        d_x_y_swapped = d_y_flipped
        r_x_y_swapped = r_y_flipped

    flipops_end = time.time()
    # my_print("Flip ops: {0}".format(flipops_end-flipops_start))
    out_dir = '{0}/r_factor_{1:.2f}/rot_{2}/x_flip_{3}/y_flip_{4}/x_y_swap_{5}'.format(
        inputdir,
        radius_multiplier,
        int(round(z_rotation * 180. / np.pi)),
        x_flip,
        y_flip,
        x_y_swap
    )
    print_func(out_dir)
    print_func(i_x_y_swapped.shape)
    reduced_dim = i_x_y_swapped.shape

    meta_data = {'Dataset Name': dataset_name,
                 # 'Dimension': '{0}x{1}x{2}'.format(reduced_dim[0], reduced_dim[1], reduced_dim[2]),
                 'Rotation around z': int(round(z_rotation * 180. / np.pi)),
                 'Max-distance_root_center': float(max_distance),
                 'Max-root-radius': float(max_radius),
                 "Radius-multiplier": radius_multiplier,
                 'Data Type': 'Artificial',
                 'X-Flip': x_flip,
                 'Y-Flip': y_flip,
                 'X-Y-swap': x_y_swap,
                 'X-Spacing': spacing_x,
                 'Y-Spacing': spacing_y,
                 'Z-Spacing': spacing_z,
                 }
    os.makedirs(out_dir, exist_ok=True)
    occupancy_grid_file_name = '{0}/{1}x{2}x{3}_occupancy.npz'.format(out_dir,
                                                                      reduced_dim[0],
                                                                      reduced_dim[1],
                                                                      reduced_dim[2])
    intensity_grid_file_name = '{0}/{1}x{2}x{3}_intensity.npz'.format(out_dir,
                                                                      reduced_dim[0],
                                                                      reduced_dim[1],
                                                                      reduced_dim[2])
    distance_grid_file_name = '{0}/{1}x{2}x{3}_distance.npz'.format(out_dir,
                                                                    reduced_dim[0],
                                                                    reduced_dim[1],
                                                                    reduced_dim[2])
    radius_grid_file_name = '{0}/{1}x{2}x{3}_radius.npz'.format(out_dir,
                                                                reduced_dim[0],
                                                                reduced_dim[1],
                                                                reduced_dim[2])
    zipped_grid_file_name = '{0}/{1}x{2}x{3}.npz'.format(out_dir,
                                                         reduced_dim[0],
                                                         reduced_dim[1],
                                                         reduced_dim[2])
    meta_data_file_name = '{0}/meta.json'.format(out_dir)

    save_start = time.time()

    # d_x_y_swapped *= 255 / d_x_y_swapped.max()
    # r_x_y_swapped *= 255 / r_x_y_swapped.max()

    np.savez(occupancy_grid_file_name, o_x_y_swapped)  # todo: delete this! only for temporary experiment use!
    np.savez(intensity_grid_file_name, i_x_y_swapped)  # todo: delete this! only for temporary experiment use!
    # np.savez_compressed(occupancy_grid_file_name, o_x_y_swapped)
    # np.savez_compressed(intensity_grid_file_name, i_x_y_swapped)
    # np.savez_compressed(distance_grid_file_name, d_x_y_swapped)
    # np.savez_compressed(radius_grid_file_name, r_x_y_swapped)
    if x_y_swap == 0 and x_flip == 0 and y_flip == 0:
        spacing = [128. / i for i in downsampled_occupancy.shape]
        # print_func(imageToVTK(occupancy_grid_file_name.replace('.npz', ''),
        #                       pointData={'occupancy_probability': o_x_y_swapped}, origin=(0, 0, 0), spacing=spacing))

    # my_print(meta_data)

    # imageToVTK(occupancy_grid_file_name, pointData={'occupancy_probability': o_x_y_swapped.copy()},
    #            origin=(0, 0, 0), spacing=[1,1,1])
    save_end = time.time()
    print_func("File save time: {0}".format(save_end - save_start))
    print_func()

    json.dump(meta_data, open(meta_data_file_name, 'w'), sort_keys=True, indent=4)
    # with open(meta_data_file_name, 'r') as fin:
    # my_print(fin.read())


def create_all_volumes(dataset_name, resolution, translation, bound_start, bound_end, fact, z_fact):
    print(dataset_name)
    z_rotation_count = 6
    z_rotations = [i * np.pi * 2 / z_rotation_count for i in range(0, int(z_rotation_count / 2))]
    radius_multipliers = [1., 0.34, np.sqrt(2.), 1. / np.sqrt(2)]
    ctr = 0
    os.makedirs("{0}/{1}/".format(manual_rec_dir, dataset_name), exist_ok=True)
    console_out_text_file = open("{0}/{1}/console.txt".format(manual_rec_dir, dataset_name), 'w+')
    double_stream = StreamTee(sys.stdout, console_out_text_file)

    if resolution[0] == resolution[1] and resolution[1] != resolution[2]:
        depth_axis = "z"
    elif resolution[0] == resolution[2] and resolution[1] != resolution[2]:  # Unlikely, but OK.
        depth_axis = "y"
    elif resolution[1] == resolution[2] and resolution[0] != resolution[1]:
        depth_axis = "x"
    else:
        raise ValueError("Unexpected resolution: {}".format(str(resolution)))


    for z_rotation in z_rotations:
        for radius_multiplier in radius_multipliers:
            voxellization_params = [dataset_name, [int(resolution[i] * 8) for i in range(3)], depth_axis,
                                    radius_multiplier, translation, bound_start, bound_end, fact, z_fact, z_rotation,
                                    ctr, double_stream]
            # params.append(voxellization_params)
            create_a_volume(voxellization_params)
            ctr += 1


class StreamTee(object):
    # Based on https://gist.github.com/327585 by Anand Kunal
    def __init__(self, stream1, stream2):
        self.stream1 = stream1
        self.stream2 = stream2
        self.__missing_method_name = None  # Hack!

    def __getattribute__(self, name):
        return object.__getattribute__(self, name)

    def __getattr__(self, name):
        self.__missing_method_name = name  # Could also be a property
        return getattr(self, '__methodmissing__')

    def __methodmissing__(self, *args, **kwargs):
        # Emit method call to the log copy
        callable2 = getattr(self.stream2, self.__missing_method_name)
        callable2(*args, **kwargs)

        # Emit method call to stdout (stream 1)
        callable1 = getattr(self.stream1, self.__missing_method_name)
        ret_val = callable1(*args, **kwargs)
        self.stream1.flush()
        self.stream2.flush()
        return ret_val

def flattening_image(img, num_slices_to_compress):
    dim_x, dim_y, dim_z = img.shape
    if dim_z%num_slices_to_compress != 0:
        img = np.pad(img, ((0,0),(0,0),(0,num_slices_to_compress-dim_z%num_slices_to_compress)), 'constant')

    print('dim_z/num_slices_to_compress:', dim_z/num_slices_to_compress)
    dim_x, dim_y, dim_z = img.shape
    compressed_img = img.reshape(dim_x, dim_y, int(dim_z/num_slices_to_compress), num_slices_to_compress)
    compressed_img = np.mean(compressed_img, axis=3)
    print('img.shape', img.shape)
    print('compressed_img.shape', compressed_img.shape)
    return compressed_img


def get_branch(branch,s, loc_spline, radii_spline):
    """
    :branch: np array of shape: n_segments * [x,y,z,r]
    """
    for i in range(1, branch.shape[0]):
        segment_length = np.linalg.norm(branch[i, :3] - branch[i - 1, :3])
        # print('segment_length:{}, s:{}, i:{}'.format(segment_length, s, i))

        if s < segment_length:
            segment_weight = (segment_length - s) / segment_length
            s_idx = i - segment_weight
            # if i != branch.shape[0]-1:  # not in the last segment
            x = loc_spline(s_idx)
            r = float(radii_spline(s_idx))
            # else:
            #     # in the last segment, simply do linear interpolation
            #     # print('doing linear interpolation in the last segment of the branch')
            #     x = segment_weight*branch[i-1,:3]+(1-segment_weight)*branch[i,:3]
            #     r = segment_weight*branch[i-1,3]+(1-segment_weight)*branch[i,3]

            # print('loc_spline 0.5, 10.5, 20.5')
            # print(loc_spline(0.5))
            # print(loc_spline(10.5))
            # print(loc_spline(20.5))
            # print('distance between adjacent segments:', loc_spline(i)-loc_spline(i-1))

            return x, r, s_idx
        s -= segment_length
    return branch[-1, :3], 0, branch.shape[0]-1


def generate_branch(branch, loc_spline, radii_spline):
    def _branch(s):
        return get_branch(branch, s, loc_spline, radii_spline)

    return _branch


def truncated_gauss_blob(x, r, t=3):
    """
    generates truncated gaussian blob at coordinates x with radius r
    Because the 'radius' of gaussian bell is around twice of the sigma,
    so when making gaussian blobs, need to use root_rad/2 as sigma!!
    :x: 3d float vector [x,y,z] (unit: voxel coordinates)
    :r: float radius (unit: voxels)
    :t: int truncation boundaries (unit: radius), default: 3
    :return: x_range,y_range,z_range,truncated_blob
    """
    x_floor , y_floor, z_floor, r_ceil = int(round(x[0])), int(round(x[1])), int(round(x[2])), int(round(r))

    # x_floor , y_floor, z_floor, r_ceil = int(x[0]), int(x[1]), int(x[2]), int(r) + 1
    x_lin = np.linspace(x_floor - int(t * r_ceil * x_stretch[0]), x_floor + int(t * r_ceil * x_stretch[0]) - 1, 2 * int(t * r_ceil * x_stretch[0]))
    y_lin = np.linspace(y_floor - int(t * r_ceil * x_stretch[1]), y_floor + int(t * r_ceil * x_stretch[1]) - 1, 2 * int(t * r_ceil * x_stretch[1]))
    z_lin = np.linspace(z_floor - int(t * r_ceil * x_stretch[2]), z_floor + int(t * r_ceil * x_stretch[2]) - 1, 2 * int(t * r_ceil * x_stretch[2]))
    y_grid, x_grid, z_grid = np.meshgrid(y_lin, x_lin, z_lin)  # numpy meshgrid swaps x and y
    blob = np.exp(-(((x_grid - x[0]) * 2 / x_stretch[0]) ** 2 + ((y_grid - x[1]) * 2 / x_stretch[1]) ** 2 + (
                (z_grid - x[2]) * 2 / x_stretch[2]) ** 2) / r ** 2)
    x_range = [x_floor - int(t * r_ceil * x_stretch[0]), x_floor + int(t * r_ceil * x_stretch[0])]
    y_range = [y_floor - int(t * r_ceil * x_stretch[1]), y_floor + int(t * r_ceil * x_stretch[1])]
    z_range = [z_floor - int(t * r_ceil * x_stretch[2]), z_floor + int(t * r_ceil * x_stretch[2])]
    return x_range, y_range, z_range, blob


if __name__ == '__main__':
    from math import sqrt
    import xml_structure_interpreter as x


    # dataset_name = 'Lupine_22august'  #'lupine_small_xml
    #
    # # xml_path = '/home/stud/zhaoy/root_mri/experiments/voxellization_experiments/Lupine_22august/lupin22august.xml'
    # # x.interpret_xml(dataset_name=dataset_name, xml_path=xml_path, mults=(1,1,1))
    #
    # target_dim_depth = 337
    # depth_fact = target_dim_depth/120
    #
    # resolution = [target_dim_depth, 256, 256]
    # translation = [63.8951*depth_fact, 49.9, 49.738153381347656]
    # bound_start = [-0.541667*depth_fact, -0.1953125, -0.1953125]
    # bound_end = [129.4582935*depth_fact, 99.8046875, 99.8046875]
    # fact = 129
    # # radius_multiplier = [1.00, 0.34, round(1/sqrt(2), ndigits=2), round(sqrt(2), ndigits=2)]
    # radius_multiplier = 1.00
    # depth_axis = 'x'
    # z_rotation = 0
    # ctr = 0
    # os.makedirs("{0}/{1}/".format(manual_rec_dir, dataset_name), exist_ok=True)
    # console_out_text_file = open("{0}/{1}/console.txt".format(manual_rec_dir, dataset_name), 'w+')
    # double_stream = StreamTee(sys.stdout, console_out_text_file)
    #
    # voxellization_params = [dataset_name, [int(resolution[i] * 8) for i in range(3)], depth_axis,
    #                         radius_multiplier, translation, bound_start, bound_end, fact, depth_fact, z_rotation,
    #                         ctr, double_stream]
    # create_a_volume(voxellization_params)



    # dataset_name = 'I_Sand_3D'  # 'lupine_small_xml
    # data_type = 'root_sys'
    #
    # # xml_path = '/home/stud/zhaoy/root_mri/experiments/voxellization_experiments/Lupine_22august/lupin22august.xml'
    # # x.interpret_xml(dataset_name=dataset_name, xml_path=xml_path, mults=(1,1,1))
    #
    # target_dim_depth = 210
    # depth_fact = target_dim_depth / 70
    #
    # print('sys.argv[1:7]:', sys.argv[1:7])
    # trans_x, trans_y, trans_z, bound_end_x, bound_end_y, bound_end_z = sys.argv[1:7]
    # trans_x, trans_y, trans_z = float(trans_x), float(trans_y), float(trans_z)
    # bound_end_x, bound_end_y, bound_end_z = float(bound_end_x), float(bound_end_y), float(bound_end_z)
    #
    # resolution = [256, 256, target_dim_depth]
    # translation = [trans_x, trans_y, trans_z*depth_fact]  #
    # # translation = [27, 41, 55* depth_fact]
    # bound_start = [0, 0, 0* depth_fact]
    # bound_end = [bound_end_x, bound_end_y, bound_end_z*depth_fact]  # [71.287, 72.68, 68.172*depth_fact] #[73, 70, 68*depth_fact]  #[60, 80, 60* depth_fact]
    # fact = 1
    # # radius_multiplier = [1.00, 0.34, round(1/sqrt(2), ndigits=2), round(sqrt(2), ndigits=2)]
    # radius_multiplier = 1.00
    # depth_axis = 'z'
    # z_rotation = 0
    # ctr = 0
    # os.makedirs("{0}/{1}/".format(manual_rec_dir, dataset_name), exist_ok=True)
    # console_out_text_file = open("{0}/{1}/console.txt".format(manual_rec_dir, dataset_name), 'w+')
    # double_stream = StreamTee(sys.stdout, console_out_text_file)
    #
    # res_mult_fac = 8  # 8
    # voxellization_params = [dataset_name, [int(resolution[i] * res_mult_fac) for i in range(3)], depth_axis,
    #                         radius_multiplier, translation, bound_start, bound_end, fact, depth_fact, z_rotation,
    #                         ctr, double_stream, data_type]
    # create_a_volume(voxellization_params)


    start_time = time.time()

    # dataset_name = 'I_Sand_3D_DAP13'
    # data_type = 'root_sys'

    # xml_path = '/home/stud/zhaoy/root_mri/experiments/voxellization_experiments/Lupine_22august/lupin22august.xml'
    # x.interpret_xml(dataset_name=dataset_name, xml_path=xml_path, mults=(1,1,1))

    # target_dim_depth = 198
    alias_fact = 3  #3  # use this because the voxel is not cubic, but the z dimension has a longer length

    trans_x, trans_y, trans_z, bound_end_x, bound_end_y, bound_end_z, \
    dataset_name, dim_x, dim_y, dim_z, data_type, fact_x, fact_y, fact_z,  \
    depth_axis, manual_rec_dir, reverse_x, reverse_y, reverse_z, radius_multiplier = sys.argv[1:21]

    trans_x, trans_y, trans_z = float(trans_x), float(trans_y), float(trans_z)
    bound_end_x, bound_end_y, bound_end_z = float(bound_end_x), float(bound_end_y), float(bound_end_z)
    dim_x, dim_y, dim_z = int(dim_x), int(dim_y), int(dim_z)
    fact = np.array([float(fact_x), float(fact_y), float(fact_z)])
    radius_multiplier = float(radius_multiplier)
    # depth_fact = float(depth_fact) * alias_fact
    # convert strings to boolean values
    if reverse_x == 'True': reverse_x = True
    else: reverse_x = False
    if reverse_y == 'True': reverse_y = True
    else: reverse_y = False
    if reverse_z == 'True': reverse_z = True
    else: reverse_z = False

    print('trans_x:{}, trans_y:{}, trans_z:{}, bound_end_x:{}, bound_end_y:{}, bound_end_z:{}, '
          'dataset_name:{}, dim_x:{}, dim_y:{}, dim_z:{}, data_type:{}, '
          'fact_x:{}, fact_y:{}, fact_z:{}, depth_axis:{}, manual_rec_dir:{}, '
          'reverse_x:{}, reverse_y:{}, reverse_z:{}, radius_multiplier:{}\n'.format(trans_x, trans_y, trans_z,
                                                     bound_end_x, bound_end_y,
                                                     bound_end_z,
                                                     dataset_name, dim_x, dim_y, dim_z,
                                                     data_type,
                                                     fact_x, fact_y, fact_z,
                                                     depth_axis, manual_rec_dir,
                                                     reverse_x, reverse_y, reverse_z,
                                                     radius_multiplier))

    if depth_axis == 'z':
        resolution = [dim_x, dim_y, dim_z * alias_fact]
    elif depth_axis == 'x':
        resolution = [dim_x * alias_fact, dim_y, dim_z]
    elif depth_axis == 'y':
        resolution = [dim_x, dim_y * alias_fact, dim_z]

    translation = [trans_x, trans_y, trans_z]  # * depth_fact
    # translation = [27, 41, 55* depth_fact]
    bound_start = [0, 0, 0]  # * depth_fact
    bound_end = [bound_end_x, bound_end_y,
                 bound_end_z]  # * depth_fact
    # [71.287, 72.68, 68.172*depth_fact] #[73, 70, 68*depth_fact]  #[60, 80, 60* depth_fact]
    # fact = 1
    # radius_multiplier = [1.00, 0.34, round(1/sqrt(2), ndigits=2), round(sqrt(2), ndigits=2)]
    z_rotation = 0  #np.deg2rad(20)
    ctr = 0
    os.makedirs("{0}/{1}/".format(manual_rec_dir, dataset_name), exist_ok=True)
    console_out_text_file = open("{0}/{1}/console.txt".format(manual_rec_dir, dataset_name), 'w+')
    double_stream = StreamTee(sys.stdout, console_out_text_file)

    res_mult_fac = 2  # 8, resolution scaling factor
    fact *= res_mult_fac
    voxellization_params = [dataset_name, [int(resolution[i] * res_mult_fac) for i in range(3)], depth_axis,
                            radius_multiplier, translation, bound_start, bound_end, fact, alias_fact, z_rotation,  # depth_fact used to be set to 1 (when not using alias_fact)
                            ctr, double_stream, data_type, reverse_x, reverse_y, reverse_z]
    print('voxellization_params:', voxellization_params)
    create_a_volume(voxellization_params)
    print('Total time used:', time.time() - start_time)



    # image_path = '/home/stud/zhaoy/root_mri/experiments/voxellization_experiments/Lupine_22august/r_factor_1.00/rot_0/x_flip_0/y_flip_0/x_y_swap_0/256x256x360_occupancy.npz'
    # output_dir = '/home/stud/zhaoy/root_mri/experiments/voxellization_experiments/Lupine_22august/r_factor_1.00/rot_0/x_flip_0/y_flip_0/x_y_swap_0'
    # image = np.load(image_path)['arr_0']
    # # a = np.ones((5,6,7))
    # # for i in range(a.shape[2]):
    # #     a[:,:,i] *= i
    # nstc = 3
    # flattened = flattening_image(image, nstc)
    # # print("a:", a)
    # # print("flattened:", flattened)
    # np.savez_compressed(join(output_dir, '256x256x360_occupancy_compressed3.npz'), flattened)



    # occupancy_grid = np.ones((2048, 2048, 337*10), dtype=np.ubyte)
    # factor = 8
    # downsampled_occupancy_grid = divide_and_reduce(occupancy_grid, factor, "mean")
    # print(downsampled_occupancy_grid.shape)







