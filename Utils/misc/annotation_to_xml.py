import sys
import os
import pickle
from os.path import expanduser, join
import time
from scipy.interpolate import CubicSpline
import numpy as np
from lxml import etree
import math


home = expanduser("~")


def annotation_to_xml(params):
    volumes_creation_start = time.time()
    [dataset_name, dim, depth_axis, radius_multiplier, [x_translate, y_translate, z_translate], [x_min, y_min, z_min],
     [x_max, y_max, z_max], fact, z_rotation, num, double_stream, data_type,
     reverse_x, reverse_y, reverse_z] = params
    # fact is used to rescale the annotation coordinates to the mri coordinates
    fact_x, fact_y, fact_z = fact
    print('**********fact:{}**********'.format(fact))

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
    # if depth_axis == 'x':
    #     segment_records[:, 1] *= z_fact
    # elif depth_axis == 'y':
    #     segment_records[:, 2] *= z_fact
    # else:
    #     segment_records[:, 3] *= z_fact

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

    print('$$$$$$$$$ shape of segment_records:', segment_records.shape)


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

    branch_tip_ids = tip_records[:, 4]
    ctr = 1
    print()
    branch_tip_ctr = 0
    progress_format = "\r{}: {:xd}/{:xd}".replace("x", str((int(np.ceil(np.log10(len(branch_tip_ids)))))))
    set_visited_seg = set()  # store the already voxelized segments, to avoid repeated voxelization

    # assert depth_fact == 3
    # assert res_mult_fac == 2

    # get the seg_ids of nodes that have more than one child (branching nodes)
    # --> equivalent to finding the ids that occur more than once in the prev_id column
    # the prev_id is the fifth (index 4) column of segment_records

    # print('segment_records:')
    # for i in range(segment_records.shape[0]):
    #     print(segment_records[i, :])

    branching_seg_ids = set()
    record_found_ids = np.zeros(int(segment_records[:, 4].max() + 1))

    for i in range(segment_records.shape[0]):
        current_seg_id = segment_records[i, 4].astype(int)
        if record_found_ids[current_seg_id] != 1:  # not recorded yet
            record_found_ids[current_seg_id] = 1
        else:
            branching_seg_ids.add(float(current_seg_id))  # if already recorded, then this seg is a branching node
    print("branching_seg_ids:", branching_seg_ids)

    dict_id_node = {}  # the dict to store the seg_id -- tree_node pairs, the seg_ids are of the branching nodes
    root_node = None  # the root lxml tree element
    tree_branch_id = 0
    dict_branchStartNode_parentID = {}  # the dict for recording the branch start tree node -- parent seg_id pairs
    for branch_tip_id in branch_tip_ids:
        print('tree_branch_id:', tree_branch_id)
        print('branch_tip_ctr:', branch_tip_ctr)
        branch_tip_ctr = branch_tip_ctr + 1
        # if branch_tip_ctr>1:
        #     continue
        print_func(progress_format.format(int(num), int(branch_tip_ctr), int(len(branch_tip_ids))), end="")
        # double_stream.write()
        # double_stream.flush()
        branch_points = list()
        branch_radii = list()
        branch_ids = list()

        current_id = branch_tip_id
        first_seg = get_segment_by_id(current_id)

        # parent_tree_node = None
        parent_tree_node_id = None
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

                # the position of each id should correspond correctly to the coordinates and the radius
                branch_ids.append(-1)
                branch_ids.append(prev_seg_id)

                set_visited_seg.add(current_id)
                if prev_seg_id in set_visited_seg:  # once reach a segment that has already been visited, stop
                    if tree_branch_id != 0:  # not the first (main) branch
                        # parent_tree_node = dict_id_node[prev_seg_id]
                        parent_tree_node_id = prev_seg_id
                    break

                current_id = prev_seg_id

        else:
            while current_id != -1.:  # -1 means prev is tree, we reached root

                seg = get_segment_by_id(current_id)
                prev_seg_id = seg[4]

                radius = seg[5] * radius_multiplier

                loc = seg[1:4]

                branch_points.append(loc)
                branch_radii.append(radius)
                branch_ids.append(current_id)

                set_visited_seg.add(current_id)
                if prev_seg_id in set_visited_seg:
                    if tree_branch_id != 0:  # not the first (main) branch
                        # xml is different from RootSys, we need to explicitly add the parent seg of the branch
                        # (because when adding to the tree, the last seg in the branch_points is not added!)
                        # so if it's actually the last point of the branch (not the parent node),
                        # then this point will be missing
                        prev_seg = get_segment_by_id(prev_seg_id)
                        prev_radius = prev_seg[5] * radius_multiplier
                        prev_loc = prev_seg[1:4]
                        branch_points.append(prev_loc)
                        branch_radii.append(prev_radius)
                        branch_ids.append(prev_seg_id)

                        parent_tree_node_id = prev_seg_id
                        print('parent_tree_node_id:', parent_tree_node_id)
                    break

                current_id = prev_seg_id
        # order is from tip to shoot
        branch_radii = [0] + branch_radii
        branch_radii = branch_radii[:-1]

        if len(branch_radii)==1 and branch_radii[-1]==0:  # normally does not occur
            continue

        print("*******dim*********", dim)

        branch_points = np.array(branch_points)
        branch_radii = np.array(branch_radii)
        if reverse_x:
            branch_points[:, 0] = dim[0] - branch_points[:, 0]
        if reverse_y:
            branch_points[:, 1] = dim[1] - branch_points[:, 1]
        if reverse_z:
            branch_points[:, 2] = dim[2] - branch_points[:, 2]


        min_dist = 2  # todo: manually set the min_dist and depth_dim
        depth_dim = 2
        # generate an etree for this branch
        if root_node is None:
            root_node = add_branch_to_xml_tree(branch_points, branch_radii, min_dist,
                                               parent_tree_node_id, branching_seg_ids,
                                               dict_id_node, branch_ids, tree_branch_id,
                                               dict_branchStartNode_parentID, depth_dim=depth_dim)
            forest = etree.Element('Forest')
            forest.append(root_node)
        else:
            add_branch_to_xml_tree(branch_points, branch_radii, min_dist,
                                   parent_tree_node_id, branching_seg_ids,
                                   dict_id_node, branch_ids, tree_branch_id,
                                   dict_branchStartNode_parentID, depth_dim=depth_dim)

        tree_branch_id += 1  # increase the id before going to the next branch

    # connect the tree nodes in branches to the parent branch
    for tree_node in dict_branchStartNode_parentID:
        parent_id = dict_branchStartNode_parentID[tree_node]
        parent_node = dict_id_node[parent_id]
        parent_node.append(tree_node)

    # in the end, write the xml file
    out_dir = '{0}/r_factor_{1:.2f}/rot_{2}/x_flip_0/y_flip_0/x_y_swap_0'.format(
        inputdir,
        radius_multiplier,
        int(round(z_rotation * 180. / np.pi))
    )
    os.makedirs(out_dir, exist_ok=True)

    # print(etree.tostring(forest, pretty_print=True))
    with open(join(out_dir, 'root_gt.xml'), 'wb') as f:
        f.write(etree.tostring(forest, pretty_print=True))
    print('root graph xml saved at {}'.format(out_dir))

    xml_generation_end = time.time()
    print_func("Data {}:{} creation took {} seconds.".format(dataset_name, num,
                                                             xml_generation_end - volumes_creation_start))


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


def euclidean_distance(point1, point2):
    x1, y1, z1 = point1
    x2, y2, z2 = point2
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)


def interpolate_intermediate_point_indices(point1, index1, point2, index2, min_dist):
    # direction: from point1 to point2
    intermediate_point_indices = []
    intermediate_point_indices.append(index1)
    seg_length = euclidean_distance(point1, point2)
    if seg_length > min_dist:
        if seg_length % min_dist == 0:
            num_in_points = seg_length // min_dist - 1
        else:
            num_in_points = seg_length // min_dist
        index_add_unit = (index2 - index1) * min_dist / seg_length
        for i in range(int(num_in_points)):
            intermediate_point_indices.append(index1 + index_add_unit * (i + 1))

    return intermediate_point_indices  # includes index1 but not index2!


def rearrange_coordinates(x, y, z, depth_dim):
    if depth_dim == 0:
        x_temp, y_temp, z_temp = z, y, x
    elif depth_dim == 1:
        x_temp, y_temp, z_temp = x, z, y
    elif depth_dim == 2:
        x_temp, y_temp, z_temp = y, x, z
    return x_temp, y_temp, z_temp


def add_branch_to_xml_tree(branch_points, branch_radii, min_dist, parent_tree_node_id,
                           branching_seg_ids, dict_id_node, branch_ids, tree_branch_id,
                           dict_branchStartNode_parentID, depth_dim=0):
    """
    depth_dim is where the user want to put the depth dimension coordinate in the tree Node,
    if depth_dim is 0, then put the depth coordinate at the first position, like (z, x, y)
    parent_tree_node is the parent node of the whole branch
    """
    # print('branch_ids:', branch_ids)
    # print('branching_seg_ids:', branching_seg_ids)
    # print('dict_id_node:', dict_id_node)
    print('parent_tree_node_id:', parent_tree_node_id)
    print('branch_ids:', branch_ids)

    radii_spline = CubicSpline([el for el in list(range(len(branch_points)))],
                               branch_radii)  # why not used?? Using linear interpolation instead
    # radii_spline_d = radii_spline.derivative(1)
    loc_spline = CubicSpline([el for el in list(range(len(branch_points)))], branch_points)

    # keep track of the current tree node
    current_tree_node = None

    print('Total points in this branch:', len(branch_points))
    for i in range(len(branch_points) - 1):

        intermediate_point_indices = interpolate_intermediate_point_indices(branch_points[i], i,
                                                                            branch_points[i + 1], i + 1, min_dist)

        current_id = branch_ids[i]  # the seg_id of the current start point
        # print('current_id:', current_id)

        seg_start = True
        for ind in intermediate_point_indices:
            inter_point_x, inter_point_y, inter_point_z = loc_spline(ind)
            inter_point_rad = radii_spline(ind)

            x_temp, y_temp, z_temp = rearrange_coordinates(inter_point_x, inter_point_y,
                                                           inter_point_z, depth_dim)
            node_next = etree.Element("Node", bo="0", id="{:d}".format(int(tree_branch_id)),
                                      rad="{:.1f}".format(inter_point_rad),
                                      x="{:d}".format(int(x_temp)), y="{:d}".format(int(y_temp)),
                                      z="{:d}".format(int(z_temp)))

            if seg_start:  # ignore the intermediate points, only look at the start point of the segment
                if current_id in branching_seg_ids:
                    # if the current node is a branching node, record it in the dict for further use
                    dict_id_node[current_id] = node_next
            seg_start = False

            if current_tree_node is None:
                current_tree_node = node_next
            else:
                # add the current_tree_node as the child of node_next:
                node_next.append(current_tree_node)
                current_tree_node = node_next

    # print('dict_id_node after:', dict_id_node)

    if parent_tree_node_id is None:
        # then this branch is the main branch starting from the shoot
        # add the shoot node
        shoot_x, shoot_y, shoot_z = branch_points[-1]
        shoot_rad = branch_radii[-1]

        x_temp, y_temp, z_temp = rearrange_coordinates(shoot_x, shoot_y, shoot_z, depth_dim)
        shoot_node = etree.Element("Node", bo="0", id="{:d}".format(int(tree_branch_id)),
                                   rad="{:.1f}".format(shoot_rad),
                                   x="{:d}".format(int(x_temp)), y="{:d}".format(int(y_temp)),
                                   z="{:d}".format(int(z_temp)))
        shoot_node.append(current_tree_node)
        return shoot_node
    else:
        # parent_tree_node.append(current_tree_node)
        dict_branchStartNode_parentID[current_tree_node] = parent_tree_node_id
        return


if __name__ == '__main__':
    from math import sqrt

    start_time = time.time()

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

    resolution = [dim_x, dim_y, dim_z]

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
    params = [dataset_name, [int(resolution[i] * res_mult_fac) for i in range(3)], depth_axis,
                            radius_multiplier, translation, bound_start, bound_end, fact, z_rotation,  # depth_fact used to be set to 1 (when not using alias_fact)
                            ctr, double_stream, data_type, reverse_x, reverse_y, reverse_z]
    print('params:', params)
    annotation_to_xml(params)
    print('Total time used:', time.time() - start_time)









