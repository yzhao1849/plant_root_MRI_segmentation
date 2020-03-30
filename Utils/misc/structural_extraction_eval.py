import numpy as np
import sys
import math
import argparse
from copy import deepcopy
from time import time


def point_distance_Euclidean(p1, p2):
    x1, y1, z1 = p1
    x2, y2, z2 = p2
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)


def dfs_tree(node, point_list, parent_node, min_dist=None):
    x, y, z = node.getCoor()

    if parent_node is None:  # this is the root node
        # print('Current node is root')
        assert len(node.getChildren()) == 1  # assert the root node only have 1 child
        child = node.getChildren()[0]
        c_x, c_y, c_z = child.getCoor()
        direction_x, direction_y, direction_z = (
        c_x - x, c_y - y, c_z - z)  # the same direction as the segment that starts with the root
        point_list.append([len(point_list), x, y, z, direction_x, direction_y, direction_z])

    else:
        # print('Current node is not root')
        # add the intermediate points between the parent and current node
        p_x, p_y, p_z = parent_node.getCoor()
        direction_x, direction_y, direction_z = (x - p_x, y - p_y, z - p_z)
        seg_length = point_distance_Euclidean(node.getCoor(), parent_node.getCoor())
        #         print('length of current segment:', seg_length)
        if min_dist is not None:
            if seg_length > min_dist:  # then this segment should contain some intermediate points
                if seg_length % min_dist == 0:
                    num_inter_points = seg_length // min_dist - 1
                else:
                    num_inter_points = seg_length // min_dist
                add_vector_unit = np.array([x - p_x, y - p_y, z - p_z]) * min_dist / seg_length
                for i in range(int(num_inter_points)):
                    x_i, y_i, z_i = p_x + (i + 1) * add_vector_unit[0], p_y + (i + 1) * add_vector_unit[1], p_z + (
                    i + 1) * add_vector_unit[2]
                    point_list.append([len(point_list), x_i, y_i, z_i, direction_x, direction_y, direction_z])
        # add the current node
        point_list.append([len(point_list), x, y, z, direction_x, direction_y, direction_z])

    # recursively add the points in the child segments
    child_list = node.getChildren()
    # print('Number of children of current node:', len(child_list))
    for child in child_list:
        dfs_tree(child, point_list, node, min_dist)

    return point_list


def points_distance(point1, point2):
    x1, y1, z1 = point1
    x2, y2, z2 = point2
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2 + (
    z1 - z2) ** 2) ** 0.5  # do not calculate square root when unnecessary! May be too slow


def calculate_TP_FP_FN(point_list_EXT, point_list_GT, distance_thres):
    used_p_E_indices = np.zeros(len(point_list_EXT)).astype(bool)  # all False in the beginning
    num_TP = 0

    for p_G in point_list_GT:
        closest_p_E = None  # to store the closest point in EXT to p_G
        id_G, x_G, y_G, z_G, dir_x_G, dir_y_G, dir_z_G = p_G
        neighborhood_EXT = point_list_EXT[
            (point_list_EXT[:, 1] >= x_G - distance_thres) * (point_list_EXT[:, 1] <= x_G + distance_thres) *
            (point_list_EXT[:, 2] >= y_G - distance_thres) * (point_list_EXT[:, 2] <= y_G + distance_thres) *
            (point_list_EXT[:, 3] >= z_G - distance_thres) * (point_list_EXT[:, 3] <= z_G + distance_thres) *
            (1 - used_p_E_indices).astype(bool)]
        # print('p_G:', p_G)
        # print('neighborhood_EXT:', neighborhood_EXT)
        for p_E in neighborhood_EXT:
            id_E, x_E, y_E, z_E, dir_x_E, dir_y_E, dir_z_E = p_E
            E_G_distance = points_distance((x_G, y_G, z_G), (x_E, y_E, z_E))
            if E_G_distance <= distance_thres:  # if the distance is within threshold
                if np.dot((dir_x_G, dir_y_G, dir_z_G),
                          (dir_x_E, dir_y_E, dir_z_E)) >= 0:  # directions are not so different
                    if (closest_p_E is None) or (E_G_distance < points_distance((x_G, y_G, z_G), closest_p_E[1:4])):
                        closest_p_E = p_E
        # print('closest_p_E:', closest_p_E)
        if closest_p_E is not None:
            used_p_E_indices[closest_p_E[0]] = True  # each point in EXT can correspond to at most 1 point in GT
            num_TP += 1

    num_FP = point_list_EXT.shape[0] - used_p_E_indices.astype(int).sum()
    num_FN = point_list_GT.shape[0] - used_p_E_indices.astype(int).sum()

    # print('TP, FP, FN:', num_TP, num_FP, num_FN)
    # print('used_p_E_indices:', used_p_E_indices)
    return num_TP, num_FP, num_FN


def calculate_precision(TP, FP):
    precision = TP / (TP + FP)
    print('Precision:', precision)
    return precision


def calculate_recall(TP, FN):
    recall = TP / (TP + FN)
    print('Recall:', recall)
    return recall


def calculate_F1(TP, FP, FN):
    precision = calculate_precision(TP, FP)
    recall = calculate_recall(TP, FN)
    f1 = 2 * precision * recall / (precision + recall)
    print('F1 score:', f1)
    return f1


def evaluate_extraction(point_list_EXT, point_list_GT, distance_thres):
    point_list_EXT = np.array(point_list_EXT)
    point_list_GT = np.array(point_list_GT)
    TP, FP, FN = calculate_TP_FP_FN(point_list_EXT, point_list_GT, distance_thres)
    print('TP, FP, FN:', TP, FP, FN)
    return calculate_F1(TP, FP, FN)


def parse_args():
    parser = argparse.ArgumentParser(description='structural extraction evaluation')
    parser.add_argument('-re', '--root_xml_extracted', type=str, required=True,
                        help='Path of the xml file of the graph of root1.')
    parser.add_argument('-rg', '--root_xml_gt', type=str, required=True,
                        help='Path of the xml file of the graph of root2.')
    parser.add_argument('-rgd', '--root_graph_dir', type=str, default='/home/user/zhaoy/git/oguz_plantRoot_repo/'
                                                                      'plant-root-MRI-display/root_extraction/'
                                                                      'RootStructureExtractor/PyUtils',
                        help='Directory containing the root_graph module.')
    parser.add_argument('-s', '--spacing', type=float, required=True,
                        help='the min spacing between adjacent anchor points on the root graph, unit: pixel side')
    parser.add_argument('-dt', '--distance_threshold', type=float, required=True,
                        help='the threshold of distance between 2 graphs. Only when the distance between '
                             '2 points (one from each graph) is within this threshold, they can be regarded '
                             'as corresponding to each other (when the directions of them are also not too different)')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    root_xml_extracted = args.root_xml_extracted
    root_xml_gt = args.root_xml_gt
    root_graph_dir = args.root_graph_dir
    spacing = args.spacing
    distance_threshold = args.distance_threshold

    sys.path.append(root_graph_dir)
    import root_graph

    root_extracted = root_graph.RootGraph(root_xml_extracted)
    root_gt = root_graph.RootGraph(root_xml_gt)

    root_point_list_extracted = dfs_tree(root_extracted.getRoot(), [], None, min_dist=spacing)
    root_point_list_extracted = np.array(root_point_list_extracted).astype(int)
    # print(root_point_list_extracted.astype(int).tolist()[:20])

    root_point_list_gt = dfs_tree(root_gt.getRoot(), [], None, min_dist=spacing)
    root_point_list_gt = np.array(root_point_list_gt).astype(int)
    # print(root_point_list_gt.astype(int).tolist()[:20])

    evaluate_extraction(root_point_list_extracted, root_point_list_gt, distance_thres=distance_threshold)
