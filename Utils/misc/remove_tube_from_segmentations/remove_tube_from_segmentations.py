# Write the function to remove the test tube from segmentation outputs (considering the resolution factor too)
# input: test tube info xml, directory of segmentation output
# output: a sub dir in the directory of segmentation output

import os
from os.path import join, exists, isdir
from shutil import copyfile
import argparse

import numpy as np
from lxml import etree


def remove_tube_from_segmentation(seg_out_dir, tube_info_xml_path, without_tube_xml_path, super_res=2):
    '''
    remove the test tube from segmentation output, and save
    '''
    
    print('\nseg_out_dir:', seg_out_dir)
    print('tube_info_xml_path:', tube_info_xml_path)
    print('without_tube_xml_path:', without_tube_xml_path)
    print('super_res:', super_res)
    
    root = etree.parse(tube_info_xml_path)
    without_tube_root = etree.parse(without_tube_xml_path)
    
    output_dir = join(seg_out_dir, 'tube_removed')
    print('\noutput_dir:', output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    for fn in os.listdir(seg_out_dir):
        print()
        if fn.startswith('visualized_out_'):
            print()
            data_name = fn.split('.')[0]
            if data_name.endswith('_uint8'):
                data_name = '_'.join(data_name.split('_')[2:-2])
            elif data_name.endswith('_mri'):
                data_name = '_'.join(data_name.split('_')[2:-1])
            print('data_name:', data_name)
            
            if exists(join(output_dir, fn)):  # if the output with removed test tube already exists, skip it
                print('this segmentation output with test tube removed is already in the output directory, skipping it...')
            else:  # if the output with removed test tube does not exist yet, create it                 
                is_without_tube = without_tube_root.find(data_name)
                if is_without_tube is not None:  # this data has no test tube to be removed
                    print('this mri image has no test tube, directly copying to the output directory...')
                    copyfile(join(seg_out_dir, fn), join(output_dir, fn))
                else:
                    test_tube_info = root.find(data_name)
                    # print('test_tube_info:', test_tube_info)
                    if test_tube_info is None:
                        print('test_tube_info not found for this data, skipping it...')
                    else:
                        center_bottom = (test_tube_info.get('center_bottom_x'), 
                                         test_tube_info.get('center_bottom_y'), 
                                         test_tube_info.get('center_bottom_z'))
                        center_bottom = str_tuple_to_number_tuple(center_bottom, int)
                        center_top = (test_tube_info.get('center_top_x'), 
                                      test_tube_info.get('center_top_y'),
                                      test_tube_info.get('center_top_z'))
                        center_top = str_tuple_to_number_tuple(center_top, int)
                        radius_bottom = float(test_tube_info.get('radius_bottom'))
                        radius_top = float(test_tube_info.get('radius_top'))

                        # use super_res to rescale the values
                        if super_res != 2:
                            scale_factor = super_res / 2
                            center_bottom = tuple((np.array(center_bottom) * scale_factor).astype(int))
                            center_top = tuple((np.array(center_top) * scale_factor).astype(int))
                            radius_bottom *= scale_factor
                            radius_top *= scale_factor

                        print('center_bottom:', center_bottom)
                        print('center_top:', center_top)
                        print('radius_bottom:', radius_bottom)
                        print('radius_top:', radius_top)

                        # load the segmentation output
                        seg_result = np.load(join(seg_out_dir, fn))['arr_0']

                        # generate the mask for the test tube, use it to remove the tube and save
                        mask = generate_cylinder_mask(seg_result, center_bottom, radius_bottom, center_top, radius_top).astype(int)
                        seg_result[mask.astype(bool)] = 0    
                        print('removing test tube and saving to the output directory...')
                        np.savez_compressed(join(output_dir, fn), seg_result)
                        

def parse_args():
    parser = argparse.ArgumentParser(description='remove test tubes')

    parser.add_argument('-sod', '--seg_out_dir', type=str, required=True,
                        help='the absolute directory of the segmentation outputs')
    parser.add_argument('-ti', '--tube_info_xml_path', type=str, required=True,
                        help='the absolute path of the xml file with the tube position information')
    parser.add_argument('-wt', '--without_tube_xml_path', type=str, required=True,
                        help='the absolute path of the xml file with the list of data names which has no test tube')
    
    args = parser.parse_args()

    return args


def str_tuple_to_number_tuple(st, dtype):
    l = []
    for s in st:
        if dtype==int:
            l.append(int(s))
        elif dtype==float:
            l.append(float(s))
    return tuple(l)


def generate_cylinder_mask(original_img, circle_center0, radius0, circle_center1, radius1):
    """ 
    generate the cylinder mask of the test tube given the coordinates of the bottom and top circle 
    coordinates and radii.
    x is the depth dimenstion 
    """
    x0, y0, z0 = circle_center0  # larger x 
    x1, y1, z1 = circle_center1  # smaller z
    assert x0 > x1
    # x0, y0, z0, r0 = 40, 20, 2, 10
    # x1, y1, z1, r1 = 50, 50, 90, 15

    max_x, max_y, max_z = original_img.shape  # (100, 100, 100)
    x = np.linspace(0, max_x, max_x)
    y = np.linspace(0, max_y, max_y)
    z = np.linspace(0, max_z, max_z)

    xv, yv, zv = np.meshgrid(x, y, z, indexing='ij')
    # print(xv)
    # print(yv)
    # print(zv)
    # img = (((xv-50)**2 + (yv-50)**2 + (zv-50)**2)**0.5 < 30).astype(int)

    ratio = (xv - x0)/(x1 - x0)
    _x = ratio * (x1 - x0) + x0
    _y = ratio * (y1 - y0) + y0
    _z = ratio * (z1 - z0) + z0
    _r = ratio * (radius1 - radius0) + radius0

    mask = ((zv - _z)**2 + (yv - _y)**2 <= _r ** 2)
    mask *= (xv >= x1) * (xv <= x0)  # truncate the cylinder

    return mask


def main():
    args = parse_args()
    seg_out_dir = args.seg_out_dir
    tube_info_xml_path = args.tube_info_xml_path
    without_tube_xml_path = args.without_tube_xml_path
    
    remove_tube_from_segmentation(seg_out_dir, tube_info_xml_path, without_tube_xml_path)
    

if __name__ == "__main__":
    main()
    
    
