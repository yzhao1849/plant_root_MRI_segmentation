import numpy as np
from matplotlib import pyplot as plt
import math


def plot_one_slice(array_3d, dim, idx):
    plt.figure(dpi=120)
    if dim=='x':  # shape of array_3d should be z,x,y
        plt.imshow(array_3d[:,idx,:], cmap='Greys')
    elif dim=='y':
        plt.imshow(array_3d[:,:,idx], cmap='Greys')
    elif dim=='z':
#         array_3d[array_3d<118]=0
        plt.imshow(array_3d[idx,:,:], cmap='Greys')
    plt.colorbar()
    plt.show()

def plot_merged_slices(array_3d, dim, idx0, idx1, title=None):
    plt.figure(dpi=200)
    if dim=='x':  # shape of array_3d should be z,x,y
        to_show=np.max(array_3d[:,idx0:idx1,:], axis=1)
        plt.imshow(to_show, cmap='Greys')  # aspect=2
    elif dim=='y':
        plt.imshow(np.max(array_3d[:,:,idx0:idx1], axis=2), cmap='Greys')
    elif dim=='z':
#         array_3d[array_3d<118]=0
        plt.imshow(np.max(array_3d[idx0:idx1,:,:], axis=0), cmap='Greys')
    plt.colorbar()
    
    if title is not None:
        plt.title(title)
    plt.show()
    
    
def load_npz_to_array(filename, key='arr_0'):
    return np.load(filename)[key]
        
    
def normalize_to(arr, min_value, max_value):
    arr = arr.astype(np.float32)
    arr -= arr.min()
    arr /= (arr.max() - arr.min())
    arr *= (max_value - min_value)
    arr += min_value
    return arr


def shuffle_data(data):
    """
    Shuffles data along first axis; somehow, numpy cannot shuffle compelx data.
    :return: sorted data
    """
    indices = list(range(data.shape[0]))
    np.random.shuffle(indices)
    return data[indices]


def randomly_divide_list(l, ratio):
    shuffled_l = shuffle_data(np.array(l))
    len1 = int(shuffled_l.shape[0] * ratio)
    shuffled_l1 = shuffled_l[:len1]
    shuffled_l2 = shuffled_l[len1:]
    return shuffled_l1, shuffled_l2


def get_min_max(file_path):
    '''get min and max coordinates in rootsys file'''
    min_x = math.inf
    min_y = math.inf
    min_z = math.inf

    max_x = -math.inf
    max_y = -math.inf
    max_z = -math.inf

    counter=0
    in_seg_range=False
    with open(file_path, 'r') as f:
        for l in f.readlines():
    #         print(counter)
            if l.startswith('segID#'):
                in_seg_range=True
                counter+=1
            if len(l)<=1:
                in_seg_range=False


            if in_seg_range==True:
                if counter%2==1 and counter!=1:  # every other line
                    x = float(l.split()[1])
                    y = float(l.split()[2])
                    z = float(l.split()[3])

                    min_x = min(min_x, x)
                    min_y = min(min_y, y)
                    min_z = min(min_z, z)

                    max_x = max(max_x, x)
                    max_y = max(max_y, y)
                    max_z = max(max_z, z)
                counter+=1
                
        print('min_x:', min_x, 'max_x:', max_x)
        print('min_y:', min_y, 'max_y:', max_y)
        print('min_z:', min_z, 'max_z:', max_z)

#         print('max_x:', max_x)
#         print('max_y:', max_y)
#         print('max_z:', max_z)
        return min_x, min_y, min_z, max_x, max_y, max_z