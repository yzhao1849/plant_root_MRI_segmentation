import math
import sys
from os import path

import torch as t
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from Utils.constants import *


log_2 = math.log(2)


class Loss2D(nn.Module):

    def __init__(self, confidence_penalty=0, super_resolution_factor=1):
        super(Loss2D, self).__init__()
        self.conf_pen = confidence_penalty
        self.super_resolution_factor = super_resolution_factor
        print("Conf Penalty is {}".format(confidence_penalty))

    def entropy_safe(self, x):
        """

        :param x: x_i \in [0,1] for all i
        :return: entropy(x)-1
        """
        minus_x=1.-x
        x = -(t.mul(x, t.log(x+1e-14)/log_2)+t.mul(minus_x, t.log(minus_x+1e-14)/log_2))-1.
        return x

    def classification_statistics(self, prediction, ground_truth, dont_care_mask):

        positive_predictions = prediction > 0.5
        byte_ground_truth = ground_truth.byte()
        if dont_care_mask is not None:
            positive_predictions[dont_care_mask] = 0
            byte_ground_truth[dont_care_mask] = 0

        tps_list = positive_predictions * byte_ground_truth
        num_tps_list = tps_list.sum(-1).sum(-1).sum(-1)
        num_fps_list = positive_predictions.sum(-1).sum(-1).sum(-1) - num_tps_list
        num_fns_list = byte_ground_truth.sum(-1).sum(-1).sum(-1) - num_tps_list
        if dont_care_mask is None:
            num_tns_list = ((1 - positive_predictions.byte()) * (1 - byte_ground_truth)).sum(-1).sum(-1).sum(-1)
        else:
            num_tns_list = ((1 - positive_predictions.byte()) * (1 - byte_ground_truth)).sum(-1).sum(-1).sum(-1) \
                           - (dont_care_mask > 0).sum(-1).sum(-1).sum(-1)

        return num_tps_list, num_fps_list, num_fns_list, num_tns_list

    def iou_loss(self, probability, ground_truth, importance_weights, dont_care_mask):
        intersection_mat = probability * ground_truth
        union_mat = probability + ground_truth
        if dont_care_mask is not None:
            intersection_mat[dont_care_mask] = 0
            union_mat[dont_care_mask] = 0

        if importance_weights is None:
            approx_intersection = intersection_mat.sum()
            approx_union = union_mat.sum()-approx_intersection
        else:  # TODO: does this make sense? I think so
            approx_intersection = (intersection_mat/importance_weights).sum()
            approx_union = (union_mat/importance_weights).sum() - approx_intersection

        loss = 1. - approx_intersection/approx_union
        return loss

    def gradient_diff_loss(self, desired_output, net_output, device, gdl_weight=0.00001, gdl_power=1):
        # shape of desired_output and net_output is (batch_size, z, x, y)
        gt_move_x, gt_move_y, gt_move_z = self.move_one_in_dims(desired_output, device)
        netout_move_x, netout_move_y, netout_move_z = self.move_one_in_dims(net_output, device)

        # using original signed gradient  # use the absolute values for calculating the gt diff or output diff,
        # because the the correctness of the direction is already taken care of by the other loss component...
        gdl_loss_x = ((desired_output - gt_move_x).abs() - (net_output - netout_move_x).abs()).abs()
        gdl_loss_y = ((desired_output - gt_move_y).abs() - (net_output - netout_move_y).abs()).abs()
        gdl_loss_z = ((desired_output - gt_move_z).abs() - (net_output - netout_move_z).abs()).abs()

        if gdl_power == 1:
            gdl_loss = gdl_loss_x + gdl_loss_y + gdl_loss_z  # the same as follow, just to save time
        else:
            gdl_loss = t.pow(gdl_loss_x, gdl_power) + t.pow(gdl_loss_y, gdl_power) \
                   + t.pow(gdl_loss_z, gdl_power)

        # remove the padded information
        gdl_loss[:, -1, :, :] = 0
        gdl_loss[:, :, -1, :] = 0
        gdl_loss[:, :, :, -1] = 0
        return gdl_loss * gdl_weight

    def move_one_in_dims(self, _tensor, device):
        bs, dim_z, dim_x, dim_y = _tensor.size()

        move_y = _tensor[:, :, :, 1:]  # will the backpropagation work here??? yes
        move_x = _tensor[:, :, 1:, :]
        move_z = _tensor[:, 1:, :, :]
        # padding to make the sizes compatible to _tensor:
        move_y = t.cat((move_y, t.zeros(bs, dim_z, dim_x, 1).to(device)), dim=3)
        move_x = t.cat((move_x, t.zeros(bs, dim_z, 1, dim_y).to(device)), dim=2)
        move_z = t.cat((move_z, t.zeros(bs, 1, dim_x, dim_y).to(device)), dim=1)

        return move_x, move_y, move_z

    def forward(self, net_output, desired_output, before_sigmoid, root_weight, **kwargs):
        if 'importance_weights' in kwargs:
            importance_weights = kwargs['importance_weights']
            importance_weights.requires_grad = False
            importance_weights = importance_weights.view(-1, 1, 1, 1)
            # expand the dimensions of importance_weights to be compatible for multiplication
        else:
            importance_weights = None

        zero_mat = t.mul(before_sigmoid, 0.)

        max_x_0 = t.max(before_sigmoid, zero_mat)
        x_z = t.mul(before_sigmoid, desired_output)
        log_1_ = t.log(t.add(t.exp(-t.abs(before_sigmoid)), 1))
        unweighted_loss_matrix = t.add(max_x_0.sub(x_z), log_1_)

        if 'dont_care_mask' in kwargs:
            dont_care_mask = kwargs['dont_care_mask']
            unweighted_loss_matrix[dont_care_mask] = 0
        else:
            dont_care_mask = None

        if 'calculate_iou' in kwargs:  # only calculate when specified
            if kwargs['calculate_iou'] is True:
                iou_loss = self.iou_loss(net_output, desired_output, importance_weights, dont_care_mask)
        else:
            iou_loss = None

        soil_mat = 1. - desired_output
        root_mat = desired_output

        soil_pixel_weight, root_pixel_weight = calculate_root_soil_weights(desired_output, root_weight)  # todo: add dont_care_mask?
        soil_weight_mat = t.mul(soil_mat, soil_pixel_weight)
        root_weight_mat = t.mul(root_mat, root_pixel_weight)
        if 'root_thickness_weight_map' in kwargs:
            root_weight_mat = t.mul(root_weight_mat, kwargs['root_thickness_weight_map'])

        unweighted_regularizer_loss = 0 if self.conf_pen == 0 else - self.entropy_safe(net_output) * self.conf_pen

        unweighted_regularized_loss = unweighted_loss_matrix + unweighted_regularizer_loss

        if 'calculate_gradient_diff_loss' in kwargs:
            if kwargs['calculate_gradient_diff_loss'] is True:
                assert 'device' in kwargs
                device = kwargs['device']
                laplace_of_gaussian_kernel = generate_3D_LoG_kernel(1, 1, 0.333)
                laplace_of_gaussian_kernel = t.Tensor(laplace_of_gaussian_kernel).to(device)
                # kernel shape from (x,y,z) to (1,1,z,x,y)
                kx, ky, kz = laplace_of_gaussian_kernel.size()
                laplace_of_gaussian_kernel = laplace_of_gaussian_kernel.permute(2, 0, 1).view(1, 1, kz, kx, ky)

                # convolve the LoG kernel with the ground truth to calculate edge loss
                bs, oz, ox, oy = desired_output.size()  # to the shape of (batch_size, 1, z, x, y)
                edginess_tensor = F.conv3d(desired_output.view(bs, 1, oz, ox, oy),
                                            laplace_of_gaussian_kernel, padding=((kz-1)//2, (kx-1)//2, (ky-1)//2))
                # take the absolute value:
                edginess_tensor = t.abs(edginess_tensor)
                # convert to shape (batch_size, z, x, y)
                edginess_tensor = edginess_tensor.view(bs, oz, ox, oy)

                unweighted_regularized_loss *= (1 + EDGE_LOSS_WEIGHT * edginess_tensor)   ###

        if dont_care_mask is not None:
            unweighted_regularized_loss[dont_care_mask] = 0
            # the two rows below are newly added in 20191212!
            assert root_weight==1  # do not apply higher root weight simultaneously with dont care flag
            # num_voxels = int(dont_care_mask.numel())
            # num_voxels_not_dc = int((dont_care_mask==0).sum().item())
            # unweighted_regularized_loss *= num_voxels/num_voxels_not_dc  # this line makes the numerator of the loss the number of non-dont-care voxels instead of all voxels


        weighted_regularized_soil_loss = t.mul(soil_weight_mat, unweighted_regularized_loss)
        weighted_regularized_root_loss = t.mul(root_weight_mat, unweighted_regularized_loss)
        weighted_regularized_loss = t.add(weighted_regularized_root_loss, weighted_regularized_soil_loss)

        # # original way of calculating the gradient different loss
        # if 'calculate_gradient_diff_loss' in kwargs:
        #     if kwargs['calculate_gradient_diff_loss'] is True:
        #         assert 'device' in kwargs
        #         device = kwargs['device']
        #         gdl_loss = self.gradient_diff_loss(desired_output, net_output, device, gdl_weight=GDL_WEIGHT)
        #         # # add only the gradient_diff_loss to weighted_regularized_loss in order to train the model
        #         # print("\n"+"!"*100)
        #         # print("Mean value of NLL loss before adding dgl:", weighted_regularized_loss.mean())
        #         # print("Mean value of Gradient Diff Loss:", gdl_loss.mean())
        #         # print("!"*100)
        #         weighted_regularized_loss += gdl_loss
        #         # print("Mean value of NLL loss after adding dgl:", weighted_regularized_loss.mean())

        if importance_weights is None:
            # get a list of avg loss for all datapoints in the batch
            # shape of weighted_regularized_loss is (batchsize, z, x, y) or (z,x,y) for a single crop
            average_regularized_loss_list = weighted_regularized_loss.mean(-1).mean(-1).mean(-1)
            total_regularized_root_loss_list = weighted_regularized_root_loss.sum(-1).sum(-1).sum(-1)
            total_regularized_soil_loss_list = weighted_regularized_soil_loss.sum(-1).sum(-1).sum(-1)
            total_regularized_loss_list = total_regularized_root_loss_list + total_regularized_soil_loss_list
        else:
            average_regularized_loss_list = (weighted_regularized_loss/importance_weights).mean(-1).mean(-1).mean(-1)
            total_regularized_root_loss_list = (weighted_regularized_root_loss/importance_weights).sum(-1).sum(-1).sum(-1)
            total_regularized_soil_loss_list = (weighted_regularized_soil_loss/importance_weights).sum(-1).sum(-1).sum(-1)
            total_regularized_loss_list = total_regularized_root_loss_list + total_regularized_soil_loss_list

        if unweighted_regularizer_loss == 0:  # no need to calculate again
            average_loss_list = average_regularized_loss_list
            total_root_loss_list = total_regularized_root_loss_list
            total_soil_loss_list = total_regularized_soil_loss_list
            total_loss_list = total_regularized_loss_list
            root_pixels_total_weight_list = root_weight_mat.sum(-1).sum(-1).sum(-1)
            soil_pixels_total_weight_list = soil_weight_mat.sum(-1).sum(-1).sum(-1)

        else:
            weighted_soil_loss = t.mul(soil_weight_mat, unweighted_loss_matrix)
            weighted_root_loss = t.mul(root_weight_mat, unweighted_loss_matrix)
            weighted_loss = t.add(weighted_root_loss, weighted_soil_loss)

            if importance_weights is None:
                average_loss_list = weighted_loss.mean(-1).mean(-1).mean(-1)
                total_root_loss_list = weighted_root_loss.sum(-1).sum(-1).sum(-1)
                total_soil_loss_list = weighted_soil_loss.sum(-1).sum(-1).sum(-1)
                total_loss_list = total_root_loss_list + total_soil_loss_list
                root_pixels_total_weight_list = root_weight_mat.sum(-1).sum(-1).sum(-1)
                soil_pixels_total_weight_list = soil_weight_mat.sum(-1).sum(-1).sum(-1)
            else:
                average_loss_list = (weighted_loss/importance_weights).mean(-1).mean(-1).mean(-1)
                total_root_loss_list = (weighted_root_loss/importance_weights).sum(-1).sum(-1).sum(-1)
                total_soil_loss_list = (weighted_soil_loss/importance_weights).sum(-1).sum(-1).sum(-1)
                total_loss_list = total_root_loss_list + total_soil_loss_list
                root_pixels_total_weight_list = (root_weight_mat/importance_weights).sum(-1).sum(-1).sum(-1)
                soil_pixels_total_weight_list = (soil_weight_mat/importance_weights).sum(-1).sum(-1).sum(-1)

        true_positive_count_list = None
        false_positive_count_list = None
        false_negative_count_list = None
        true_negative_count_list = None

        if 'mode' in kwargs:
            if kwargs['mode'] == 'validation':
                true_positive_count_list, false_positive_count_list, \
                false_negative_count_list, true_negative_count_list = self.classification_statistics(net_output,
                                                                           desired_output,
                                                                           dont_care_mask)

        if len(net_output.size())==3:  # if this loss is for a single crop
            if math.isnan(average_regularized_loss_list.item()):
                raise ValueError('Loss is NaN')
        else:  # if the calculated loss is a list of losses for a batch of data crops
            for i in range(average_regularized_loss_list.size()[0]):
                if math.isnan(average_regularized_loss_list[i].item()):
                    raise ValueError('Loss is NaN')

        # when the batch size is larger than 1, so prediction shape of (bs, z, x, y), the returning values are lists
        # when calculating loss for only one input crop, namely the prediction shape is (z, x, y),
        # the returnning values are single numbers
        return average_regularized_loss_list, total_regularized_loss_list, total_regularized_root_loss_list, \
               total_regularized_soil_loss_list, average_loss_list, total_loss_list, total_root_loss_list, \
               total_soil_loss_list, root_pixels_total_weight_list, soil_pixels_total_weight_list,  \
               iou_loss, true_positive_count_list, \
               false_positive_count_list, false_negative_count_list, true_negative_count_list


def calculate_root_soil_weights(batch_ground_truth, root_weight, mask=None):
    # the mask is 1 in the center, 0 in the surrounding
    if mask is None:
        num_roots = (batch_ground_truth>0).sum()
        num_soils = np.prod(batch_ground_truth.size()) - num_roots
    else:
        num_roots = ((batch_ground_truth > 0)*mask).sum()
        num_soils = mask.sum() - num_roots
    soil_loss_weight = (float(num_soils) + float(num_roots)) \
                       / (float(num_soils) + float(root_weight) * float(num_roots))  # corrected
    root_loss_weight = soil_loss_weight * root_weight

    if root_loss_weight == float('Inf') or root_loss_weight == float('Inf'):  # why duplicated conditions
        root_loss_weight = 0

    return soil_loss_weight, root_loss_weight


def generate_3D_LoG_kernel(sigma_x, sigma_y, sigma_z):

    x = np.linspace(-3*int(np.ceil(sigma_x)), 3*int(np.ceil(sigma_x)), 6*int(np.ceil(sigma_x))+1)
    y = np.linspace(-3*int(np.ceil(sigma_y)), 3*int(np.ceil(sigma_y)), 6*int(np.ceil(sigma_y))+1)
    z = np.linspace(-3*int(np.ceil(sigma_z)), 3*int(np.ceil(sigma_z)), 6*int(np.ceil(sigma_z))+1)

    xx,yy,zz = np.meshgrid(x, y, z, indexing='ij', sparse=True)

    coefficient = 1/((sigma_x * (2*np.pi)**0.5) * (sigma_y * (2*np.pi)**0.5) * (sigma_z * (2*np.pi)**0.5))
    power = -(xx**2/(2*sigma_x**2) + yy**2/(2*sigma_y**2) + zz**2/(2*sigma_z**2))
    LoG_kernel = coefficient * np.exp(power) * (xx**2/sigma_x**4 + yy**2/sigma_y**4 + zz**2/sigma_z**4
                                                - 1/sigma_x**2 - 1/sigma_y**2 - 1/sigma_z**2)

    LoG_kernel -= LoG_kernel.mean()  # make sure the sum is 0

    return LoG_kernel


if __name__ == '__main__':
    import numpy as np

    device = t.device("cuda:0")

    gt_path = '/home/user/zhaoy/Root_MRI/data/ground_truth_res_2x_real_lupinesmall.npy'
    gt = np.load(gt_path).astype(np.float32)
    gt = np.rollaxis(gt, 1, 0)
    print('gt shape:', gt.shape)
    gt /= gt.max()  # to [0,1]
    gt = t.from_numpy(gt).to(device)

    laplace_of_gaussian_kernel = generate_3D_LoG_kernel(1, 1, 0.333)
    laplace_of_gaussian_kernel = t.Tensor(laplace_of_gaussian_kernel).to(device)
    # kernel shape from (x,y,z) to (1,1,z,x,y)
    kx, ky, kz = laplace_of_gaussian_kernel.size()
    laplace_of_gaussian_kernel = laplace_of_gaussian_kernel.permute(2, 0, 1).view(1, 1, kz, kx, ky)

    # convolve the LoG kernel with the ground truth to calculate edge loss
    bs, oz, ox, oy = gt.size()  # to the shape of (batch_size, 1, z, x, y)
    edginess_tensor = F.conv3d(gt.view(bs, 1, oz, ox, oy),
                               laplace_of_gaussian_kernel, padding=((kz - 1) // 2, (kx - 1) // 2, (ky - 1) // 2))
    # take the absolute value:
    edginess_tensor = t.abs(edginess_tensor)
    # convert to shape (batch_size, z, x, y)
    edginess_tensor = edginess_tensor.view(oz, ox, oy).cpu().numpy()

    # save as file
    print('max of edginess_tensor:', edginess_tensor.max())
    print('min of edginess_tensor:', edginess_tensor.min())
    print('shape of edginess_tensor:', edginess_tensor.shape)
    # edginess_tensor = (255 * edginess_tensor / edginess_tensor.max()).astype(np.uint8)
    edginess_tensor = edginess_tensor.astype(np.uint8)
    print('max of edginess_tensor:', edginess_tensor.max())
    print('min of edginess_tensor:', edginess_tensor.min())
    print('shape of LoG kernel:', laplace_of_gaussian_kernel.size())
    np.savez('/home/user/zhaoy/Root_MRI/experiments/temp/lupinesmall_edginess_tensor4', edginess_tensor)
    # np.savez('/home/user/zhaoy/Root_MRI/experiments/temp/laplace_of_gaussian_kernel',
    #          np.squeeze(laplace_of_gaussian_kernel.cpu().numpy()))



