from Loss import *
from Utils.constants import *

log_2 = math.log(2)

class Loss2D(nn.Module):

    def __init__(self, confidence_penalty=0, super_resolution_factor=1):
        super(Loss2D, self).__init__()
        self.conf_pen = confidence_penalty
        self.super_resolution_factor = super_resolution_factor
        print("Conf Penalty is {}".format(confidence_penalty))

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
            num_tns_list = ((1 - positive_predictions) * (1 - byte_ground_truth)).sum(-1).sum(-1).sum(-1)
        else:
            num_tns_list = ((1 - positive_predictions) * (1 - byte_ground_truth)).sum(-1).sum(-1).sum(-1) \
                           - (dont_care_mask > 0).sum(-1).sum(-1).sum(-1)

        return num_tps_list, num_fps_list, num_fns_list, num_tns_list

    def MSE_loss(self, probability, ground_truth, importance_weights):
        if importance_weights is None:
            loss_mat = (probability-ground_truth)**2
        else:
            loss_mat = (probability - ground_truth) ** 2 / importance_weights

        return loss_mat

    def forward(self, net_output, desired_output, before_sigmoid, root_weight, **kwargs):
        if 'importance_weights' in kwargs:
            importance_weights = kwargs['importance_weights']
            importance_weights.requires_grad = False
            importance_weights = importance_weights.view(-1, 1, 1, 1)
            # expand the dimensions of importance_weights to be compatible for multiplication
        else:
            importance_weights = None

        side_mask = t.ones(desired_output.size())
        side_mask = side_mask.float()
        side_mask = side_mask.to(kwargs['device'])
        if COPY_EVERYTHING_FROM_NILS and kwargs['mode'] == 'training':  # mask outer side of the crop
            assert MASK_SIDE*2 <= min(desired_output.size()[1:])
            side_mask[:, :MASK_SIDE, :, :] = 0
            side_mask[:, -MASK_SIDE:, :, :] = 0
            side_mask[:, :, :MASK_SIDE, :] = 0
            side_mask[:, :, -MASK_SIDE:, :] = 0
            side_mask[:, :, :, :MASK_SIDE] = 0
            side_mask[:, :, :, -MASK_SIDE:] = 0

        MSE_loss = self.MSE_loss(net_output, desired_output, importance_weights)
        MSE_loss *= side_mask

        desired_output = (desired_output>0).float()  # important for the next steps! convert desired_output to binary
        soil_mat = 1. - desired_output  # TODO: !!! Only works when desired_output is binary
        root_mat = desired_output
        soil_mat *= side_mask
        root_mat *= side_mask

        soil_pixel_weight, root_pixel_weight = calculate_root_soil_weights(desired_output, root_weight, mask=side_mask.byte())
        soil_weight_mat = t.mul(soil_mat, soil_pixel_weight)
        root_weight_mat = t.mul(root_mat, root_pixel_weight)
        root_pixels_total_weight_list = root_weight_mat.sum(-1).sum(-1).sum(-1)
        soil_pixels_total_weight_list = soil_weight_mat.sum(-1).sum(-1).sum(-1)

        weighted_soil_MSE_loss = t.mul(soil_weight_mat, MSE_loss)
        weighted_root_MSE_loss = t.mul(root_weight_mat, MSE_loss)
        weighted_MSE_loss = t.add(weighted_root_MSE_loss, weighted_soil_MSE_loss)

        average_MSE_loss_list = weighted_MSE_loss.sum(-1).sum(-1).sum(-1) / side_mask.sum(-1).sum(-1).sum(-1)  # !!!
        total_MSE_root_loss_list = weighted_root_MSE_loss.sum(-1).sum(-1).sum(-1)
        total_MSE_soil_loss_list = weighted_soil_MSE_loss.sum(-1).sum(-1).sum(-1)
        total_MSE_loss_list = total_MSE_root_loss_list + total_MSE_soil_loss_list

        true_positive_count_list = None
        false_positive_count_list = None
        false_negative_count_list = None
        true_negative_count_list = None

        if 'mode' in kwargs:
            if kwargs['mode'] == 'validation':
                true_positive_count_list, false_positive_count_list, \
                false_negative_count_list, true_negative_count_list = self.classification_statistics(net_output,
                                                                           desired_output,
                                                                           dont_care_mask=(1-side_mask.byte()))

        if math.isnan(MSE_loss.mean().item()):
            raise ValueError('Loss is NaN')

        return average_MSE_loss_list, total_MSE_loss_list, total_MSE_root_loss_list, total_MSE_soil_loss_list, \
               None, None, None, None, \
               root_pixels_total_weight_list, soil_pixels_total_weight_list, None, \
               true_positive_count_list, false_positive_count_list, false_negative_count_list, true_negative_count_list


if __name__ == '__main__':

    def prepare_net_output(path):
        net_output = np.load(path)['arr_0']
        net_output = net_output/255
        net_output = np.pad(net_output, ((5, 5), (5, 5), (5, 5)), 'constant', constant_values=0)
        net_output = t.from_numpy(net_output)
        net_output = t.unsqueeze(net_output, 0)
        net_output = net_output.float()
        print(net_output.dtype)
        print(net_output.max())
        print(net_output.min())
        print(net_output.size())
        return net_output

    # load data that is used for training
    net_output1 = prepare_net_output('/home/user/zhaoy/Root_MRI/temp/out_sand_unsat_0.34%wc_70x1x256x256_-1_rootLoss0.000000_soilLoss0.059775_time1570356209.npz')
    net_output2 = prepare_net_output('/home/user/zhaoy/Root_MRI/temp/out_sand_unsat_0.34%wc_70x1x256x256_-1_rootLoss1.194313_soilLoss1.158121_time1570356210.npz')
    net_output = t.cat((net_output1, net_output2), dim=0)

    def prepare_desired_output(path):
        desired_output = np.load(path)['arr_0']
        print('root percentage:', (desired_output > 0).sum() / desired_output.size)
        desired_output = desired_output / 255
        desired_output = np.pad(desired_output, ((5, 5), (5, 5), (5, 5)), 'constant', constant_values=0)
        desired_output = t.from_numpy(desired_output)
        desired_output = t.unsqueeze(desired_output, 0)
        desired_output = desired_output.float()
        print(desired_output.dtype)
        print(desired_output.max())
        print(desired_output.size())
        return desired_output

    desired_output1 = prepare_desired_output('/home/user/zhaoy/Root_MRI/temp/gt_sand_unsat_0.34%wc_70x1x256x256_-1_rootPerc0.00000_time1570356209.npz')
    desired_output2 = prepare_desired_output('/home/user/zhaoy/Root_MRI/temp/gt_sand_unsat_0.34%wc_70x1x256x256_-1_rootPerc0.03983_time1570356210.npz')
    desired_output = t.cat((desired_output1, desired_output2), dim=0)


    COPY_EVERYTHING_FROM_NILS = True
    MASK_SIDE = 5

    root_weight = 1
    # test the loss calculation
    kwargs = {}
    kwargs['mode'] = 'training'
    kwargs['device'] = 'cpu'
    l = Loss2D()
    loss_results = l(net_output, desired_output, None, root_weight, **kwargs)
    result_name_list = ['average_regularized_loss_list', 'total_regularized_loss_list',
                        'total_regularized_root_loss_list',
                        'total_regularized_soil_loss_list', 'average_loss_list', 'total_loss_list',
                        'total_root_loss_list',
                        'total_soil_loss_list', 'root_pixels_total_weight_list', 'soil_pixels_total_weight_list',
                        'iou_loss', 'true_positive_count_list',
                        'false_positive_count_list', 'false_negative_count_list', 'true_negative_count_list']
    for i in range(len(loss_results)):
        out = loss_results[i]
        print(result_name_list[i]+':', end=' ')
        if out is not None:
            print(out.numpy())
        else:
            print(out)

    loss_mse = nn.MSELoss(reduction='none')  #
    loss = loss_mse(net_output, desired_output)

    soil_pixel_weight, root_pixel_weight = calculate_root_soil_weights(desired_output, root_weight,
                                                                       mask=None)
    loss[desired_output>0] *= root_pixel_weight
    loss[desired_output==0] *= soil_pixel_weight
    # print(loss)
    print('sum of loss:', loss.sum().item())
    print('mean of loss:', loss.mean().item())
