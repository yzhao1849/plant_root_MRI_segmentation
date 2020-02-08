import torch.nn as nn
import torch as t

from BaseNet import BaseNet

from BaseNet import ForwardLayersBase
import torch.nn.functional as F


class ForwardLayers(ForwardLayersBase):
    resnet_factory = None

    def layer_n_name(self, n):
        return "resconv_{0}".format(n)

    def __init__(self, **kwargs): ###
        """
        7-Cascade RefineNet for super-resolution binary image segmentation
        :param residual_layers: number of residual layers
        :param residual_connection_size: number of channels at each RefineNet block
        :param super_resolution_factor: the output has k times the resolution in each axis
        :param kwargs:
        """
        # # super(ForwardLayers, self).__init__()
        # super(ForwardLayers, self).__init__(residual_layers, residual_connection_size, super_resolution_factor)
        #
        # self.residual_layers = residual_layers
        # self.residual_connection_size = residual_connection_size
        # self.padding = 6 + residual_layers * 2
        #
        # self.size_stats = kwargs.get("size_stats", 0)
        # # self.use_hd = kwargs['use_hd'] # not used if using the random cropping new DataLoader.py
        # self.input_depth = kwargs['input_depth']
        #
        # freeze_resnet = kwargs['freeze_resnet']
        #
        # # add one conv layer with 3 channels before ResNet
        # self.first_conv = t.nn.Conv2d(self.input_depth, 3, kernel_size = 1)
        # # TODO: initialize the weights and biases with certain values
        #
        # # using sigmoid to do normalization, the result is in the range (-0.5,0.5)
        # self.sigmoid0=t.nn.Sigmoid()
        #
        #
        # self.srf = super_resolution_factor
        # self.resnet, resnet_layer_sizes = self.resnet_factory(pretrained=True)
        # # for resnet18, resnet_layer_sizes is [64, 64, 128, 256, 512], meaning number of channels?
        #
        # self.upsampler = nn.Upsample(scale_factor=super_resolution_factor, mode='bilinear')
        #
        # for param in self.resnet.parameters():
        #     param.requires_grad = not freeze_resnet
        #     # No need to set requires_grad to True to be able to update the trainable layer before ResNet
        #
        # self.refinenet4 = RefineNetBlock(resnet_layer_sizes[4], resnet_layer_sizes[4])
        # self.refinenet3 = RefineNetBlock(residual_connection_size, resnet_layer_sizes[4], resnet_layer_sizes[3])
        # self.refinenet2 = RefineNetBlock(residual_connection_size, residual_connection_size, resnet_layer_sizes[2])
        # self.refinenet1 = RefineNetBlock(residual_connection_size, residual_connection_size, resnet_layer_sizes[1])
        # self.refinenet0 = RefineNetBlock(residual_connection_size, residual_connection_size, resnet_layer_sizes[0])
        #
        # # non_tra_input_size =  3 + self.size_stats if not self.use_hd else 5 + self.size_stats
        # non_tra_input_size =  self.input_depth + self.size_stats # self.size_stats is the number of additional info layer
        # self.refinenet_input_fuser = RefineNetBlock(residual_connection_size, residual_connection_size, non_tra_input_size)
        # self.refinenet_super_res_fuser = RefineNetBlock(residual_connection_size, residual_connection_size, non_tra_input_size)
        #
        # self.final_conv = t.nn.Conv2d(residual_connection_size, super_resolution_factor*self.input_depth, kernel_size=1)
        self.srf = super_resolution_factor

        super(ForwardLayers, self).__init__(num_layers, num_channels, super_resolution_factor)
        input_depth = kwargs['input_depth']
        self.conv1 = nn.Sequential(nn.Conv2d(input_depth, input_depth * 3, 3, padding=1),
                                   nn.ReLU(), nn.BatchNorm2d(input_depth*3),
                                   nn.Conv2d(input_depth * 3, input_depth * 3, 3, padding=1),
                                   nn.ReLU(), nn.BatchNorm2d(input_depth*3))
        self.conv2 = nn.Sequential(nn.Conv2d(input_depth * 3, input_depth * 6, 3, padding=1),
                                   nn.ReLU(), nn.BatchNorm2d(input_depth*6),
                                   nn.Conv2d(input_depth * 6, input_depth * 6, 3, padding=1),
                                   nn.ReLU(), nn.BatchNorm2d(input_depth*6))
        self.conv3 = nn.Sequential(nn.Conv2d(input_depth * 6, input_depth * 12, 3, padding=1),
                                   nn.ReLU(), nn.BatchNorm2d(input_depth*12),
                                   nn.Conv2d(input_depth * 12, input_depth * 12, 3, padding=1),
                                   nn.ReLU(), nn.BatchNorm2d(input_depth*12))
        self.upconv1 = nn.ConvTranspose2d(input_depth * 12, input_depth * 6, 2, stride=2)
        self.conv4 = nn.Sequential(nn.Conv2d(input_depth * 12, input_depth * 6, 3, padding=1),
                                   nn.ReLU(), nn.BatchNorm2d(input_depth*6),
                                   nn.Conv2d(input_depth * 6, input_depth * 6, 3, padding=1),
                                   nn.ReLU(), nn.BatchNorm2d(input_depth*6))
        self.upconv2 = nn.ConvTranspose2d(input_depth * 6, input_depth * 3, 2, stride=2)
        self.conv5 = nn.Sequential(nn.Conv2d(input_depth * 6, input_depth * 3, 3, padding=1),
                                   nn.ReLU(), nn.BatchNorm2d(input_depth*3),
                                   nn.Conv2d(input_depth * 3, input_depth, 3, padding=1),
                                   nn.ReLU(), nn.BatchNorm2d(input_depth))
        # upconv3: for super-resolution, upsample 2*2 the output of the previous part of network
        self.upconv3 = nn.ConvTranspose2d(input_depth, input_depth * 2, 2, stride=2)
        # upconv4: upsample 2*2 the input directly
        self.upconv4 = nn.ConvTranspose2d(input_depth, input_depth * 2, 2, stride=2)

        # after concatenating the input and the upsampled input and the network output, another conv3*3 layer
        self.final_conv = nn.Conv2d(input_depth * 4, input_depth * 2, 3, padding=1)

    def forward(self, input, **kwargs):
        # print("input size", input.size())

        # # input = t.ones([1,1,23,17])
        # [width, height] = input.size()[2:4]
        #
        # # non_normed_input = kwargs.get('batch_data_non_normalized', None) # not use at all
        # ## orig_5 = kwargs.get('noisies') # not used if using the random cropping new DataLoader.py
        # if width % 32 != 0:
        #     req_hor_padding = 32 - (width % 32)
        #     left_pad = int(req_hor_padding/2)
        #     right_pad = req_hor_padding - left_pad
        # else:
        #     left_pad = right_pad = 0
        #
        # if height % 32 != 0:
        #     req_ver_padding = 32 - (height % 32)
        #     top_pad = int(req_ver_padding/2)
        #     bottom_pad = req_ver_padding - top_pad
        # else:
        #     top_pad = bottom_pad = 0
        #
        # padded_input = F.pad(input=input, pad=(bottom_pad, top_pad, left_pad, right_pad), mode="constant")
        # original_padded_input = padded_input.clone()
        # padded_input = self.first_conv(padded_input)  # the shape of tensor remain the same
        # padded_input = self.sigmoid0(padded_input)-0.5  # normalize to (-0.5, 0.5)
        # # if non_normed_input is not None:
        # #     padded_non_normed_input = F.pad(input=non_normed_input, pad=(bottom_pad, top_pad, left_pad, right_pad),
        # #                                     mode="constant")
        # # else:
        # #     padded_non_normed_input = None
        # ## padded_orig_5 = F.pad(input=orig_5, pad=(bottom_pad, top_pad, left_pad, right_pad), mode="constant") if self.use_hd else None
        #
        # res_0, res_1, res_2, res_3, res_4 = self.resnet(padded_input)
        #
        # path_4 = self.refinenet4(res_4)
        # path_3 = self.refinenet3(path_4, res_3)
        # path_2 = self.refinenet2(path_3, res_2)
        # path_1 = self.refinenet1(path_2, res_1)
        # path_0 = self.refinenet0(path_1, res_0)
        # srf    = self.srf
        #
        # if self.size_stats > 1:
        #     stats = kwargs.get("stats", None)
        #     #upsampled_orig_5 =  padded_orig_5 if self.use_hd else padded_input
        #     padded_stats = F.pad(input=stats, pad=(original_padded_input.size(2)-1, 0, original_padded_input.size(3)-1, 0), mode="replicate")
        #     original_padded_input = t.cat((padded_stats, original_padded_input), 1)
        #
        # upsampled_input = self.upsampler(original_padded_input)
        # input_res = self.refinenet_input_fuser(path_0, original_padded_input)
        # super_res = self.refinenet_super_res_fuser(input_res, upsampled_input)
        # # TODO if no padding, just identity
        # if left_pad != 0:
        #     super_res = super_res[:, :, (left_pad*srf):(-srf*right_pad), (srf*bottom_pad):(-srf*top_pad)]
        #
        # before_sigmoid = self.final_conv(super_res)
        #
        # prediction = sigmoid(before_sigmoid)
        #
        # return prediction, before_sigmoid
        height, width = input.size()[2:4]
        assert (height % 4 == 0) and (width % 4 == 0)  # because there will be 2 rounds of maxpooling, so in order to loss no information
        if height % 4 != 0:  # because there will be 2 rounds of maxpooling, so in order to make the shape compatible
            # TODO: deal with the hard coded number 4
            padding = 4 - height % 4
            pad_top = int(padding / 2)
            pad_bottom = padding - pad_top
        else:
            pad_top = 0
            pad_bottom = 0

        if width % 4 != 0:
            padding = 4 - width % 4
            pad_left = int(padding / 2)
            pad_right = padding - pad_left
        else:
            pad_left = 0
            pad_right = 0
        input = F.pad(input=input, pad=(pad_left, pad_right, pad_top, pad_bottom), mode="constant") # zero padding
        # print("input size after padding:", input.size())

        # print('conv1')
        conv1out = self.conv1(input)
        # print("conv1out.size()", conv1out.size())
        # print('conv2')
        maxpool2d_2 = nn.MaxPool2d(2)
        conv2out = self.conv2(maxpool2d_2(conv1out))
        # print("conv2out.size()", conv2out.size())
        # print('conv3')
        out = self.conv3(maxpool2d_2(conv2out))
        # print("conv3.size()", out.size())
        # print('upconv1')
        out = self.upconv1(out)
        # print("upconv1.size()", out.size())
        out = t.cat((out, conv2out), dim=1)
        # print('conv4')
        out = self.conv4(out)
        # print('upconv2')
        out = self.upconv2(out)
        out = t.cat((out, conv1out), dim=1)
        # print('conv5')
        out = self.conv5(out)

        out = self.upconv3(out)
        input_upsampled = self.upconv4(input)
        out = t.cat((out, input_upsampled), dim=1)
        # print("output size", out.size())

        before_sigmoid = self.final_conv(out)
        # print("before_sigmoid size", before_sigmoid.size())
        if pad_left!=0 or pad_right!=0 or pad_top!=0 or pad_bottom!=0:
            # make sure the result is expected shape (super_res_factor * original size)
            before_sigmoid = before_sigmoid[:, :, self.srf*pad_top:-self.srf*pad_bottom, self.srf*pad_left:-self.srf*pad_right]

        sigmoid = nn.Sigmoid()
        prediction = sigmoid(before_sigmoid)  # normalize between 0,1
        # print("output size", prediction.size())


        return prediction, before_sigmoid


class Net(BaseNet):
    def __init__(self, **kwargs):
        super(Net, self).__init__(ForwardLayers=ForwardLayers, **kwargs)
        # self.input_depth = kwargs["input_depth"]

    def model_type(self):
        return "2DUNet_inputDepth:{0}".format(self.input_depth)










