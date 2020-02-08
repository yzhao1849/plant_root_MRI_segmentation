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

        super(ForwardLayers, self).__init__(**kwargs)

        # # TODO: initialize the weights and biases with certain values

        # Reduce the number of conv layers from 2 to 1 in each conv module before pooling/upconv
        # But the number of channels remain the same
        self.conv1 = nn.Sequential(nn.Conv3d(1, 32, 3, padding=0),
                                   nn.BatchNorm3d(32), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv3d(32, 64, 3, padding=0),
                                   nn.BatchNorm3d(64), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv3d(64, 128, 3, padding=0),
                                   nn.BatchNorm3d(128), nn.ReLU())

        self.upconv1 = nn.ConvTranspose3d(128, 128, 2, stride=2)
        self.conv4 = nn.Sequential(nn.Conv3d(64+128, 64, 3, padding=0),
                                   nn.BatchNorm3d(64), nn.ReLU())
        self.upconv2 = nn.ConvTranspose3d(64, 64, 2, stride=2)
        self.conv5 = nn.Sequential(nn.Conv3d(32+64, 32, 3, padding=0),
                                   nn.BatchNorm3d(32), nn.ReLU())###
        # upconv3: for super-resolution, upsample 2*2 the output of the previous part of network
        self.upconv3 = nn.ConvTranspose3d(32, 32, 2, stride=2)
        # upconv4: upsample 2*2 the input directly
        self.upconv4 = nn.ConvTranspose3d(1, 16, 2, stride=2)

        # after concatenating the input and the upsampled input and the network output, another conv3*3 layer
        self.final_conv = nn.Sequential(nn.Conv3d(32+16, 16, 3, padding=0),
                                   nn.BatchNorm3d(16), nn.ReLU(),
                                   nn.Conv3d(16, 1, 3, padding=0))


    def forward(self, x, **kwargs):
        depth, width, height = x.size()[1:4]

        assert (depth % 4 == 0) and (height % 4 == 0) and (width % 4 == 0)
        # because there will be 2 rounds of maxpooling, so in order to lose no information
        ## TODO: deal with the hard coded number 4

        # expand the dimension of input x, from (batchsize, z, x, y) to (batchsize, 1, z, x, y) for 3D convolution
        x = x.unsqueeze(1)

        # print('conv1')
        conv1out = self.conv1(x)
        # print("conv1out.size()", conv1out.size())
        # print('conv2')
        maxpool3d_2 = nn.MaxPool3d(2)
        conv2out = self.conv2(maxpool3d_2(conv1out))
        # print("conv2out.size()", conv2out.size())
        # print('conv3')
        out = self.conv3(maxpool3d_2(conv2out))
        # print("conv3.size()", out.size())
        # print('upconv1')
        out = self.upconv1(out)
        # print("upconv1.size()", out.size())
        out = t.cat((out, match_3D_shape_to(conv2out, out)), dim=1)
        # print('conv4')
        out = self.conv4(out)
        # print('upconv2')
        out = self.upconv2(out)
        out = t.cat((out, match_3D_shape_to(conv1out, out)), dim=1)
        # print('conv5')
        out = self.conv5(out)
        # print("out.size()", out.size())

        out = self.upconv3(out)
        input_upsampled = self.upconv4(x)
        out = t.cat((out, match_3D_shape_to(input_upsampled, out)), dim=1)
        # print("out+input_upsampled.size()", out.size())

        before_sigmoid = self.final_conv(out)

        sigmoid = nn.Sigmoid()
        prediction = sigmoid(before_sigmoid)  # normalize between 0,1
        # print("output size", prediction.size())

        # squeeze the dimension of input x, from (batchsize, 1, z, x, y) to (batchsize, z, x, y)
        prediction = prediction.squeeze(1)
        before_sigmoid = before_sigmoid.squeeze(1)

        return prediction, before_sigmoid


def match_3D_shape_to(t_big, t_small):
    diff_w = t_big.size()[4] - t_small.size()[4]
    diff_h = t_big.size()[3] - t_small.size()[3]
    diff_d = t_big.size()[2] - t_small.size()[2]

    t_big = t_big[:, :,
                  diff_d//2:diff_d//2 + t_small.size()[2],
                  diff_h//2:diff_h//2 + t_small.size()[3],
                  diff_w//2:diff_w//2 + t_small.size()[4]]
    return t_big


class Net(BaseNet):
    def __init__(self, **kwargs):
        super(Net, self).__init__(ForwardLayers=ForwardLayers, **kwargs)
        # self.input_depth = kwargs["input_depth"]

    def model_type(self):
        return "UNet_3D_noPadding_BNbeforeReLU_smallerReceptiveField"










