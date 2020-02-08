import torch.nn as nn
import torch as t

from BaseNet import BaseNet

from BaseNet import ForwardLayersBase
import torch.nn.functional as F


class ForwardLayers(ForwardLayersBase):
    resnet_factory = None

    def layer_n_name(self, n):
        return "resconv_{0}".format(n)

    def __init__(self, num_layers, num_channels, super_resolution_factor, **kwargs): ###
        """
        7-Cascade RefineNet for super-resolution binary image segmentation
        :param residual_layers: number of residual layers
        :param residual_connection_size: number of channels at each RefineNet block
        :param super_resolution_factor: the output has k times the resolution in each axis
        :param kwargs:
        """

        super(ForwardLayers, self).__init__(num_layers, num_channels, super_resolution_factor)

        # # TODO: initialize the weights and biases with certain values
        self.conv1 = nn.Sequential(nn.Conv3d(1, 1, (5,9,9), padding=0), nn.BatchNorm3d(1))


        self.upconv1 = nn.ConvTranspose3d(1, 1, 2, stride=2)


    def forward(self, x, **kwargs):
        depth, width, height = x.size()[1:4]
        assert (depth % 4 == 0) and (height % 4 == 0) and (width % 4 == 0)
        # because there will be 2 rounds of maxpooling, so in order to loss no information
        ## TODO: deal with the hard coded number 4

        # expand the dimension of input x, from (batchsize, z, x, y) to (batchsize, 1, z, x, y) for 3D convolution
        x = x.unsqueeze(1)

        # print('conv1')
        conv1out = self.conv1(x)
        # print("conv1out.size()", conv1out.size())

        before_sigmoid = self.upconv1(conv1out)

        sigmoid = nn.Sigmoid()
        prediction = sigmoid(before_sigmoid)  # normalize between 0,1
        # print("output size", prediction.size())

        # squeeze the dimension of input x, from (batchsize, 1, z, x, y) to (batchsize, z, x, y)
        prediction = prediction.squeeze(1)
        before_sigmoid = before_sigmoid.squeeze(1)

        return prediction, before_sigmoid



class Net(BaseNet):
    def __init__(self, num_layers=None, num_channels=None, super_resolution_factor=2, **kwargs):
        super().__init__(num_layers, num_channels, ForwardLayers, super_resolution_factor, **kwargs)
        # self.input_depth = kwargs["input_depth"]

    def model_type(self):
        return "3D_simpleConv_plus_upsampling"










