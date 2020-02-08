import torch.nn as nn
import torch.nn.functional as F
import torch


class BaseNet(nn.Module):
    """
    Must be used and
    """
    def __init__(self, ForwardLayers, **kwargs):
        """

        param num_layers: An integer, number of layers
        param num_channels: An integer, number of layers
        :param ForwardLayers: The class object constructor, not an instance!!! Internally called to instantiate network
                                Must extend from ForwardLayersBase
        :param Loss: Simply the Loss class constructor inside Loss.py, no other Loss mechanism is implemented yet.
        :param confidence_penalty:
        :param super_resolution_factor:
        :param kwargs:
        """
        super(BaseNet, self).__init__()
        self.forward_net = ForwardLayers(**kwargs)

    # def forward(self, input, desired_output, root_weight, root_weight_map, **kwargs):
    def forward(self, input, **kwargs):
        net_output, before_sigmoid = self.forward_net(input, **kwargs)
        return net_output, before_sigmoid

    def calculate_shape_decreases_3D_Net(self, input_crop_size):
        """
        Calculate the shape decrease between the output and input image
        This decrease is not dependent on input size, but only on network structure
        :param input_crop_size: should be a list of crop side lengths, with the order in x,y,z
        :return:
        """
        cropsize_x, cropsize_y, cropsize_z = input_crop_size
        input_crop = torch.ones((1, cropsize_z, cropsize_x, cropsize_y))
        net_output, _ = self.forward_net(input_crop)
        _, outsize_z, outsize_y, outsize_x = net_output.size()

        # #debugging
        # print("input shape:", input_crop.size())
        # print("output shape:", net_output.size())

        # outsize_z = int(outsize_z/self.super_resolution_factor)
        # outsize_y = int(outsize_y/self.super_resolution_factor)
        # outsize_x = int(outsize_x/self.super_resolution_factor)

        return cropsize_x-outsize_x, cropsize_y-outsize_y, cropsize_z-outsize_z


class ForwardLayersBase(nn.Module):
    def __init__(self, **kwargs):
        """
        Must be implemented in the subclass, you put whatever your network uses (convolutional layers etc.) into this
        class.
        :param num_layers:
        :param num_channels:
        :param super_resolution_factor:
        """
        super(ForwardLayersBase, self).__init__()
        pass

    def forward(self, x, **kwargs):
        """
        Must be implemented in the subclass
        :param x: the input to the network, volume mapped to RGB
        :param kwargs: if the network uses additional information, they can be provided inside kwargs
        :return: the network before and after sigmoid function
        """
        pass



