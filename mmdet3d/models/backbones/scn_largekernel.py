# Copyright (c) LargeKernel. All rights reserved.
import torch
from torch import nn

from mmdet3d.ops import spconv
from mmdet3d.ops.spconv import SparseConv3d, SubMConv3d
from mmdet3d.ops import make_sparse_convmodule
from mmcv.cnn import build_norm_layer
from mmdet.models import BACKBONES


def conv(in_planes, out_planes, kernel_size=3, stride=1, indice_key=None, bias=True):
    """convolution with padding"""
    return SubMConv3d(
        in_planes,
        out_planes,
        kernel_size=kernel_size,
        stride=stride,
        padding=int(kernel_size//2),
        bias=bias,
        indice_key=indice_key,
    )

def conv1x1(in_planes, out_planes, stride=1, indice_key=None, bias=True):
    """1x1 convolution"""
    return SubMConv3d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        padding=1,
        bias=bias,
        indice_key=indice_key,
    )

class SpatialGroupConv(spconv.SparseModule):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, indice_key=None, bias=False):
        super(SpatialGroupConv, self).__init__()
        self.kernel_size = kernel_size
        self.indice_key = indice_key
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.block = SubMConv3d(in_channels,
                                out_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=int(kernel_size//2),
                                bias=bias,
                                indice_key=indice_key,
                                )

        self.conv3x3_1 = SubMConv3d(in_channels,
                                    out_channels,
                                    kernel_size=3,
                                    stride=stride,
                                    padding=int(kernel_size//2)-1,
                                    bias=bias,
                                    dilation=int(kernel_size//2)-1,
                                    indice_key=indice_key+'conv_3x3_1',
                                )
        self._indice_list = []

        if kernel_size==7:
            _list = [0, 3, 4, 7]
        elif kernel_size==5:
            _list = [0, 2, 3, 5]
        else:
            raise ValueError('Unknown kernel size %d'%kernel_size)
        for i in range(len(_list)-1):
            for j in range(len(_list)-1):
                for k in range(len(_list)-1):
                    a = torch.zeros((kernel_size, kernel_size, kernel_size)).long()
                    a[_list[i]:_list[i+1], _list[j]:_list[j+1], _list[k]:_list[k+1]] = 1
                    b = torch.range(0, kernel_size**3-1, 1)[a.reshape(-1).bool()]
                    self._indice_list.append(b.long())

    def _convert_weight(self, weight):
        weight_reshape = self.block.weight.permute(3, 4, 0, 1, 2).reshape(self.out_channels, self.in_channels, -1).clone()
        weight_return = self.block.weight.permute(3, 4, 0, 1, 2).reshape(self.out_channels, self.in_channels, -1).clone()
        for _indice in self._indice_list:
            _mean_weight = torch.mean(weight_reshape[:, :, _indice], dim=-1, keepdim=True)
            weight_return[:, :, _indice] = _mean_weight
        return weight_return.reshape(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size, self.kernel_size).permute(2, 3, 4, 0, 1)

    def forward(self, x_conv):
        if self.training:
            self.block.weight.data = self._convert_weight(self.block.weight.data)
        x_conv_block = self.block(x_conv)
        x_conv_conv3x3_1 = self.conv3x3_1(x_conv)
        x_conv_block.features = x_conv_block.features + x_conv_conv3x3_1.features
        return x_conv_block


class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        kernel_size=3,
        stride=1,
        norm_cfg=None,
        downsample=None,
        indice_key=None,
        conv_type='common',
    ):
        super(SparseBasicBlock, self).__init__()

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)

        bias = norm_cfg is not None
        if conv_type=="spatialgroupconv":
            conv_func = SpatialGroupConv
        elif conv_type=='common':
            conv_func = conv
        else:
            raise ValueError('Unknown conv type %s.'%conv_type)

        self.conv1 = conv_func(inplanes, planes, kernel_size, stride, indice_key=indice_key, bias=bias)
        self.bn1 = build_norm_layer(norm_cfg, planes)[1]
        self.relu = nn.ReLU(True)
        self.conv2 = conv_func(planes, planes, kernel_size, indice_key=indice_key, bias=bias)
        self.bn2 = build_norm_layer(norm_cfg, planes)[1]
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x.features

        assert x.features.dim() == 2, f"x.features.dim()={x.features.dim()}"

        out = self.conv1(x)
        out.features = self.bn1(out.features)
        out.features = self.relu(out.features)

        out = self.conv2(out)
        out.features = self.bn2(out.features)

        if self.downsample is not None:
            identity = self.downsample(x)

        out.features += identity
        out.features = self.relu(out.features)

        return out


@BACKBONES.register_module()
class SparseLargeKernelEncoder(nn.Module):
    r"""Sparse LargeKernel Encoder for LargeKernel.

    Args:
        in_channels (int): The number of input channels.
        sparse_shape (list[int]): The sparse shape of input tensor.
        norm_cfg (dict, optional): Config of normalization layer. Defaults to
            dict(type='BN1d', eps=1e-3, momentum=0.01).
        name (str): The name of Sparse LargeKernel Encoder.
    """
    def __init__(
        self,
        in_channels,
        sparse_shape,
        output_channels,
        kernel_sizes,
        conv_types,
        kernel_sizes_downsample,
        norm_cfg=dict(type="BN1d", eps=1e-3, momentum=0.01),
        name="SparseLargeKernelLargeKernel",
        **kwargs
    ):
        super(SparseLargeKernelEncoder, self).__init__()
        self.name = name
        self.sparse_shape = sparse_shape
        self.in_channels = in_channels
        self.output_channels = output_channels
        self.conv_types = conv_types
        self.kernel_sizes = kernel_sizes
        self.kernel_sizes_downsample = kernel_sizes_downsample

        self.conv_input = make_sparse_convmodule(
            self.in_channels,
            16,
            3,
            norm_cfg=norm_cfg,
            padding=1,
            indice_key="subm",
            conv_type="SubMConv3d",
        )

        self.conv1 = spconv.SparseSequential(
            SparseBasicBlock(16, 16, kernel_size=self.kernel_sizes[0], norm_cfg=norm_cfg, indice_key="res0",
                conv_type=self.conv_types[0]),
            SparseBasicBlock(16, 16, kernel_size=self.kernel_sizes[0], norm_cfg=norm_cfg, indice_key="res0",
                conv_type=self.conv_types[0]),
        )

        self.conv2 = spconv.SparseSequential(
            SparseConv3d(16, 32, self.kernel_sizes_downsample[0], 2, padding=int(self.kernel_sizes_downsample[0]//2), bias=False),
            build_norm_layer(norm_cfg, 32)[1],
            nn.ReLU(inplace=True),
            SparseBasicBlock(32, 32, kernel_size=self.kernel_sizes[1], norm_cfg=norm_cfg, indice_key="res1",
                conv_type=self.conv_types[1]),
            SparseBasicBlock(32, 32, kernel_size=self.kernel_sizes[1], norm_cfg=norm_cfg, indice_key="res1",
                conv_type=self.conv_types[1]),
        )

        self.conv3 = spconv.SparseSequential(
            SparseConv3d(32, 64, self.kernel_sizes_downsample[1], 2, padding=int(self.kernel_sizes_downsample[1]//2), bias=False),
            build_norm_layer(norm_cfg, 64)[1],
            nn.ReLU(inplace=True),
            SparseBasicBlock(64, 64, kernel_size=self.kernel_sizes[2], norm_cfg=norm_cfg, indice_key="res2",
                conv_type=self.conv_types[2]),
            SparseBasicBlock(64, 64, kernel_size=self.kernel_sizes[2], norm_cfg=norm_cfg, indice_key="res2",
                conv_type=self.conv_types[2]),
        )

        self.conv4 = spconv.SparseSequential(
            SparseConv3d(64, 128, 3, 2, padding=[0, 1, 1], bias=False),
            build_norm_layer(norm_cfg, 128)[1],
            nn.ReLU(inplace=True),
            SparseBasicBlock(128, 128, kernel_size=self.kernel_sizes[3], norm_cfg=norm_cfg, indice_key="res3",
                conv_type=self.conv_types[3]),
            SparseBasicBlock(128, 128, kernel_size=self.kernel_sizes[3], norm_cfg=norm_cfg, indice_key="res3",
                conv_type=self.conv_types[3]),
        )

        self.conv_out = make_sparse_convmodule(
            128,
            self.output_channels,
            kernel_size=(1, 1, 3),
            stride=(1, 1, 2),
            norm_cfg=norm_cfg,
            padding=0,
            indice_key="spconv_down",
            conv_type="SparseConv3d",
        )

    def forward(self, voxel_features, coors, batch_size, **kwargs):
        """Forward of SparseEncoder.

        Args:
            voxel_features (torch.float32): Voxel features in shape (N, C).
            coors (torch.int32): Coordinates in shape (N, 4),
                the columns in the order of (batch_idx, z_idx, y_idx, x_idx).
            batch_size (int): Batch size.

        Returns:
            dict: Backbone features.
        """
        coors = coors.int()
        input_sp_tensor = spconv.SparseConvTensor(
            voxel_features, coors, self.sparse_shape, batch_size
        )

        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)
        spatial_features = out.dense()

        N, C, H, W, D = spatial_features.shape
        spatial_features = spatial_features.permute(0, 1, 4, 2, 3).contiguous()
        spatial_features = spatial_features.view(N, C * D, H, W)

        return spatial_features
