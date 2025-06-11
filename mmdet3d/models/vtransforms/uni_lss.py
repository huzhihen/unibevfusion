from typing import Tuple

import torch
import torch.nn.functional as F
from torch.cuda.amp.autocast_mode import autocast
from mmcv.cnn.resnet import BasicBlock
from mmcv.cnn import build_conv_layer
from mmcv.runner import force_fp32
from torch import nn

from mmdet3d.models.builder import VTRANSFORMS

from .base import BaseDepthTransform

__all__ = ["UniLSSTransform"]


class Map_Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Map_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out


class MapNet(nn.Module):
    def __init__(self, block,  input_channel, nb_filter, block_nums):
        super(MapNet, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv0 = self._make_layer(block, input_channel, nb_filter[0])
        self.conv1 = self._make_layer(block, nb_filter[0], nb_filter[1], block_nums[0])
        self.conv2 = self._make_layer(block, nb_filter[1], nb_filter[2], block_nums[1])
        self.conv3 = self._make_layer(block, nb_filter[2], nb_filter[3], block_nums[2])

    def _make_layer(self, block, input_channels, output_channels, block_nums=1):
        layers = []
        layers.append(block(input_channels, output_channels))
        for i in range(block_nums - 1):
            layers.append(block(output_channels, output_channels))
        return nn.Sequential(*layers)

    def forward(self, input):
        x0 = self.conv0(input)           # x0->256*704*8
        x1 = self.conv1(self.pool(x0))   # x1->128*352*16
        x2 = self.conv2(self.pool(x1))   # x2->64*176*32
        x3 = self.conv3(self.pool(x2))   # x3->32*88*64
        return x3


class DepthRefinement(nn.Module):

    def __init__(self, in_channels, mid_channels, out_channels):
        super(DepthRefinement, self).__init__()

        self.reduce_conv = nn.Sequential(
            nn.Conv2d(in_channels,
                      mid_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        self.conv = nn.Sequential(
            nn.Conv2d(mid_channels,
                      mid_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels,
                      mid_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        self.out_conv = nn.Sequential(
            nn.Conv2d(mid_channels,
                      out_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True),
            # nn.BatchNorm3d(out_channels),
            # nn.ReLU(inplace=True),
        )

    @autocast(False)
    def forward(self, x):
        x = self.reduce_conv(x)
        x = self.conv(x) + x
        x = self.out_conv(x)
        return x


class _ASPPModule(nn.Module):

    def __init__(self, inplanes, planes, kernel_size, padding, dilation,
                 BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=1,
                                     padding=padding, dilation=dilation, bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()
        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)
        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPP(nn.Module):

    def __init__(self, inplanes, mid_channels=256, BatchNorm=nn.BatchNorm2d):
        super(ASPP, self).__init__()

        dilations = [1, 6, 12, 18]

        self.aspp1 = _ASPPModule(inplanes, mid_channels, 1, padding=0, dilation=dilations[0], BatchNorm=BatchNorm)
        self.aspp2 = _ASPPModule(inplanes, mid_channels, 3, padding=dilations[1], dilation=dilations[1], BatchNorm=BatchNorm)
        self.aspp3 = _ASPPModule(inplanes, mid_channels, 3, padding=dilations[2], dilation=dilations[2], BatchNorm=BatchNorm)
        self.aspp4 = _ASPPModule(inplanes, mid_channels, 3, padding=dilations[3], dilation=dilations[3], BatchNorm=BatchNorm)

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(inplanes, mid_channels, 1, stride=1, bias=False),
            BatchNorm(mid_channels),
            nn.ReLU(),
        )
        self.conv1 = nn.Conv2d(int(mid_channels * 5), mid_channels, 1, bias=False)
        self.bn1 = BatchNorm(mid_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Mlp(nn.Module):

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.ReLU,
                 drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class SELayer(nn.Module):

    def __init__(self, channels, act_layer=nn.ReLU, gate_layer=nn.Sigmoid):
        super().__init__()
        self.conv_reduce = nn.Conv2d(channels, channels, 1, bias=True)
        self.act1 = act_layer()
        self.conv_expand = nn.Conv2d(channels, channels, 1, bias=True)
        self.gate = gate_layer()

    def forward(self, x, x_se):
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)


class DepthEdgeNet(nn.Module):

    def __init__(self, in_channels, mid_channels, context_channels,
                 depth_channels):
        super(DepthEdgeNet, self).__init__()
        self.bn = nn.BatchNorm1d(27)
        self.reduce_edge_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.context_mlp = Mlp(27, mid_channels, mid_channels)
        self.context_se = SELayer(mid_channels)
        self.context_conv = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(True),
            nn.Conv2d(mid_channels, context_channels, kernel_size=1, stride=1, padding=0)
        )

        self.reduce_depth_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.depth_mlp = Mlp(27, mid_channels, mid_channels)
        self.depth_se = SELayer(mid_channels)
        self.depth_conv = nn.Sequential(
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
            ASPP(mid_channels, mid_channels),
            build_conv_layer(cfg=dict(
                type='DCN',
                in_channels=mid_channels,
                out_channels=mid_channels,
                kernel_size=3,
                padding=1,
                groups=4,
                im2col_step=128,
            )),
            nn.Conv2d(mid_channels,
                      depth_channels,
                      kernel_size=1,
                      stride=1,
                      padding=0),
        )

    def forward(self, x_depth, x_edge, mats_dict):
        batch_size, num_cams, _, _ = mats_dict['sensor2ego_mats'].shape
        intrins = mats_dict['intrin_mats']
        ida = mats_dict['ida_mats']
        sensor2ego = mats_dict['sensor2ego_mats']
        bda = mats_dict['bda_mat'].view(batch_size, 1, 4, 4).repeat(1, num_cams, 1, 1)
        mlp_input = torch.stack([
            intrins[:, :, 0, 0],
            intrins[:, :, 1, 1],
            intrins[:, :, 0, 2],
            intrins[:, :, 1, 2],
            ida[:, :, 0, 0],
            ida[:, :, 0, 1],
            ida[:, :, 0, 3],
            ida[:, :, 1, 0],
            ida[:, :, 1, 1],
            ida[:, :, 1, 3],
            bda[:, :, 0, 0],
            bda[:, :, 0, 1],
            bda[:, :, 1, 0],
            bda[:, :, 1, 1],
            bda[:, :, 2, 2], ], dim=-1)
        sensor2ego = sensor2ego[:, :, :3, :].reshape(batch_size, num_cams, -1)
        mlp_input = torch.cat([mlp_input, sensor2ego], dim=-1)
        mlp_input = self.bn(mlp_input.reshape(-1, mlp_input.shape[-1]))
        x_edge = self.reduce_edge_conv(x_edge)
        context_se = self.context_mlp(mlp_input)[..., None, None]
        context = self.context_se(x_edge, context_se)
        context = self.context_conv(context)
        x_depth = self.reduce_depth_conv(x_depth)
        depth_se = self.depth_mlp(mlp_input)[..., None, None]
        depth = self.depth_se(x_depth, depth_se)
        depth = self.depth_conv(depth)
        return torch.cat([depth, context], dim=1)


@VTRANSFORMS.register_module()
class UniLSSTransform(BaseDepthTransform):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        image_size: Tuple[int, int],
        feature_size: Tuple[int, int],
        xbound: Tuple[float, float, float],
        ybound: Tuple[float, float, float],
        zbound: Tuple[float, float, float],
        dbound: Tuple[float, float, float],
        downsample: int = 1,
        bevdepth_refine: bool = False,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            image_size=image_size,
            feature_size=feature_size,
            xbound=xbound,
            ybound=ybound,
            zbound=zbound,
            dbound=dbound,
        )
        # bevfusion
        # self.dtransform = nn.Sequential(
        #     nn.Conv2d(1, 8, 1),
        #     nn.BatchNorm2d(8),
        #     nn.ReLU(True),
        #     nn.Conv2d(8, 32, 5, stride=4, padding=2),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(True),
        #     nn.Conv2d(32, 64, 5, stride=2, padding=2),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(True),
        # )
        # self.depthnet = nn.Sequential(
        #     nn.Conv2d(in_channels + 64, in_channels, 3, padding=1),
        #     nn.BatchNorm2d(in_channels),
        #     nn.ReLU(True),
        #     nn.Conv2d(in_channels, in_channels, 3, padding=1),
        #     nn.BatchNorm2d(in_channels),
        #     nn.ReLU(True),
        #     nn.Conv2d(in_channels, self.D + self.C, 1),
        # )
        # unibevfusion
        # self.depthmapnet = MapNet(block=Map_Block, input_channel=1, nb_filter=[8, 16, 32, 64],
        #                           block_nums=[2, 2, 2])
        # self.edgemapnet = MapNet(block=Map_Block, input_channel=4, nb_filter=[8, 16, 32, 64],
        #                           block_nums=[2, 2, 2])
        self.depthmapnet = nn.Sequential(
            nn.Conv2d(1, 8, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.Conv2d(8, 32, 5, stride=4, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.edgemapnet = nn.Sequential(
            nn.Conv2d(4, 8, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.Conv2d(8, 32, 5, stride=4, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        self.depth_edge_net = DepthEdgeNet(in_channels + 64, in_channels, self.C, self.D)
        self.bevdepth_refine = bevdepth_refine
        if self.bevdepth_refine:
            self.refinement = DepthRefinement(self.C, self.C, self.C)
        if downsample > 1:
            assert downsample == 2, downsample
            self.downsample = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    3,
                    stride=downsample,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True),
            )
        else:
            self.downsample = nn.Identity()

    def get_downsampled_gt_depth(self, gt_depths, downsample=8):
        B, N, H, W = gt_depths.shape
        gt_depths = gt_depths.view(B * N, H // downsample, downsample, W // downsample, downsample, 1)
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_depths = gt_depths.view(-1, downsample * downsample)
        gt_depths_tmp = torch.where(gt_depths == 0.0, 1e5 * torch.ones_like(gt_depths), gt_depths)
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values
        gt_depths = gt_depths.view(B * N, H // downsample, W // downsample)
        gt_depths = (gt_depths - (self.dbound[0] - self.dbound[2])) / self.dbound[2]
        gt_depths = torch.where((gt_depths < self.D + 1) & (gt_depths >= 0.0), gt_depths, torch.zeros_like(gt_depths))
        gt_depths = F.one_hot(gt_depths.long(), num_classes=self.D + 1).view(-1, self.D + 1)[:, 1:].float()
        return gt_depths

    @force_fp32()
    def get_depth_loss(self, depth_labels, depth_preds, loss_depth_weight=3.0):
        depth_labels = self.get_downsampled_gt_depth(depth_labels)
        depth_preds = depth_preds.permute(0, 2, 3, 1).contiguous().view(-1, self.D)
        depth_mask = torch.max(depth_labels, dim=1).values > 0.0
        depth_labels = depth_labels[depth_mask]
        depth_preds = depth_preds[depth_mask]

        with autocast(enabled=False):
            depth_loss = F.binary_cross_entropy(
                depth_preds,
                depth_labels,
                reduction='none',
            ).sum() / max(1.0, depth_mask.sum())
        return depth_loss * loss_depth_weight

    # @force_fp32()
    # def get_cam_feats(self, x, d):
    #     B, N, C, fH, fW = x.shape
    #
    #     d = d.view(B * N, *d.shape[2:])
    #     x = x.view(B * N, C, fH, fW)
    #
    #     d = self.dtransform(d)
    #     x = torch.cat([d, x], dim=1)
    #     x = self.depthnet(x)
    #
    #     depth = x[:, : self.D].softmax(dim=1)
    #     x = depth.unsqueeze(1) * x[:, self.D: (self.D + self.C)].unsqueeze(2)
    #
    #     x = x.view(B, N, self.C, self.D, fH, fW)
    #     x = x.permute(0, 1, 3, 4, 5, 2)
    #     return x, depth

    @force_fp32()
    def get_cam_feats(self, x, d, e, mats_dict):
        B, N, C, fH, fW = x.shape

        d = d.view(B * N, *d.shape[2:])
        e = e.view(B * N, *e.shape[2:])
        x = x.view(B * N, C, fH, fW)

        d = self.depthmapnet(d)
        x_depth = torch.cat([d, x], dim=1)
        e = self.edgemapnet(e)
        x_edge = torch.cat([e, x], dim=1)
        x = self.depth_edge_net(x_depth, x_edge, mats_dict)

        if self.use_bevpool == 'bevpoolv1':
            depth = x[:, : self.D].softmax(dim=1)
            x = depth.unsqueeze(1) * x[:, self.D: (self.D + self.C)].unsqueeze(2)

            if self.bevdepth_refine:
                x = x.permute(0, 3, 1, 4, 2).contiguous()  # [n, c, d, h, w] -> [n, h, c, w, d]
                n, h, c, w, d = x.shape
                x = x.view(-1, c, w, d)
                x = self.refinement(x)
                x = x.view(n, h, c, w, d).permute(0, 2, 4, 1, 3).contiguous().float()

            x = x.view(B, N, self.C, self.D, fH, fW)
            x = x.permute(0, 1, 3, 4, 5, 2)

        elif self.use_bevpool == 'bevpoolv2':
            depth = x[:, : self.D].softmax(dim=1)
            x = x[:, self.D: (self.D + self.C)]
            depth = depth.view(B, N, self.D, fH, fW)
            x = x.view(B, N, self.C, fH, fW)

        return x, depth

    def forward(self, *args, **kwargs):
        if self.use_depth:
            x = super().forward(*args, **kwargs)
            final_x = self.downsample(x[0]), x[1]
            return final_x
        else:
            x = super().forward(*args, **kwargs)
            x = self.downsample(x)
            return x
