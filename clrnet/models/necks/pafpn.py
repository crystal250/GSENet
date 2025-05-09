import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import auto_fp16
import numpy as np
import torch
from ..registry import NECKS
from .fpn import FPN


@NECKS.register_module
class PAFPN(FPN):
    """Path Aggregation Network for Instance Segmentation.

    This is an implementation of the `PAFPN in Path Aggregation Network
    <https://arxiv.org/abs/1803.01534>`_.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Default: -1, which means the last level.
        add_extra_convs (bool): Whether to add conv layers on top of the
            original feature maps. Default: False.
        extra_convs_on_inputs (bool): Whether to apply extra conv on
            the original feature from the backbone. Default: False.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Default: False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Default: False.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (str): Config dict for activation layer in ConvModule.
            Default: None.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 cfg=None,
                 attention=False):
        super(PAFPN, self).__init__(in_channels,
                                    out_channels,
                                    num_outs,
                                    start_level,
                                    end_level,
                                    add_extra_convs,
                                    extra_convs_on_inputs,
                                    relu_before_extra_convs,
                                    no_norm_on_lateral,
                                    conv_cfg,
                                    norm_cfg,
                                    attention,
                                    act_cfg,
                                    cfg=cfg)
        # add extra bottom up pathway
        self.downsample_convs = nn.ModuleList()
        self.pafpn_convs = nn.ModuleList()
        for i in range(self.start_level + 1, self.backbone_end_level):
            d_conv = ConvModule(out_channels,
                                out_channels,
                                3,
                                stride=2,
                                padding=1,
                                conv_cfg=conv_cfg,
                                norm_cfg=norm_cfg,
                                act_cfg=act_cfg,
                                inplace=False)
            pafpn_conv = ConvModule(out_channels,
                                    out_channels,
                                    3,
                                    padding=1,
                                    conv_cfg=conv_cfg,
                                    norm_cfg=norm_cfg,
                                    act_cfg=act_cfg,
                                    inplace=False)
            self.downsample_convs.append(d_conv)
            self.pafpn_convs.append(pafpn_conv)

    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) >= len(self.in_channels)

        if len(inputs) > len(self.in_channels):
            for _ in range(len(inputs) - len(self.in_channels)):
                del inputs[0]

        x0=inputs[2]
        # print(np.array(x0.detach()).shape)
        fevc_out0=self.evcblock(x0)
        fevc_out0=self.l_conv_evc(fevc_out0)
        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        fevc_out1=F.interpolate(fevc_out0,size=(20,50),**self.upsample_cfg)
        fevc_out2=F.interpolate(fevc_out0,size=(40,100),**self.upsample_cfg)
        laterals[2] =torch.cat((laterals[2],fevc_out0),dim=1)
        laterals[1]=torch.cat((laterals[1],fevc_out1),dim=1)
        laterals[0]=torch.cat((laterals[0],fevc_out2),dim=1)
        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] += F.interpolate(laterals[i],
                                             size=prev_shape,
                                             mode='nearest')

        # build outputs
        # part 1: from original levels
        inter_outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]

        # part 2: add bottom-up path
        for i in range(0, used_backbone_levels - 1):
            inter_outs[i + 1] += self.downsample_convs[i](inter_outs[i])

        outs = []
        outs.append(inter_outs[0])
        print(np.array(outs[0].cpu().detach()).shape)

        outs.extend([
            self.pafpn_convs[i - 1](inter_outs[i])
            for i in range(1, used_backbone_levels)
        ])
        print(np.array(outs[0].cpu().detach()).shape)
        print(np.array(outs[1].cpu().detach()).shape)
        print(np.array(outs[2].cpu().detach()).shape)
        # part 3: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    orig = inputs[self.backbone_end_level - 1]
                    outs.append(self.fpn_convs[used_backbone_levels](orig))
                elif self.add_extra_convs == 'on_lateral':
                    outs.append(self.fpn_convs[used_backbone_levels](
                        laterals[-1]))
                elif self.add_extra_convs == 'on_output':
                    outs.append(self.fpn_convs[used_backbone_levels](outs[-1]))
                else:
                    raise NotImplementedError
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)
