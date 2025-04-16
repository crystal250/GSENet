import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mmcv.cnn import ConvModule
from ..registry import NECKS
from .evc import BaseConv, CSPLayer, DWConv, EVCBlock


@NECKS.register_module
class FPN(nn.Module):
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
                 attention=False,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest'),
                 init_cfg=dict(type='Xavier',
                               layer='Conv2d',
                               distribution='uniform'),
                 cfg=None,
                 act='silu',
                 depthwise=False):
        super(FPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.attention = attention
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.upsample_cfg = upsample_cfg.copy()
        self.evcblock = EVCBlock(
            in_channels=  512,
            out_channels= 512,
            channel_ratio=4, base_channel=16,
            ) 
        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            if extra_convs_on_inputs:
                # TODO: deprecate `extra_convs_on_inputs`
                warnings.simplefilter('once')
                warnings.warn(
                    '"extra_convs_on_inputs" will be deprecated in v2.9.0,'
                    'Please use "add_extra_convs"', DeprecationWarning)
                self.add_extra_convs = 'on_input'
            else:
                self.add_extra_convs = 'on_output'

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        # self.l_conv_evc =ConvModule(
        #         512,
        #         64,   
        #         1,
        #         conv_cfg=conv_cfg,
        #         norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
        #         act_cfg=act_cfg,
        #         inplace=False)
        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            fpn_conv = ConvModule(out_channels,
                                  out_channels,
                                  3,
                                  padding=1,
                                  conv_cfg=conv_cfg,
                                  norm_cfg=norm_cfg,
                                  act_cfg=act_cfg,
                                  inplace=False)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(in_channels,
                                            out_channels,
                                            3,
                                            stride=2,
                                            padding=1,
                                            conv_cfg=conv_cfg,
                                            norm_cfg=norm_cfg,
                                            act_cfg=act_cfg,
                                            inplace=False)
                self.fpn_convs.append(extra_fpn_conv)
        Conv = DWConv if depthwise else BaseConv
        self.lateral_conv0 = BaseConv(
            512, 256, 1, 1, act=act
        )
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.C3_p4 = CSPLayer(
            512,
            256,
            False,
            depthwise=depthwise,
            act=act,
        )  # cat
        self.C3_p3 = CSPLayer(
            256,
            128,
            False,
            depthwise=depthwise,
            act=act,
        )
        self.reduce_conv1 = BaseConv(
            256, 128, 1, 1, act=act
        )
        self.bu_conv2 = Conv(
            128, 128, 3, 2, act=act
        )
        self.bu_conv1 = Conv(
            256, 256, 3, 2, act=act
        )
        self.C3_n2 = CSPLayer(
            128,
            64,
            False,
            depthwise=depthwise,
            act=act,
        )
        self.C3_n3 = CSPLayer(
            256,
            64,
            False,
            depthwise=depthwise,
            act=act,
        )
        self.C3_n4 = CSPLayer(
            512,
            64,
            False,
            depthwise=depthwise,
            act=act,
        )
    def forward(self, inputs):
        """
        Args:
            inputs: input images.

        Returns:
            Tuple[Tensor]: FPN feature.
        """
        #  backbone
        # out_features = self.backbone(input)
        # features = [out_features[f] for f in self.in_features]
        x0=inputs[3]
        print(np.array(x0.cpu().detach()).shape)
        x1=inputs[2]
        x2=inputs[1]

        fpn_out0 = self.lateral_conv0(x0)  #   512->256
        f_out0 = self.upsample(fpn_out0)  # 256/20/50  

        fevc_out0 = self.evcblock(f_out0)
        f_out0 = torch.cat([fevc_out0, x1], 1)  # 512/20/50
        f_out0 = self.C3_p4(f_out0)  # 256

        fpn_out1 = self.reduce_conv1(f_out0)  # 256->128
        f_out1 = self.upsample(fpn_out1)  # 128/40/100
        f_out1 = torch.cat([f_out1, x2], 1)  # 256/40/100
        pan_out2 = self.C3_p3(f_out1)  # 256->128

        p_out1 = self.bu_conv2(pan_out2)  # 128/40/100->128/20/50
        p_out1 = torch.cat([p_out1, fpn_out1], 1)  # 256/20/50
        pan_out1 = self.C3_n3(p_out1)  # 256/20/50

        p_out0 = self.bu_conv1(pan_out1)  # 256/20/50->256/10/25
        p_out0 = torch.cat([p_out0, fpn_out0], 1)  # 512/10/25
        pan_out0 = self.C3_n4(p_out0)  # 512/10/25
        # pan_out2 =self.C3_n2(pan_out2)
        
        outputs = (pan_out2, pan_out1, pan_out0)
        return outputs
    # def forward(self, inputs):
    #     """Forward function."""
    #     assert len(inputs) >= len(self.in_channels)

    #     if len(inputs) > len(self.in_channels):
    #         for _ in range(len(inputs) - len(self.in_channels)):
    #             del inputs[0]
    #     x0=inputs[2]
    #     # print(np.array(x0.detach()).shape)
    #     fevc_out0=self.evcblock(x0)
    #     fevc_out0=self.l_conv_evc(fevc_out0)
    #     # print(np.array(fevc_out0.cpu().detach()).shape)
    #     # build laterals
    #     laterals = [
    #         lateral_conv(inputs[i + self.start_level])
    #         for i, lateral_conv in enumerate(self.lateral_convs)
    #     ]
    #     fevc_out1=F.interpolate(fevc_out0,size=(20,50),**self.upsample_cfg)
    #     fevc_out2=F.interpolate(fevc_out0,size=(40,100),**self.upsample_cfg)
    #     # for i in range(len(laterals)-1,0,-1):
    #     #     fevc_out0=F.interpolate(
    #     #         fevc_out0,
    #     #         size=laterals[i-1].shape[2:],
    #     #         **self.upsample_cfg
    #     #     )
    #     # print(np.array(laterals[2].cpu().detach()).shape)
    #     laterals[2] =torch.cat((laterals[2],fevc_out0),dim=1)
    #     laterals[1]=torch.cat((laterals[1],fevc_out1),dim=1)
    #     laterals[0]=torch.cat((laterals[0],fevc_out2),dim=1)
    #     # build top-down path
    #     used_backbone_levels = len(laterals)
    #     for i in range(used_backbone_levels - 1, 0, -1):
    #         # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
    #         #  it cannot co-exist with `size` in `F.interpolate`.
    #         if 'scale_factor' in self.upsample_cfg:
    #             laterals[i - 1] += F.interpolate(laterals[i],
    #                                              **self.upsample_cfg)
    #         else:
    #             prev_shape = laterals[i - 1].shape[2:]
    #             laterals[i - 1] += F.interpolate(laterals[i],
    #                                              size=prev_shape,
    #                                              **self.upsample_cfg)
    #             print(np.array(laterals[i-1].cpu().detach()).shape)
    #     # build outputs
    #     # part 1: from original levels
    #     outs = [
    #         self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
    #     ]
    #     # part 2: add extra levels
    #     if self.num_outs > len(outs):
    #         # use max pool to get more levels on top of outputs
    #         # (e.g., Faster R-CNN, Mask R-CNN)
    #         if not self.add_extra_convs:
    #             for i in range(self.num_outs - used_backbone_levels):
    #                 outs.append(F.max_pool2d(outs[-1], 1, stride=2))
    #         # add conv layers on top of original feature maps (RetinaNet)
    #         else:
    #             if self.add_extra_convs == 'on_input':
    #                 extra_source = inputs[self.backbone_end_level - 1]
    #             elif self.add_extra_convs == 'on_lateral':
    #                 extra_source = laterals[-1]
    #             elif self.add_extra_convs == 'on_output':
    #                 extra_source = outs[-1]
    #             else:
    #                 raise NotImplementedError
    #             outs.append(self.fpn_convs[used_backbone_levels](extra_source))
    #             for i in range(used_backbone_levels + 1, self.num_outs):
    #                 if self.relu_before_extra_convs:
    #                     outs.append(self.fpn_convs[i](F.relu(outs[-1])))
    #                 else:
    #                     outs.append(self.fpn_convs[i](outs[-1]))
    #     return tuple(outs)
