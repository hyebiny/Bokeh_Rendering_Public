#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

import torch
from torch import nn, Tensor
from typing import Optional, Dict, Tuple
import argparse

from utils import logger

from ....misc.common import parameter_list
from ....misc.init_utils import initialize_weights
from ....layers import ConvLayer, Dropout2d, UpSample

from . import register_bokeh_head
from . import BaseBokHead

@register_bokeh_head(name="dpt")
class dpt(BaseBokHead):
    """
    dpt class for bokeh heads
    """

    def __init__(self, opts, enc_conf: dict):
        super(dpt, self).__init__(opts=opts, enc_conf=enc_conf)

        # features = 128
        features = getattr(opts, "model.bokeh.dpt.features", 128)
        # use_bn = False
        use_bn = getattr(opts, "model.bokeh.dpt.use_bn", False)

        enc_ch_l5_exp_out = _check_out_channels(enc_conf, "exp_before_cls")
        enc_ch_l5_out = _check_out_channels(enc_conf, "layer5")
        enc_ch_l4_out = _check_out_channels(enc_conf, "layer4")
        enc_ch_l3_out = _check_out_channels(enc_conf, "layer3")
        enc_ch_l2_out = _check_out_channels(enc_conf, "layer2")
        enc_ch_l1_out = _check_out_channels(enc_conf, "layer1")

        self.enc_l5_exp_channels = enc_ch_l5_exp_out
        self.enc_l5_channels = enc_ch_l5_out
        self.enc_l4_channels = enc_ch_l4_out
        self.enc_l3_channels = enc_ch_l3_out
        self.enc_l2_channels = enc_ch_l2_out
        self.enc_l1_channels = enc_ch_l1_out

        self.scratch = _make_scratch([128, 256, 384, 512], features, groups=1, expand=False ) # features = 128
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn, True)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn, True)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn, True)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn, False)

        self.scratch.output_conv = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
            nn.Identity(),
        )

    def forward(self, enc_out: Dict) -> Tensor : # or Tuple[Tensor]:

        layer_1 = self.enc_l1_channels
        layer_2 = self.enc_l2_channels
        layer_3 = self.enc_l3_channels
        layer_4 = self.enc_l4_channels

        layer_1 = enc_out['out_l2']
        layer_2 = enc_out['out_l3']
        layer_3 = enc_out['out_l4']
        layer_4 = enc_out['out_l5']

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        out = self.scratch.output_conv(path_1)

        print('----------- hyebin in bokeh/heads/dpt.py')
        print(layer_1.shape)
        print(layer_2.shape)
        print(layer_3.shape)
        print(layer_4.shape)
        print(layer_1_rn.shape)
        print(layer_2_rn.shape)
        print(layer_3_rn.shape)
        print(layer_4_rn.shape)
        print(path_4.shape)
        print(path_3.shape)
        print(path_2.shape)
        print(path_1.shape)
        print(out.shape)

        return out

    def reset_head_parameters(self, opts) -> None:
        # weight initialization
        initialize_weights(opts=opts, modules=self.modules())


    @classmethod
    def add_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Add bokeh head specific arguments"""
        group = parser.add_argument_group(
            title="Bokeh head arguments",
            description="Bokeh head arguments",
        )
        group.add_argument(
            "--model.bokeh.dpt.features",
            type=int,
            default=128,
            help="feature size",
        )
        group.add_argument(
            "--model.bokeh.dpt.use_bn",
            action="store_true",
            default=False,
            help="use_bn",
        )

        return parser

    def profile_module(self, x: Tensor) -> Tuple[Tensor, float, float]:
        """
        Child classes must implement this function to compute FLOPs and parameters
        """
        raise NotImplementedError

    def get_trainable_parameters(
        self, weight_decay: float = 0.0, no_decay_bn_filter_bias: bool = False
    ):
        param_list = parameter_list(
            named_parameters=self.named_parameters,
            weight_decay=weight_decay,
            no_decay_bn_filter_bias=no_decay_bn_filter_bias,
        )
        return param_list, [1.0] * len(param_list)


def _check_out_channels(config: dict, layer_name: str) -> int:
    enc_ch_l: dict = config.get(layer_name, None)
    if enc_ch_l is None or not enc_ch_l:
        logger.error(
            "Encoder does not define input-output mapping for {}: Got: {}".format(
                layer_name, config
            )
        )

    enc_ch_l_out = enc_ch_l.get("out", None)
    if enc_ch_l_out is None or not enc_ch_l_out:
        logger.error(
            "Output channels are not defined in {} of the encoder. Got: {}".format(
                layer_name, enc_ch_l
            )
        )
    return enc_ch_l_out


def _make_fusion_block(features, use_bn, double):
    return FeatureFusionBlock_custom(
        features,
        nn.ReLU(False),
        double,
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True
    )

def _make_scratch(in_shape, out_shape, groups=1, expand=False):
    scratch = nn.Module()

    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    out_shape4 = out_shape
    if expand == True:
        out_shape1 = out_shape
        out_shape2 = out_shape * 2
        out_shape3 = out_shape * 3 # 4
        out_shape4 = out_shape * 4 # 8

    scratch.layer1_rn = nn.Conv2d(
        in_shape[0],
        out_shape1,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )
    scratch.layer2_rn = nn.Conv2d(
        in_shape[1],
        out_shape2,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )
    scratch.layer3_rn = nn.Conv2d(
        in_shape[2],
        out_shape3,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )
    scratch.layer4_rn = nn.Conv2d(
        in_shape[3],
        out_shape4,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )

    return scratch


class FeatureFusionBlock_custom(nn.Module):
    """Feature fusion block."""

    def __init__(
        self,
        features,
        activation,
        double,
        deconv=False,
        bn=False,
        expand=False,
        align_corners=True,
    ):
        """Init.

        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock_custom, self).__init__()

        self.deconv = deconv
        self.align_corners = align_corners

        self.groups = 1

        self.expand = expand
        out_features = features
        if self.expand == True:
            out_features = features // 2

        self.out_conv = nn.Conv2d(
            features,
            out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
            groups=1,
        )

        self.resConfUnit1 = ResidualConvUnit_custom(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit_custom(features, activation, bn)

        self.skip_add = nn.quantized.FloatFunctional()
        self.double = double

    def forward(self, *xs):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)
            # output += res

        output = self.resConfUnit2(output)
        if self.double:
            output = nn.functional.interpolate(
                output, scale_factor=2, mode="bilinear", align_corners=self.align_corners
            )
        output = self.out_conv(output)

        return output


class ResidualConvUnit_custom(nn.Module):
    """Residual convolution module."""

    def __init__(self, features, activation, bn):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()

        self.bn = bn

        self.groups = 1

        self.conv1 = nn.Conv2d(
            features,
            features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=not self.bn,
            groups=self.groups,
        )

        self.conv2 = nn.Conv2d(
            features,
            features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=not self.bn,
            groups=self.groups,
        )

        if self.bn == True:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)

        self.activation = activation

        self.skip_add = nn.quantized.FloatFunctional()


class Interpolate(nn.Module):
    """Interpolation module."""

    def __init__(self, scale_factor, mode, align_corners=False):
        """Init.

        Args:
            scale_factor (float): scaling
            mode (str): interpolation mode
        """
        super(Interpolate, self).__init__()

        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: interpolated data
        """

        x = self.interp(
            x,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
        )

        return x