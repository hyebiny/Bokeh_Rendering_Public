#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

from torch import nn, Tensor
from typing import Optional, Dict, Tuple, Union
import argparse

from utils import logger

from ... import parameter_list
from ...layers import norm_layers_tuple, LinearLayer
from ...misc.profiler import module_profile
from ...misc.init_utils import initialize_weights, initialize_fc_layer
from ..classification import BaseEncoder


class BaseBokeh(nn.Module):
    """
    Base class for different bokeh rendering models
    """
    def __init__(self, opts, encoder: BaseEncoder) -> None:
        super(BaseBokeh, self).__init__()
        self.lr_multiplier = getattr(opts, "model.segmentation.lr_multiplier", 1.0)
        assert isinstance(
            encoder, BaseEncoder
        ), "encoder should be an instance of BaseEncoder"
        self.encoder: BaseEncoder = encoder


    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        """Add model-specific arguments"""
        return parser

    def update_classifier(self, opts, n_classes: int):
        """
        This function updates the classification layer in a model. Useful for finetuning purposes.
        """
        linear_init_type = getattr(opts, "model.layer.linear_init", "normal")
        if isinstance(self.classifier, nn.Sequential):
            in_features = self.classifier[-1].in_features
            layer = LinearLayer(
                in_features=in_features, out_features=n_classes, bias=True
            )
            initialize_fc_layer(layer, init_method=linear_init_type)
            self.classifier[-1] = layer
        else:
            in_features = self.classifier.in_features
            layer = LinearLayer(
                in_features=in_features, out_features=n_classes, bias=True
            )
            initialize_fc_layer(layer, init_method=linear_init_type)

            # re-init head
            head_init_scale = 0.001
            layer.weight.data.mul_(head_init_scale)
            layer.bias.data.mul_(head_init_scale)

            self.classifier = layer
        return

    @staticmethod
    def reset_layer_parameters(layer, opts) -> None:
        """Reset weights of a given layer"""
        initialize_weights(opts=opts, modules=layer.modules())

    def freeze_norm_layers(self) -> None:
        """Freeze normalization layers"""
        for m in self.modules():
            if isinstance(m, norm_layers_tuple):
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False
                m.training = False

    def get_trainable_parameters(
        self,
        weight_decay: Optional[float] = 0.0,
        no_decay_bn_filter_bias: Optional[bool] = False,
        *args,
        **kwargs
    ):
        """Get trainable parameters"""
        param_list = parameter_list(
            named_parameters=self.named_parameters,
            weight_decay=weight_decay,
            no_decay_bn_filter_bias=no_decay_bn_filter_bias,
        )
        return param_list, [1.0] * len(param_list)


