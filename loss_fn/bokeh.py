#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.
#

from torch import Tensor
import argparse
from utils import logger
from typing import Any

from . import BaseCriteria, register_loss_fn
from .bokeh_loss_fns import MSSSIM, MS_SSIM_L1_LOSS, SSIM, SUPPORTED_BOK_LOSS_FNS


@register_loss_fn("bokeh_loss_fns")
class BokehLoss(BaseCriteria):
    def __init__(self, opts):
        loss_fn_name = getattr(opts, "loss.bokeh.name", "msssim")
        super(BokehLoss, self).__init__()
        if loss_fn_name == "msssim":
            self.criteria = MSSSIM(opts=opts)
        elif loss_fn_name == "L1_msssim":
            self.criteria = MS_SSIM_L1_LOSS(opts=opts)
        elif loss_fn_name == "ssim":
            self.criteria = SSIM(opts=opts)
        else:
            temp_str = (
                "Loss function ({}) not yet supported. "
                "\n Supported bokeh_loss_fns loss functions are:".format(loss_fn_name)
            )
            for i, m_name in enumerate(SUPPORTED_BOK_LOSS_FNS):
                temp_str += "\n\t {}: {}".format(i, logger.color_text(m_name))
            logger.error(temp_str)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group(
            title="".format(cls.__name__), description="".format(cls.__name__)
        )
        group.add_argument(
            "--loss.bokeh.name",
            type=str,
            default="msssim",
            help="Bokeh loss function name",
        )
        parser = MSSSIM.add_arguments(parser=parser)
        return parser

    def forward(
        self, input_sample: Any, prediction: Any, target: Any, *args, **kwargs
    ) -> Tensor:
        # in msssim, i will compare input_sample & prediction
        # return self.criteria(img1=input_sample, img2=prediction)
        return self.criteria(input_sample, prediction)

    def __repr__(self):
        return self.criteria.__repr__()
